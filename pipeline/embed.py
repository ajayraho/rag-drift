"""
Incremental embedding pipeline.

Reads issues from Postgres where embedded=false (new issues, or issues
whose content changed since last embed -- see db.upsert_issue's
content_hash logic from Day 1), embeds only those with a local
sentence-transformers model, and upserts them into a persistent
ChromaDB collection. This is deliberately NOT a full re-embed every
run -- that's the expensive mistake this project exists to avoid.

Runnable standalone: `python -m pipeline.embed`
"""

import os
import threading
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from pipeline.db import get_connection, get_unembedded_issues, mark_embedded
from pipeline.log_utils import get_run_logger

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
COLLECTION_NAME = "github_issues"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 200


_collection_lock = threading.Lock()
_collection = None


@lru_cache(maxsize=1)
def get_embedding_function():
    """
    Loaded once per process. Building SentenceTransformerEmbeddingFunction
    loads the model weights from disk (~90MB for MiniLM, far more for the
    bigger models on the roadmap) -- this used to happen on EVERY
    retrieval call, i.e. every agent tool call, every eval question and
    every /repos request.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_chroma_collection(fresh: bool = False):
    """
    Returns a process-wide cached collection handle (HttpClient + the
    cached embedding function above).

    The docstring here used to claim this was cached while the function
    built a brand-new client and reloaded the embedding model on every
    call (lru_cache was imported but never applied). It's cached now, for
    real, behind a lock so concurrent API threads don't race to load the
    model twice.

    fresh=True drops the cached handle and reconnects. A cached handle
    can go stale if the Chroma server restarts or the collection is
    deleted and recreated (its internal id changes) -- callers like
    agent.retrieve_issues retry once with fresh=True on failure instead
    of failing every request until the process restarts.
    """
    global _collection
    with _collection_lock:
        if _collection is None or fresh:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=get_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
        return _collection


def issue_to_document(issue: dict) -> tuple[str, dict]:
    """
    Builds the text that actually gets embedded, plus metadata Chroma
    stores alongside the vector (used for filtering/display at query time,
    not for the embedding itself).
    """
    title = issue["title"] or ""
    body = issue["body"] or ""
    document_text = f"{title}\n\n{body}".strip()

    metadata = {
        "repo": issue["repo"],
        "number": issue["number"],
        "state": issue["state"],
        "labels": ",".join(issue["labels"]) if issue["labels"] else "",
        "url": f"https://github.com/{issue['repo']}/issues/{issue['number']}",
        "updated_at": issue["updated_at"].isoformat(),
    }
    return document_text, metadata


def run_embedding(batch_size: int = BATCH_SIZE) -> dict:
    log = get_run_logger(__name__)
    collection = get_chroma_collection()

    total_embedded = 0

    with get_connection() as conn:
        while True:
            issues = get_unembedded_issues(conn, limit=batch_size)
            if not issues:
                break

            ids = [str(issue["id"]) for issue in issues]
            documents = []
            metadatas = []
            for issue in issues:
                doc_text, meta = issue_to_document(issue)
                documents.append(doc_text)
                metadatas.append(meta)

            # upsert (not add): if an issue is being re-embedded because its
            # content changed, this overwrites the old vector instead of
            # creating a duplicate entry under the same id.
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

            mark_embedded(conn, [issue["id"] for issue in issues])

            total_embedded += len(issues)
            log.info("Embedded batch of %s issues (running total=%s)", len(issues), total_embedded)

            if len(issues) < batch_size:
                break

    result = {
        "embedded_count": total_embedded,
        "collection_count": collection.count(),
    }
    log.info("Embedding run complete: %s", result)
    return result


if __name__ == "__main__":
    run_embedding()