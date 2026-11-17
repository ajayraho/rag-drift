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
import re
import threading
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from pipeline.db import get_comments_for_issues, get_connection, get_unembedded_issues, mark_embedded
from pipeline.log_utils import get_run_logger

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
# Configurable so a better model can be compared on the golden set.
# Each model gets its own collection (vectors from different models
# can't be mixed); switching = set the env var + `python -m pipeline.embed --reembed`.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# CHUNKING=0 rebuilds the original "one vector per issue (title + raw body)"
# index in its own collection -- only to measure the baseline on the
# current golden set. The chunked collection is left untouched.
CHUNKING = os.environ.get("CHUNKING", "1") == "1"
COLLECTION_NAME = ("issue_chunks_" if CHUNKING else "issue_whole_") + re.sub(
    r"[^a-z0-9]+", "_", EMBEDDING_MODEL.lower()).strip("_")
BATCH_SIZE = 100
CHUNK_WORDS = 150  # ~200 tokens: fits MiniLM's 256-token window with the title prefix


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


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_CHECKBOX_LINE = re.compile(r"^\s*[-*]\s*\[[ xX]\].*$", re.M)


def clean_text(text: str | None) -> str:
    """
    Removes issue-template noise before embedding: hidden <!-- guidance -->
    blocks and "- [x] I searched existing issues" checklists. Code blocks
    and stack traces are kept -- they're often what people search for.
    """
    text = _HTML_COMMENT.sub("", text or "")
    text = _CHECKBOX_LINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _windows(words: list[str], size: int) -> list[list[str]]:
    return [words[i:i + size] for i in range(0, len(words), size)] or [[]]


def build_chunks(issue: dict, comments: list[dict]) -> list[str]:
    """
    One issue thread -> text chunks of <= CHUNK_WORDS words:
      * the body, split into windows
      * human comments (bots skipped), packed together in order until a
        chunk is full; a long comment is split into windows
    Every chunk starts with the issue title, so a chunk from the middle of
    a thread still says what the thread is about.
    """
    if not CHUNKING:
        return [f"{issue['title'] or ''}\n\n{issue['body'] or ''}".strip()]
    header = f"Issue #{issue['number']}: {issue['title']}\n\n"
    chunks = [header + " ".join(w) for w in _windows(clean_text(issue["body"]).split(), CHUNK_WORDS)]

    buffer: list[str] = []
    for c in comments:
        if c["is_bot"]:
            continue
        words = clean_text(c["body"]).split()
        if not words:
            continue
        words = [f"{c['author']}:"] + words
        for piece in _windows(words, CHUNK_WORDS):
            if buffer and len(buffer) + len(piece) > CHUNK_WORDS:
                chunks.append(header + " ".join(buffer))
                buffer = []
            buffer.extend(piece)
    if buffer:
        chunks.append(header + " ".join(buffer))
    return chunks


def issue_metadata(issue: dict) -> dict:
    return {
        "issue_id": issue["id"],
        "repo": issue["repo"],
        "number": issue["number"],
        "state": issue["state"],
        "labels": ",".join(issue["labels"]) if issue["labels"] else "",
        "url": f"https://github.com/{issue['repo']}/issues/{issue['number']}",
        "updated_at": issue["updated_at"].isoformat(),
    }


def run_embedding(batch_size: int = BATCH_SIZE) -> dict:
    log = get_run_logger(__name__)
    collection = get_chroma_collection()
    total_issues = total_chunks = 0

    with get_connection() as conn:
        while True:
            issues = get_unembedded_issues(conn, limit=batch_size)
            if not issues:
                break
            comments = get_comments_for_issues(conn, [(i["repo"], i["number"]) for i in issues])

            ids, documents, metadatas = [], [], []
            for issue in issues:
                # Drop this issue's old chunks first: if a thread now yields
                # fewer chunks, upsert alone would leave stale ones behind.
                collection.delete(where={"issue_id": issue["id"]})
                meta = issue_metadata(issue)
                for idx, text in enumerate(build_chunks(issue, comments[(issue["repo"], issue["number"])])):
                    ids.append(f"{issue['id']}:{idx}")
                    documents.append(text)
                    metadatas.append({**meta, "chunk": idx})

            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            mark_embedded(conn, [i["id"] for i in issues])
            total_issues += len(issues)
            total_chunks += len(ids)
            log.info("Embedded %s issues as %s chunks (running total=%s issues)", len(issues), len(ids), total_issues)
            if len(issues) < batch_size:
                break

    result = {"embedded_count": total_issues, "chunk_count": total_chunks, "collection_count": collection.count()}
    log.info("Embedding run complete: %s", result)
    return result


def reset_for_reembed() -> int:
    """Marks every issue unembedded -- run after changing EMBEDDING_MODEL."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE issues SET embedded = false")
        return cur.rowcount


if __name__ == "__main__":
    import sys

    if "--reembed" in sys.argv:
        print(f"Marked {reset_for_reembed()} issues for re-embedding into {COLLECTION_NAME}")
    print(run_embedding())