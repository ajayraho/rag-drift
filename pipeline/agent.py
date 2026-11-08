"""
The RAG agent. Built with LangChain's current (v1.0+) agent API.

ask_stream() runs the agent in a background thread and yields step
events through a queue as they happen -- this is what powers the live
"what's the agent doing right now" trace in the UI, and it's also what
lets us enforce our OWN hard timeout. ChatGoogleGenerativeAI's own
timeout/max_retries params are documented but known to be unreliably
enforced (open bugs in langchain-google-genai as of writing) -- a 503
can trigger internal SDK retries that hang for minutes regardless of
what we pass in. Running in a thread with a real deadline means we
always return a clean answer or a clean error within a bounded time,
even if the underlying SDK call is still stuck retrying in the
background (that thread is a daemon, so it's abandoned, not killed --
acceptable for this project's scope; a production system would need a
real cancellable request instead).

Also implements a simple in-memory response cache: an exact repeat of
the same question (+ repo) within CACHE_TTL_SECONDS skips retrieval and
generation entirely, returning the cached answer immediately -- a real,
if modest, token/cost optimization, not just rate-limit pacing. It's
process-local and exact-match (no semantic similarity matching, no
cross-instance sharing) -- explicitly scoped that way for this
project; a production deployment would want a shared cache (Redis) and
ideally semantic matching, plus invalidation on re-ingestion rather
than pure TTL expiry.

IMPORTANT: get_agent(repo) is cached per-repo, and each agent gets its
OWN search tool bound to that repo via a closure in _build_search_tool.
This is deliberate: the LLM is never trusted to correctly pass a `repo`
argument on every tool call -- if it forgot even once, or the UI's repo
field didn't match exactly, retrieval would silently search across every
ingested repo's issues at once. Binding repo at agent-build time closes
that off entirely; the tool physically cannot search outside its repo.
"""

import os
import queue
import sys
import threading
import time
from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline.embed import get_chroma_collection
from pipeline.log_utils import extract_text, get_run_logger

log = get_run_logger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "45"))

SYSTEM_PROMPT_SINGLE_REPO = """You are a helpful assistant answering questions about \
GitHub issues in the {repo} repository. Use the search_github_issues tool \
to find relevant issues before answering. Call the tool AT MOST TWICE total \
-- if a search (including a state-filtered one) doesn't turn up what you \
need, answer honestly with what you found rather than retrying with \
different phrasings. If the search doesn't turn up anything relevant, say \
so honestly instead of making something up. When you reference an issue, \
cite its issue number, its repo, and its URL."""

SYSTEM_PROMPT_ALL_REPOS = """You are a helpful assistant answering questions \
about GitHub issues across every repository that's been ingested into this \
system. Use the search_github_issues tool to find relevant issues before \
answering -- results may come from different repos, so always state which \
repo each issue belongs to when you reference it. Call the tool AT MOST \
TWICE total -- if a search doesn't turn up what you need, answer honestly \
with what you found rather than retrying with different phrasings. If the \
search doesn't turn up anything relevant, say so honestly instead of making \
something up."""


# --- Response cache --------------------------------------------------------
# Small, real, in-process. See module docstring for the honest scoping notes.
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "1800"))
_response_cache: dict[tuple, tuple[dict, float]] = {}


def _cache_key(question: str, repo: str | None) -> tuple:
    normalized_question = " ".join(question.strip().lower().split())
    return (normalized_question, repo or "__all_repos__")


def _get_cached(key: tuple) -> dict | None:
    entry = _response_cache.get(key)
    if entry is None:
        return None
    result, cached_at = entry
    if time.monotonic() - cached_at > CACHE_TTL_SECONDS:
        del _response_cache[key]
        return None
    return result


def _store_cache(key: tuple, result: dict) -> None:
    _response_cache[key] = (result, time.monotonic())


def _build_where(state: str | None, repo: str | None) -> dict | None:
    conditions = []
    if state in ("open", "closed"):
        conditions.append({"state": state})
    if repo:
        conditions.append({"repo": repo})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_issues(query: str, n_results: int = 5, state: str | None = None, repo: str | None = None) -> list[dict]:
    """
    repo filters via Chroma's `where` clause -- without this, search runs
    across EVERY repo ever ingested into the same collection, mixed
    together. That was a real bug here: questions about one repo were
    getting answered (and cited!) using another repo's issues entirely,
    because nothing constrained retrieval to the repo the user was
    actually asking about.
    """
    where = _build_where(state, repo)
    try:
        results = get_chroma_collection().query(query_texts=[query], n_results=n_results, where=where)
    except Exception as e:
        # The collection handle is cached process-wide (see embed.py); if
        # Chroma restarted or the collection was recreated it can be stale.
        # Reconnect once, then let a second failure propagate normally.
        log.warning("Chroma query failed (%s); reconnecting and retrying once", e)
        results = get_chroma_collection(fresh=True).query(query_texts=[query], n_results=n_results, where=where)

    if not results["ids"] or not results["ids"][0]:
        return []

    issues = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        issues.append({
            "number": meta["number"],
            "repo": meta["repo"],
            "state": meta["state"],
            "labels": meta["labels"],
            "url": meta["url"],
            "snippet": doc[:300] + ("..." if len(doc) > 300 else ""),
        })
    return issues


def _format_issues_for_llm(issues: list[dict]) -> str:
    if not issues:
        return "No relevant issues found."
    return "\n\n---\n\n".join(
        f"Issue #{i['number']} [{i['state']}] (labels: {i['labels'] or 'none'})\n"
        f"URL: {i['url']}\n"
        f"Content: {i['snippet']}"
        for i in issues
    )


def _build_search_tool(repo: str):
    """
    Returns a search_github_issues tool BOUND to one specific repo via
    closure. The LLM still supplies query/state, but repo is fixed at
    creation time -- not an LLM-controllable argument -- so there's no
    path by which a tool call can accidentally search a different repo.
    """

    @tool
    def search_github_issues(query: str, state: str | None = None) -> str:
        """Search the GitHub issues corpus for issues relevant to the query.
        Returns the top matching issues with their title, state, labels, and URL.

        Args:
            query: the semantic search text (topic/keywords), WITHOUT filter
                syntax like "state:open" -- use the separate `state` arg for that.
            state: optional filter, either "open" or "closed". Leave unset to
                search across both states.
        """
        return _format_issues_for_llm(retrieve_issues(query, state=state, repo=repo))

    return search_github_issues


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and set it in your .env file."
        )
    return key


@lru_cache(maxsize=8)
def get_agent(repo: str | None = None):
    api_key = _get_api_key()

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=api_key,
        # Best-effort -- see module docstring on why we don't rely on these
        # alone to bound wait time.
        timeout=30,
        max_retries=1,
    )

    system_prompt = SYSTEM_PROMPT_SINGLE_REPO.format(repo=repo) if repo else SYSTEM_PROMPT_ALL_REPOS

    return create_agent(
        model=llm,
        tools=[_build_search_tool(repo)],
        system_prompt=system_prompt,
    )


def ask_stream(question: str, repo: str | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
    """
    Generator yielding step events as the agent works, then a final
    result or error event. Shape of yielded dicts:
      {"type": "step", "node": "model"|"tools"|"cache", "detail": "..."}
      {"type": "final", "answer": "...", "sources": [...], "cached": bool}
      {"type": "error", "detail": "..."}

    repo=None means search across every ingested repo -- this is a
    deliberate mode (the UI's repo field, left empty, triggers it), not
    a missing value we should silently paper over with a default.

    Checks the response cache FIRST, before spawning any work -- an exact
    repeat of the same question skips retrieval and generation entirely,
    not just avoiding a slow response but avoiding the LLM call altogether.
    """
    cache_key = _cache_key(question, repo)
    cached = _get_cached(cache_key)
    if cached is not None:
        yield {
            "type": "step", "node": "cache",
            "detail": "Exact match found in the response cache -- skipping retrieval and generation entirely.",
        }
        yield {"type": "final", "answer": cached["answer"], "sources": cached["sources"], "cached": True}
        return

    event_queue: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def worker():
        try:
            graph = get_agent(repo)
            inputs = {"messages": [{"role": "user", "content": question}]}
            config = {"recursion_limit": 10}
            last_node_output = None

            for chunk in graph.stream(inputs, stream_mode="updates", config=config):
                for node_name, node_output in chunk.items():
                    for msg in node_output.get("messages", []):
                        if node_name == "model":
                            tool_calls = getattr(msg, "tool_calls", None)
                            if tool_calls:
                                for tc in tool_calls:
                                    q = tc.get("args", {}).get("query", "")
                                    state = tc.get("args", {}).get("state")
                                    detail = f'Searching issues for "{q}"' + (f" (state={state})" if state else "")
                                    event_queue.put({"type": "step", "node": "model", "detail": detail})
                            else:
                                event_queue.put({"type": "step", "node": "model", "detail": "Composing the final answer"})
                        elif node_name == "tools":
                            event_queue.put({"type": "step", "node": "tools", "detail": "Retrieved results, reasoning over them"})
                    last_node_output = node_output

            answer = extract_text(last_node_output["messages"][-1].content)
            sources = retrieve_issues(question, repo=repo)

            _store_cache(cache_key, {"answer": answer, "sources": sources})
            event_queue.put({"type": "final", "answer": answer, "sources": sources, "cached": False})
        except Exception as e:
            log.exception("Agent worker failed")
            event_queue.put({"type": "error", "detail": str(e)})
        finally:
            event_queue.put(_SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield {
                "type": "error",
                "detail": f"Timed out after {timeout_seconds}s -- the model provider may be "
                          f"experiencing high demand. Please try again in a bit.",
            }
            return
        try:
            item = event_queue.get(timeout=remaining)
        except queue.Empty:
            yield {
                "type": "error",
                "detail": f"Timed out after {timeout_seconds}s -- the model provider may be "
                          f"experiencing high demand. Please try again in a bit.",
            }
            return
        if item is _SENTINEL:
            return
        yield item


def ask(question: str, repo: str | None = None, verbose: bool = False) -> dict:
    """Blocking wrapper for CLI/non-streaming callers -- drains ask_stream()."""
    for event in ask_stream(question, repo):
        if verbose and event["type"] == "step":
            print(f"--- {event['node']}: {event['detail']}")
        if event["type"] == "final":
            return {"answer": event["answer"], "sources": event["sources"]}
        if event["type"] == "error":
            raise RuntimeError(event["detail"])
    raise RuntimeError("Agent stream ended without a final result.")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are some open issues related to embeddings?"
    default_repo = os.environ.get("GITHUB_REPO", "langchain-ai/langchain")
    result = ask(q, repo=default_repo, verbose=True)
    print("\n--- ANSWER ---")
    print(result["answer"])