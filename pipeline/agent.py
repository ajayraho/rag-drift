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

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "45"))

SYSTEM_PROMPT = """You are a helpful assistant answering questions about \
GitHub issues in the {repo} repository. Use the search_github_issues tool \
to find relevant issues before answering. Call the tool AT MOST TWICE total \
-- if a search (including a state-filtered one) doesn't turn up what you \
need, answer honestly with what you found rather than retrying with \
different phrasings. If the search doesn't turn up anything relevant, say \
so honestly instead of making something up. When you reference an issue, \
cite its issue number and URL."""


def retrieve_issues(query: str, n_results: int = 5, state: str | None = None) -> list[dict]:
    collection = get_chroma_collection()
    where = {"state": state} if state in ("open", "closed") else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where)

    if not results["ids"] or not results["ids"][0]:
        return []

    issues = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        issues.append({
            "number": meta["number"],
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
    return _format_issues_for_llm(retrieve_issues(query, state=state))


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and set it in your .env file."
        )
    return key


@lru_cache(maxsize=1)
def get_agent(repo: str | None = None):
    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")
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

    return create_agent(
        model=llm,
        tools=[search_github_issues],
        system_prompt=SYSTEM_PROMPT.format(repo=repo),
    )


def ask_stream(question: str, repo: str | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
    """
    Generator yielding step events as the agent works, then a final
    result or error event. Shape of yielded dicts:
      {"type": "step", "node": "model"|"tools", "detail": "..."}
      {"type": "final", "answer": "...", "sources": [...]}
      {"type": "error", "detail": "..."}

    Runs the actual agent work in a background thread so we can enforce
    a real wall-clock deadline (see module docstring) -- if nothing
    arrives within timeout_seconds, we yield an error and stop, even if
    the underlying LLM call is still stuck retrying.
    """
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
            sources = retrieve_issues(question)
            event_queue.put({"type": "final", "answer": answer, "sources": sources})
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
    result = ask(q, verbose=True)
    print("\n--- ANSWER ---")
    print(result["answer"])