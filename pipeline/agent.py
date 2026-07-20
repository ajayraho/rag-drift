"""
The RAG agent. Built with LangChain's current (v1.0+) agent API:
create_agent() replaces the older AgentExecutor / create_tool_calling_agent
pattern, and is built on LangGraph under the hood -- a single function
that wires a model, tools, and system prompt into a tool-calling loop.

Unlike a plain "always retrieve, then always answer" chain, this is a
genuine agent: Gemini decides whether it needs to search the issue
corpus at all, and can call the tool multiple times with different
queries if one search isn't enough.

Runnable standalone for testing: `python -m pipeline.agent "your question"`
"""

import os
import sys
from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline.embed import get_chroma_collection
from pipeline.log_utils import extract_text, get_run_logger

log = get_run_logger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are a helpful assistant answering questions about \
GitHub issues in the {repo} repository. Use the search_github_issues tool \
to find relevant issues before answering. Call the tool AT MOST TWICE total \
-- if a search (including a state-filtered one) doesn't turn up what you \
need, answer honestly with what you found rather than retrying with \
different phrasings. If the search doesn't turn up anything relevant, say \
so honestly instead of making something up. When you reference an issue, \
cite its issue number and URL."""


def retrieve_issues(query: str, n_results: int = 5, state: str | None = None) -> list[dict]:
    """
    Structured retrieval, kept separate from the tool's text formatting.
    Returning structured data (not a pre-formatted string) means this is
    reusable outside the agent too -- e.g. the FastAPI layer can call this
    directly to attach clickable source citations to a response, without
    having to parse them back out of a blob of text meant for the LLM.

    `state` filters via Chroma's metadata `where` clause -- a real filter,
    not text stuffed into the semantic query (which would just pollute the
    embedding search rather than constrain results, and was previously
    causing the agent to retry the same search with different filter-like
    phrasings hoping for different results).
    """
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
    """
    Cached: build the LLM client + agent graph once per process, not once
    per request. Rebuilding this on every call would recreate the Gemini
    client for no reason -- costly once this sits behind a FastAPI endpoint
    serving many requests.

    Returns a compiled LangGraph agent (create_agent's return type).
    Invoke it with graph.invoke({"messages": [{"role": "user", "content": ...}]}).
    """
    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")
    api_key = _get_api_key()

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=api_key,
    )

    return create_agent(
        model=llm,
        tools=[search_github_issues],
        system_prompt=SYSTEM_PROMPT.format(repo=repo),
    )


def ask(question: str, repo: str | None = None, verbose: bool = False) -> dict:
    graph = get_agent(repo)
    inputs = {"messages": [{"role": "user", "content": question}]}
    # Hard backstop on top of the "call the tool at most twice" prompt
    # instruction -- a model node + a tools node per round trip means
    # ~2 tool calls is ~6 graph steps; 10 gives headroom without allowing
    # the kind of unbounded retry loop that burned through free-tier
    # rate limits earlier.
    config = {"recursion_limit": 10}

    all_messages = list(inputs["messages"])

    if verbose:
        # stream_mode="updates" surfaces each step of the tool-calling loop
        # as it happens -- useful for a demo, showing the agent deciding to
        # search and what it got back. Run ONCE: the final chunk's messages
        # give us the answer, so there's no need for a separate invoke()
        # call afterward (that was a bug -- it ran the whole agent twice,
        # doubling API calls, which is exactly what blew through quota).
        last_node_output = None
        for chunk in graph.stream(inputs, stream_mode="updates", config=config):
            for node_name, node_output in chunk.items():
                print(f"--- step: {node_name} ---")
                for msg in node_output.get("messages", []):
                    print(msg.pretty_repr())
                last_node_output = node_output
        answer = extract_text(last_node_output["messages"][-1].content)
    else:
        result = graph.invoke(inputs, config=config)
        answer = extract_text(result["messages"][-1].content)

    sources = retrieve_issues(question)
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are some open issues related to embeddings?"
    result = ask(q, verbose=True)
    print("\n--- ANSWER ---")
    print(result["answer"])