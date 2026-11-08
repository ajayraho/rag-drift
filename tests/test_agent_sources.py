"""
Sources must be what the agent's tool calls actually returned -- not a
separate retrieval on the raw question (the old behaviour).
"""

import pytest

pytest.importorskip("langchain")
pytest.importorskip("chromadb")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from pipeline import agent  # noqa: E402


def _issue(n, repo="o/r"):
    return {"number": n, "repo": repo, "state": "open", "labels": "", "url": f"u{n}", "snippet": "s"}


def test_tool_returns_text_for_llm_and_issues_as_artifact(monkeypatch):
    calls = []

    def fake_retrieve(query, state=None, repo=None, n_results=5):
        calls.append((query, state, repo))
        return [_issue(1), _issue(2)]

    monkeypatch.setattr(agent, "retrieve_issues", fake_retrieve)
    tool = agent._build_search_tool("o/r")
    msg = tool.invoke({
        "type": "tool_call", "id": "c1", "name": tool.name,
        "args": {"query": "rewritten query", "state": "closed"},
    })

    assert isinstance(msg, ToolMessage)
    assert "Issue #1" in msg.content and "Issue #2" in msg.content
    assert msg.artifact == [_issue(1), _issue(2)]
    # repo comes from the closure, not from the LLM
    assert calls == [("rewritten query", "closed", "o/r")]


def test_collect_sources_dedupes_across_calls_and_ignores_non_tool_messages():
    msgs = [
        AIMessage(content="thinking"),
        ToolMessage(content="x", tool_call_id="a", artifact=[_issue(1), _issue(2)]),
        ToolMessage(content="y", tool_call_id="b", artifact=[_issue(2), _issue(3), _issue(3, repo="other/r")]),
        ToolMessage(content="no artifact", tool_call_id="c"),
    ]
    got = [(s["repo"], s["number"]) for s in agent._collect_sources(msgs)]
    assert got == [("o/r", 1), ("o/r", 2), ("o/r", 3), ("other/r", 3)]


def test_no_tool_calls_means_no_sources():
    assert agent._collect_sources([AIMessage(content="answer without searching")]) == []


def test_end_to_end_sources_come_from_the_agents_own_search(monkeypatch):
    """
    Full create_agent loop with a scripted tool-calling model: the agent
    searches with a REWRITTEN query + state filter. Sources must be that
    search's results, and there must be no second retrieval on the raw
    question.
    """
    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ScriptedModel(BaseChatModel):
        turns: int = 0

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "scripted"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.turns += 1
            if self.turns == 1:
                msg = AIMessage(content="", tool_calls=[{
                    "name": "search_github_issues", "id": "t1",
                    "args": {"query": "agent rewritten query", "state": "open"},
                }])
            else:
                msg = AIMessage(content="Answer citing #7")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    queries = []

    def fake_retrieve(query, n_results=5, state=None, repo=None):
        queries.append(query)
        return [_issue(7, repo=repo)]

    monkeypatch.setattr(agent, "retrieve_issues", fake_retrieve)
    monkeypatch.setattr(agent, "get_agent", lambda repo=None: create_agent(
        model=ScriptedModel(), tools=[agent._build_search_tool(repo)], system_prompt="test",
    ))
    agent._response_cache.clear()

    result = agent.ask("the raw user question", repo="o/r")

    assert queries == ["agent rewritten query"]
    assert [(s["repo"], s["number"]) for s in result["sources"]] == [("o/r", 7)]
    assert result["answer"] == "Answer citing #7"
