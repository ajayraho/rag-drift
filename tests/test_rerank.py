import pytest

pytest.importorskip("chromadb")
pytest.importorskip("langchain")

from pipeline import agent  # noqa: E402


class FakeCollection:
    def __init__(self):
        self.asked_n = None

    def query(self, query_texts, n_results, where):
        self.asked_n = n_results
        docs = [f"doc {i}" for i in range(n_results)]
        metas = [{"number": i, "repo": "o/r", "state": "open", "labels": "", "url": "u"} for i in range(n_results)]
        return {"ids": [[str(i) for i in range(n_results)]], "documents": [docs], "metadatas": [metas]}


def test_reranker_reorders_a_wider_candidate_pool(monkeypatch):
    coll = FakeCollection()
    monkeypatch.setattr(agent, "get_chroma_collection", lambda fresh=False: coll)
    monkeypatch.setattr(agent, "RERANK_ENABLED", True)
    monkeypatch.setattr(agent, "rerank", lambda q, docs: list(reversed(range(len(docs)))))

    got = agent.retrieve_issues("q", n_results=5, repo="o/r")

    assert coll.asked_n == agent.RERANK_CANDIDATES      # fetched 20, not 5
    assert [i["number"] for i in got] == [29, 28, 27, 26, 25]


def test_rerank_off_keeps_vector_order(monkeypatch):
    coll = FakeCollection()
    monkeypatch.setattr(agent, "get_chroma_collection", lambda fresh=False: coll)
    monkeypatch.setattr(agent, "RERANK_ENABLED", False)
    got = agent.retrieve_issues("q", n_results=3)
    assert coll.asked_n == 12 and [i["number"] for i in got] == [0, 1, 2]


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(agent, "CACHE_MAX_ENTRIES", 3)
    agent._response_cache.clear()
    for i in range(5):
        agent._store_cache((str(i), "r"), {"answer": i})
    assert list(agent._response_cache) == [("2", "r"), ("3", "r"), ("4", "r")]


def test_chunks_collapse_to_distinct_issues(monkeypatch):
    class Chunks:
        def query(self, query_texts, n_results, where):
            nums = [7, 7, 3, 7, 9, 3]
            metas = [{"number": n, "repo": "o/r", "state": "open", "labels": "", "url": "u"} for n in nums]
            return {"ids": [[str(i) for i in range(6)]], "documents": [[f"chunk {i}" for i in range(6)]], "metadatas": [metas]}
    monkeypatch.setattr(agent, "get_chroma_collection", lambda fresh=False: Chunks())
    monkeypatch.setattr(agent, "RERANK_ENABLED", False)
    got = agent.retrieve_issues("q", n_results=5)
    assert [i["number"] for i in got] == [7, 3, 9]
    assert got[0]["snippet"] == "chunk 0"   # best chunk of issue 7


def test_get_issue_rebuilds_thread_in_chunk_order(monkeypatch):
    class Coll:
        def get(self, where, include):
            assert where == {"$and": [{"repo": "o/r"}, {"number": 7}]}
            meta = {"number": 7, "repo": "o/r", "state": "closed", "labels": "", "url": "u7"}
            return {"ids": ["a", "b"], "documents": ["Issue #7: T\n\nalice: fixed in 1.2", "Issue #7: T\n\nbody text"],
                    "metadatas": [{**meta, "chunk": 1}, {**meta, "chunk": 0}]}
    monkeypatch.setattr(agent, "get_chroma_collection", lambda fresh=False: Coll())
    text, sources = agent.fetch_issue_thread(7, "o/r")
    assert text.index("body text") < text.index("alice: fixed in 1.2")
    assert text.count("Issue #7: T") == 1                       # header not repeated per chunk
    assert sources[0]["number"] == 7 and sources[0]["url"] == "u7"


def test_get_issue_tool_is_bound_to_repo_and_reports_sources(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "fetch_issue_thread", lambda n, r: (calls.append((n, r)) or ("t", [{"number": n}])))
    tool = agent._build_get_issue_tool("o/r")
    msg = tool.invoke({"type": "tool_call", "id": "1", "name": "get_issue", "args": {"number": 7, "repo": "evil/x"}})
    assert calls == [(7, "o/r")] and msg.artifact == [{"number": 7}]
    unbound = agent._build_get_issue_tool(None)
    assert "repo" in unbound.invoke({"number": 7})
