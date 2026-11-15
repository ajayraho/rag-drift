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
