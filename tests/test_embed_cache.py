"""
get_chroma_collection must build the Chroma client and load the
embedding model ONCE per process -- it used to do both on every call
(the docstring claimed caching that didn't exist).
"""

import pytest

pytest.importorskip("chromadb")

from pipeline import embed  # noqa: E402


@pytest.fixture
def fake_chroma(monkeypatch):
    counts = {"client": 0, "model": 0}

    class FakeCollection:
        pass

    class FakeClient:
        def __init__(self, host, port):
            counts["client"] += 1

        def get_or_create_collection(self, **kwargs):
            return FakeCollection()

    def fake_model(model_name):
        counts["model"] += 1
        return object()

    monkeypatch.setattr(embed.chromadb, "HttpClient", FakeClient)
    monkeypatch.setattr(embed.embedding_functions, "SentenceTransformerEmbeddingFunction", fake_model)
    embed.get_embedding_function.cache_clear()
    monkeypatch.setattr(embed, "_collection", None)
    yield counts
    embed.get_embedding_function.cache_clear()
    embed._collection = None


def test_collection_and_model_are_built_once(fake_chroma):
    first = embed.get_chroma_collection()
    for _ in range(5):
        assert embed.get_chroma_collection() is first
    assert fake_chroma == {"client": 1, "model": 1}


def test_fresh_reconnects_but_keeps_the_loaded_model(fake_chroma):
    first = embed.get_chroma_collection()
    second = embed.get_chroma_collection(fresh=True)
    assert second is not first
    assert fake_chroma == {"client": 2, "model": 1}
