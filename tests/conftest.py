import pytest


@pytest.fixture(autouse=True)
def _no_real_github_comments(monkeypatch):
    """Tests fake the issues endpoint; keep the comments sync offline too."""
    from pipeline import fetch
    monkeypatch.setattr(fetch, "fetch_comments_page", lambda *a, **k: [])
