import pytest

pytest.importorskip("chromadb")

from pipeline import embed  # noqa: E402
from pipeline.fetch import parse_comment  # noqa: E402

ISSUE = {"id": 1, "number": 42, "title": "Crash on start", "body": "", "repo": "o/r"}


def _c(author, body, bot=False):
    return {"author": author, "body": body, "is_bot": bot}


def test_clean_text_strips_template_noise_keeps_code():
    raw = "<!-- describe your bug -->\n- [x] I searched issues\nIt fails:\n```\nTraceback ...\n```"
    out = embed.clean_text(raw)
    assert "describe your bug" not in out and "I searched" not in out
    assert "Traceback" in out and "It fails" in out


def test_every_chunk_has_title_and_fits_size(monkeypatch):
    monkeypatch.setattr(embed, "CHUNK_WORDS", 10)
    issue = {**ISSUE, "body": " ".join(f"b{i}" for i in range(25))}
    comments = [_c("alice", " ".join(f"a{i}" for i in range(4))), _c("bob", " ".join(f"x{i}" for i in range(30)))]
    chunks = embed.build_chunks(issue, comments)
    assert all(c.startswith("Issue #42: Crash on start") for c in chunks)
    assert all(len(c.split("\n\n", 1)[1].split()) <= 10 for c in chunks)
    joined = " ".join(chunks)
    assert "b24" in joined and "a3" in joined and "x29" in joined  # nothing dropped


def test_bot_and_empty_comments_skipped():
    chunks = embed.build_chunks({**ISSUE, "body": "body"}, [_c("ci-bot", "coverage 90%", bot=True), _c("u", "  ")])
    assert len(chunks) == 1 and "coverage" not in chunks[0]


def test_short_thread_comments_packed_into_one_chunk():
    chunks = embed.build_chunks({**ISSUE, "body": "body"}, [_c("a", "fixed in 1.2"), _c("b", "thanks")])
    assert len(chunks) == 2 and "a: fixed in 1.2" in chunks[1] and "b: thanks" in chunks[1]


def test_parse_comment():
    raw = {"id": 5, "issue_url": "https://api.github.com/repos/o/r/issues/42", "body": "hi",
           "user": {"login": "dependabot[bot]", "type": "Bot"},
           "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-02T00:00:00Z"}
    c = parse_comment(raw, "o/r")
    assert c["issue_number"] == 42 and c["is_bot"] is True
    assert parse_comment({"id": 1}, "o/r") is None
