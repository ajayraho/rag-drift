import json
import uuid

import psycopg2
import pytest

from pipeline import golden_set
from pipeline.db import get_conn_params, get_connection


@pytest.fixture
def golden_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(golden_set, "GOLDEN_DIR", tmp_path)
    return tmp_path


def test_path_is_derived_from_validated_repo(golden_dir):
    assert golden_set.golden_path("facebook/react").name == "facebook__react.jsonl"
    with pytest.raises(ValueError):
        golden_set.golden_path("../../etc/passwd")


def test_missing_file_means_empty_set(golden_dir):
    assert golden_set.load_golden_set("o/r") == []
    assert golden_set.golden_set_version("o/r") is None


def test_save_load_roundtrip_and_version_changes_on_edit(golden_dir):
    items = [{"question": "why does X crash?", "expected": [12], "source_issue": 12, "reviewed": False}]
    golden_set.save_golden_set("o/r", items)
    assert golden_set.load_golden_set("o/r") == items
    v1 = golden_set.golden_set_version("o/r")

    items[0]["reviewed"] = True
    golden_set.save_golden_set("o/r", items)
    assert golden_set.golden_set_version("o/r") != v1


def test_load_rejects_entries_without_expected(golden_dir):
    (golden_dir / "o__r.jsonl").write_text(json.dumps({"question": "q", "expected": []}) + "\n")
    with pytest.raises(ValueError, match="expected"):
        golden_set.load_golden_set("o/r")


def test_blank_lines_ignored_and_expected_coerced_to_int(golden_dir):
    (golden_dir / "o__r.jsonl").write_text('\n{"question": "q", "expected": ["5"]}\n\n')
    assert golden_set.load_golden_set("o/r") == [{"question": "q", "expected": [5]}]


@pytest.mark.parametrize("text,expected", [
    ('{"question": "How do I fix X?"}', "How do I fix X?"),
    ('```json\n{"question": "  spaced   out  "}\n```', "spaced out"),
    ("not json", None),
    ('{"question": ""}', None),
    ('["a list"]', None),
])
def test_parse_question(text, expected):
    assert golden_set._parse_question(text) == expected


def _postgres_available() -> bool:
    try:
        psycopg2.connect(**get_conn_params()).close()
        return True
    except Exception:
        return False


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)

        class Msg:
            content = reply
        return Msg()


@pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")
def test_generate_picks_discussed_closed_issues_and_appends(golden_dir):
    repo = f"pytest/golden-{uuid.uuid4().hex[:6]}"
    body = "Steps to reproduce: " + "x" * 300
    rows = [
        # (id, number, state, comments, body)
        (880001, 1, "closed", 3, body),      # good candidate
        (880002, 2, "open", 3, body),        # open -> excluded
        (880003, 3, "closed", 0, body),      # no discussion -> excluded
        (880004, 4, "closed", 2, "short"),   # too short -> excluded
        (880005, 5, "closed", 1, body),      # good candidate
    ]
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for id_, num, state, comments, b in rows:
                    cur.execute(
                        """INSERT INTO issues (id, repo, number, title, body, state, created_at, updated_at,
                                               comments_count, content_hash)
                           VALUES (%s, %s, %s, %s, %s, %s, now(), now(), %s, 'h')""",
                        (id_, repo, num, f"title {num}", b, state, comments),
                    )

        llm = FakeLLM(['{"question": "first?"}', "garbage"])
        result = golden_set.generate_golden_set(repo, n=10, llm=llm, min_interval=0)

        assert result["candidates"] == 2 and result["added"] == 1 and result["failed"] == 1
        saved = golden_set.load_golden_set(repo)
        assert len(saved) == 1 and saved[0]["reviewed"] is False
        assert saved[0]["expected"] == [saved[0]["source_issue"]]
        assert saved[0]["source_issue"] in (1, 5)

        # Second run: keeps the existing entry, only drafts for the unused issue.
        llm2 = FakeLLM(['{"question": "second?"}'])
        result2 = golden_set.generate_golden_set(repo, n=10, llm=llm2, min_interval=0)
        assert result2["candidates"] == 1 and result2["total"] == 2
        assert {i["source_issue"] for i in golden_set.load_golden_set(repo)} == {1, 5}
    finally:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM issues WHERE repo = %s", (repo,))
