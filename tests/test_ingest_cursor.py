"""
Tests for the incremental ingest cursor.

Unit part (always runs): _max_updated_at picks the cursor from raw
records, including PRs and records that will be quarantined.

Integration part (auto-skips without Postgres): the cursor is scoped
per repo -- ingesting repo B must not inherit repo A's high-water mark.
That was a real bug: B's older issues were never fetched.
"""

import uuid
from datetime import datetime, timezone

import psycopg2
import pytest

from pipeline.db import get_conn_params, get_connection
from pipeline.fetch import _max_updated_at


def test_max_updated_at_takes_latest_including_non_issues():
    records = [
        {"updated_at": "2024-01-01T00:00:00Z"},
        {"updated_at": "2024-03-01T00:00:00Z", "pull_request": {}},  # a PR
        {"updated_at": "2024-02-01T00:00:00Z", "title": "   "},     # will be quarantined
    ]
    cursor = None
    for r in records:
        cursor = _max_updated_at(cursor, r)
    assert cursor == datetime(2024, 3, 1, tzinfo=timezone.utc)


def test_max_updated_at_ignores_missing_or_garbage_timestamps():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert _max_updated_at(start, {}) == start
    assert _max_updated_at(start, {"updated_at": None}) == start
    assert _max_updated_at(start, {"updated_at": 12345}) == start
    assert _max_updated_at(start, {"updated_at": "not a date"}) == start
    assert _max_updated_at(None, {"updated_at": "not a date"}) is None


def _postgres_available() -> bool:
    try:
        psycopg2.connect(**get_conn_params()).close()
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")

REPO_A = f"pytest/cursor-a-{uuid.uuid4().hex[:8]}"
REPO_B = f"pytest/cursor-b-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_repos():
    yield
    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in ("issues", "quarantine", "pipeline_runs", "schema_snapshots", "drift_alerts"):
                cur.execute(f"DELETE FROM {table} WHERE repo = ANY(%s)", ([REPO_A, REPO_B],))


def _issue(issue_id: int, updated_at: str) -> dict:
    return {
        "id": issue_id, "number": issue_id, "title": f"issue {issue_id}", "body": "b",
        "state": "open", "labels": [], "user": {"login": "t"},
        "created_at": updated_at, "updated_at": updated_at, "comments": 0,
    }


@needs_db
def test_cursor_is_scoped_per_repo(cleanup_repos, monkeypatch):
    from pipeline import fetch

    seen_since: dict[str, list] = {REPO_A: [], REPO_B: []}
    pages = {
        REPO_A: [_issue(910001, "2025-06-01T00:00:00Z")],
        REPO_B: [_issue(920001, "2020-01-01T00:00:00Z")],
    }

    def fake_page(session, repo, page, since, log):
        seen_since[repo].append(since)
        return pages[repo] if page == 1 else []

    monkeypatch.setattr(fetch, "fetch_issues_page", fake_page)

    fetch.run_ingestion(repo=REPO_A, token=None, max_pages=1)
    fetch.run_ingestion(repo=REPO_B, token=None, max_pages=1)

    # B's first run must start from scratch, NOT from A's 2025 cursor.
    assert seen_since[REPO_B][0] is None

    fetch.run_ingestion(repo=REPO_A, token=None, max_pages=1)
    fetch.run_ingestion(repo=REPO_B, token=None, max_pages=1)

    # Each repo's second run resumes from ITS OWN high-water mark.
    assert seen_since[REPO_A][1] == datetime(2025, 6, 1, tzinfo=timezone.utc)
    assert seen_since[REPO_B][1] == datetime(2020, 1, 1, tzinfo=timezone.utc)


@needs_db
def test_cursor_advances_on_a_page_of_only_prs(cleanup_repos, monkeypatch):
    from pipeline import fetch

    pr = {**_issue(930001, "2024-05-05T00:00:00Z"), "pull_request": {"url": "x"}}
    seen = []

    def fake_page(session, repo, page, since, log):
        seen.append(since)
        return [pr] if page == 1 else []

    monkeypatch.setattr(fetch, "fetch_issues_page", fake_page)
    fetch.run_ingestion(repo=REPO_A, token=None, max_pages=1)
    fetch.run_ingestion(repo=REPO_A, token=None, max_pages=1)

    assert seen[1] == datetime(2024, 5, 5, tzinfo=timezone.utc)
