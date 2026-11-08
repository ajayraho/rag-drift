"""
Integration test: proves quarantine actually catches malformed data
end-to-end against a REAL Postgres -- not a unit-level mock assertion.

Feeds a batch of GitHub-shaped payloads (a mix of valid, malformed, and
a pull request) through the real run_ingestion() pipeline. Only the
GitHub HTTP call is mocked out; validation, quarantine insertion, issue
upserts, and drift checking all run for real against the real database.

Requires Postgres reachable via the same env vars pipeline/db.py uses
(defaults matching docker-compose: localhost:5432). Automatically
skipped if that connection isn't available, so the rest of the suite
still runs fine in a sandbox/CI environment without a DB.
"""

import uuid

import psycopg2
import pytest

from pipeline.db import get_conn_params, get_connection

TEST_REPO = f"pytest/quarantine-stress-{uuid.uuid4().hex[:8]}"


def _postgres_available() -> bool:
    try:
        conn = psycopg2.connect(**get_conn_params())
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres not reachable -- run with the docker-compose stack up to execute this test.",
)


@pytest.fixture
def cleanup_test_repo():
    yield
    # Runs even if the test fails partway through, so a failed run
    # doesn't leave stray rows for the next test run to trip over.
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Every table a run writes to -- leftover pipeline_runs rows used
            # to move the (then global) ingest cursor for real repos.
            for table in ("issues", "quarantine", "pipeline_runs", "schema_snapshots", "drift_alerts"):
                cur.execute(f"DELETE FROM {table} WHERE repo = %s", (TEST_REPO,))


def _fake_github_page(session, repo, page, since, log):
    """
    Replaces the real GitHub HTTP call with a fixed, deliberately mixed
    batch: one valid issue, one PR (should be skipped, not quarantined),
    and three genuinely malformed records covering the actual quarantine-
    worthy cases -- blank title, missing required field, invalid enum value.
    """
    if page > 1:
        return []

    good_issue = {
        "id": 900001, "number": 1, "title": "A perfectly normal issue",
        "body": "Nothing wrong here.", "state": "open",
        "labels": [{"name": "bug"}], "user": {"login": "tester"},
        "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
        "comments": 0,
    }
    blank_title_issue = {**good_issue, "id": 900002, "number": 2, "title": "   "}
    missing_state_issue = {k: v for k, v in good_issue.items() if k != "state"}
    missing_state_issue.update({"id": 900003, "number": 3})
    invalid_state_issue = {**good_issue, "id": 900004, "number": 4, "state": "archived"}
    a_pull_request = {**good_issue, "id": 900005, "number": 5, "pull_request": {"url": "https://x/pull/5"}}

    return [good_issue, blank_title_issue, missing_state_issue, invalid_state_issue, a_pull_request]


def test_quarantine_catches_malformed_records_end_to_end(cleanup_test_repo, monkeypatch):
    from pipeline import fetch

    monkeypatch.setattr(fetch, "fetch_issues_page", _fake_github_page)

    result = fetch.run_ingestion(repo=TEST_REPO, token=None, max_pages=2)

    # 1 valid, 1 PR (skipped -- by design, not a quarantine case),
    # 3 genuinely malformed (blank title, missing state, invalid state).
    assert result["fetched"] == 5
    assert result["valid"] == 1
    assert result["skipped"] == 1
    assert result["quarantined"] == 3

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM issues WHERE repo = %s", (TEST_REPO,))
            issue_ids = {row[0] for row in cur.fetchall()}
            assert issue_ids == {900001}

            cur.execute(
                "SELECT raw_payload->>'id', error_reason FROM quarantine WHERE repo = %s ORDER BY id",
                (TEST_REPO,),
            )
            quarantined = cur.fetchall()

    assert len(quarantined) == 3
    quarantined_ids = {row[0] for row in quarantined}
    assert quarantined_ids == {"900002", "900003", "900004"}

    # Every quarantined row has a real, specific reason -- not blank or
    # a placeholder -- since that reason is the whole point of quarantine
    # over silently dropping bad data.
    for _, reason in quarantined:
        assert reason and len(reason) > 5


def test_pull_requests_are_never_quarantined(cleanup_test_repo, monkeypatch):
    """
    A narrower, explicit check that PRs specifically land in the
    'skipped' bucket, not quarantine -- conflating the two was a real
    issue we caught and fixed on Day 1 (an 88% quarantine rate that
    looked like a broken pipeline but was actually just PR filtering).
    """
    from pipeline import fetch

    def only_a_pr(session, repo, page, since, log):
        if page > 1:
            return []
        return [{
            "id": 900099, "number": 99, "title": "A PR, not an issue",
            "body": "", "state": "open", "labels": [], "user": {"login": "tester"},
            "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
            "comments": 0, "pull_request": {"url": "https://x/pull/99"},
        }]

    monkeypatch.setattr(fetch, "fetch_issues_page", only_a_pr)
    result = fetch.run_ingestion(repo=TEST_REPO, token=None, max_pages=2)

    assert result["skipped"] == 1
    assert result["quarantined"] == 0