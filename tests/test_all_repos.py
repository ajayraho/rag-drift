"""
The scheduled run refreshes every tracked repo, and one failing repo
must not stop the rest.
"""

import uuid

import psycopg2
import pytest

pytest.importorskip("mlflow")

from pipeline import mlflow_tracking  # noqa: E402
from pipeline.db import get_conn_params, get_connection  # noqa: E402


def test_run_all_repos_continues_past_a_failing_repo():
    order = []

    def runner(repo):
        order.append(repo)
        if repo == "bad/repo":
            raise RuntimeError("GitHub 404")

    summary = mlflow_tracking.run_all_repos(["a/one", "bad/repo", "c/three"], runner=runner)

    assert order == ["a/one", "bad/repo", "c/three"]
    assert summary["succeeded"] == ["a/one", "c/three"]
    assert summary["failed"] == {"bad/repo": "RuntimeError: GitHub 404"}


def _postgres_available() -> bool:
    try:
        psycopg2.connect(**get_conn_params()).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")
def test_tracked_repos_include_legacy_issue_rows_and_default(monkeypatch):
    run_repo = f"pytest/tracked-run-{uuid.uuid4().hex[:6]}"
    legacy_repo = f"pytest/tracked-legacy-{uuid.uuid4().hex[:6]}"
    monkeypatch.setenv("GITHUB_REPO", "default-owner/default-repo")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO pipeline_runs (repo) VALUES (%s)", (run_repo,))
                # pre-0002 style: only an issues row, no run row with a repo
                cur.execute(
                    """INSERT INTO issues (id, repo, number, title, state, created_at, updated_at, content_hash)
                       VALUES (987654321, %s, 1, 't', 'open', now(), now(), 'h')""",
                    (legacy_repo,),
                )
                cur.execute("INSERT INTO pipeline_runs (repo) VALUES ('evil/x;rm -rf')")

        repos = mlflow_tracking.resolve_tracked_repos()

        assert repos[0] == "default-owner/default-repo"
        assert run_repo in repos and legacy_repo in repos
        assert "evil/x;rm -rf" not in repos  # names from the DB are re-validated
        assert len(repos) == len(set(repos))
    finally:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pipeline_runs WHERE repo = ANY(%s)", ([run_repo, "evil/x;rm -rf"],))
                cur.execute("DELETE FROM issues WHERE id = 987654321")
