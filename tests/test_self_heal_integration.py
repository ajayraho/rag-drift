"""End to end: upstream renames `state`; records get quarantined, the run
heals by value evidence, replays quarantine, and they become issues."""

import uuid

import psycopg2
import pytest

from pipeline.db import get_conn_params, get_connection


def _pg():
    try:
        psycopg2.connect(**get_conn_params()).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg(), reason="Postgres not reachable")
REPO = f"pytest/heal-{uuid.uuid4().hex[:6]}"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    with get_connection() as conn, conn.cursor() as cur:
        for t in ("issues", "quarantine", "pipeline_runs", "schema_snapshots", "drift_alerts",
                  "field_mapping_overrides", "mapping_reviews"):
            cur.execute(f"DELETE FROM {t} WHERE repo = %s", (REPO,))


def _issue(i, updated):
    return {"id": 810000 + i, "number": i, "title": f"t{i}", "body": "b", "state": "open" if i % 2 else "closed",
            "labels": [], "user": {"login": "u"}, "created_at": "2024-01-01T00:00:00Z",
            "updated_at": updated, "comments": 0}


def _run(monkeypatch, payloads):
    from pipeline import fetch
    monkeypatch.setattr(fetch, "fetch_issues_page", lambda s, r, p, since, log: payloads if p == 1 else [])
    return fetch.run_ingestion(repo=REPO, token=None, max_pages=1)


def test_rename_is_healed_and_quarantine_replayed(monkeypatch):
    _run(monkeypatch, [_issue(i, "2024-01-01T00:00:00Z") for i in range(25)])      # baseline, stored

    drifted = []
    for i in range(25):                                                           # same issues, updated, state renamed
        p = _issue(i, "2024-02-01T00:00:00Z")
        p["lifecycle"] = p.pop("state")
        drifted.append(p)
    result = _run(monkeypatch, drifted)

    assert result["quarantined"] == 25                                            # couldn't parse this run...
    healed = [d for d in result["heal_decisions"] if d["status"] == "self_healed"]
    assert healed and healed[0]["candidate_key"] == "lifecycle"
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM quarantine WHERE repo = %s", (REPO,))
        assert cur.fetchone()[0] == 0                                             # ...but replay recovered all
        cur.execute("SELECT count(*) FROM issues WHERE repo = %s AND updated_at >= '2024-02-01'", (REPO,))
        assert cur.fetchone()[0] == 25


def test_weak_evidence_goes_to_review_and_approval_replays(monkeypatch):
    from pipeline import review

    _run(monkeypatch, [_issue(i, "2024-01-01T00:00:00Z") for i in range(25)])
    drifted = []
    for i in range(25):
        p = _issue(i, "2024-02-01T00:00:00Z")
        value = p.pop("state")
        p["status"] = value if i < 18 else "unknown"                              # 18/25 = 72% match
        drifted.append(p)
    result = _run(monkeypatch, drifted)
    assert [d["status"] for d in result["heal_decisions"]] == ["needs_review"]

    (pending,) = review.list_pending(REPO)
    assert pending["candidate_field"] == "status" and pending["matched"] == 18

    decided = review.decide(pending["id"], approve=True)
    assert decided["replay"]["recovered"] == 18 and decided["replay"]["still_quarantined"] == 7
    assert review.list_pending(REPO) == []
    with pytest.raises(LookupError):
        review.decide(pending["id"], approve=False)
