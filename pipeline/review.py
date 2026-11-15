"""
Human side of self-heal: mappings without enough evidence wait here for
approve/reject, and approving one replays quarantined records with it.

Replay: every quarantined record for the repo is re-parsed with the
current mappings; the ones that now pass are stored as issues and leave
quarantine. Same code runs after an auto-applied mapping.
"""

import psycopg2.extras
from pydantic import ValidationError

from pipeline.db import get_active_mapping_overrides, get_connection, save_mapping_override, upsert_issue
from pipeline.log_utils import get_run_logger
from pipeline.schema import GitHubIssue, NotAnIssueError

log = get_run_logger(__name__)


def queue_review(conn, repo: str, decision: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO mapping_reviews (repo, original_field, candidate_field, evidence, matched, overlap)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo, original_field, candidate_field) WHERE status = 'pending' DO NOTHING
            """,
            (repo, decision["field"], decision["candidate_key"], decision["confidence"],
             decision["matched"], decision["overlap"]),
        )
    conn.commit()


def replay_quarantine(conn, repo: str) -> dict:
    overrides = get_active_mapping_overrides(conn, repo)
    recovered = still_bad = 0
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, raw_payload FROM quarantine WHERE repo = %s", (repo,))
        rows = cur.fetchall()
    for row in rows:
        try:
            issue = GitHubIssue.from_github_payload(row["raw_payload"], repo=repo, field_overrides=overrides)
        except (ValidationError, ValueError, KeyError, NotAnIssueError):
            still_bad += 1
            continue
        upsert_issue(conn, issue)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM quarantine WHERE id = %s", (row["id"],))
        recovered += 1
    conn.commit()
    log.info("Quarantine replay for %s: recovered=%s still_bad=%s", repo, recovered, still_bad)
    return {"recovered": recovered, "still_quarantined": still_bad}


def list_pending(repo: str | None = None) -> list[dict]:
    with get_connection() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, repo, original_field, candidate_field, evidence::float, matched, overlap, created_at
            FROM mapping_reviews WHERE status = 'pending' AND (%s::text IS NULL OR repo = %s)
            ORDER BY created_at
            """,
            (repo, repo),
        )
        return [dict(r, created_at=r["created_at"].isoformat()) for r in cur.fetchall()]


def decide(review_id: int, approve: bool) -> dict:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE mapping_reviews SET status = %s, decided_at = now()
                WHERE id = %s AND status = 'pending' RETURNING *
                """,
                ("approved" if approve else "rejected", review_id),
            )
            review = cur.fetchone()
        if review is None:
            raise LookupError(f"no pending review with id {review_id}")
        result = {"id": review_id, "status": review["status"]}
        if approve:
            save_mapping_override(conn, review["repo"], review["original_field"],
                                  review["candidate_field"], float(review["evidence"]))
            result["replay"] = replay_quarantine(conn, review["repo"])
        return result
