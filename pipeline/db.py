"""
Postgres access layer. Keeps raw SQL in one place so fetch.py stays
focused on "get data from GitHub, validate it" rather than SQL details.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras

from pipeline.schema import GitHubIssue


def get_conn_params() -> dict:
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "dbname": os.environ.get("POSTGRES_DB", "rag_pipeline"),
        "user": os.environ.get("POSTGRES_USER", "rag_user"),
        "password": os.environ.get("POSTGRES_PASSWORD", "rag_pass"),
    }


@contextmanager
def get_connection():
    conn = psycopg2.connect(**get_conn_params())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_last_synced_updated_at(conn, repo: str) -> Optional[datetime]:
    """
    Reads THIS repo's high-water mark from its most recent run that set
    one. This is what makes ingestion incremental: the next GitHub call
    asks for 'since=<this timestamp>' instead of pulling everything.

    Scoped by repo on purpose. It used to read the latest cursor across
    ALL runs, so ingesting repo B after repo A started B from A's
    timestamp -- B's older issues were never fetched at all. Rows written
    before migration 0002 have repo = NULL and are ignored here.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT last_synced_updated_at
            FROM pipeline_runs
            WHERE repo = %s
              AND last_synced_updated_at IS NOT NULL
            ORDER BY run_started_at DESC
            LIMIT 1
            """,
            (repo,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def get_tracked_repos(conn) -> list[str]:
    """
    Every repo the pipeline has ever ingested -- what the scheduled hourly
    run refreshes. Reads both pipeline_runs (repo column since migration
    0002) and issues (covers repos ingested before 0002, whose run rows
    have repo = NULL).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT repo FROM pipeline_runs WHERE repo IS NOT NULL
            UNION
            SELECT DISTINCT repo FROM issues
            ORDER BY 1
            """
        )
        return [row[0] for row in cur.fetchall()]


def start_run(conn, repo: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_runs (run_started_at, repo) VALUES (now(), %s) RETURNING id",
            (repo,),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_run(conn, run_id: int, fetched: int, valid: int, quarantined: int, skipped: int, newest_updated_at: Optional[datetime]):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET run_finished_at = now(),
                fetched_count = %s,
                valid_count = %s,
                quarantined_count = %s,
                skipped_count = %s,
                last_synced_updated_at = COALESCE(%s, last_synced_updated_at)
            WHERE id = %s
            """,
            (fetched, valid, quarantined, skipped, newest_updated_at, run_id),
        )
    conn.commit()


def upsert_issue(conn, issue: GitHubIssue):
    """
    Insert or update an issue. On conflict (same idseen again),
    we update the content_hash and embedded flag stays as-is unless
    the hash actually changed -- that's the incremental-embedding hook.
    """
    new_hash = issue.content_hash()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO issues (
                id, repo, number, title, body, state, labels,
                created_at, updated_at, author, comments_count, content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                body = EXCLUDED.body,
                state = EXCLUDED.state,
                labels = EXCLUDED.labels,
                updated_at = EXCLUDED.updated_at,
                comments_count = EXCLUDED.comments_count,
                -- only flip embedded back to false if content actually changed
                embedded = CASE
                    WHEN issues.content_hash != EXCLUDED.content_hash THEN false
                    ELSE issues.embedded
                END,
                content_hash = EXCLUDED.content_hash
            """,
            (
                issue.id, issue.repo, issue.number, issue.title, issue.body,
                issue.state, issue.labels, issue.created_at, issue.updated_at,
                issue.author, issue.comments_count, new_hash,
            ),
        )


def insert_quarantine(conn, repo: str, raw_payload: dict, error_reason: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO quarantine (repo, raw_payload, error_reason)
            VALUES (%s, %s, %s)
            """,
            (repo, json.dumps(raw_payload), error_reason),
        )


def get_unembedded_issues(conn, limit: int = 200) -> list[dict]:
    """
    Returns issues where embedded=false -- i.e. brand new issues, or
    issues whose content_hash changed since they were last embedded
    (upsert_issue flips embedded back to false in that case). This is
    the query that makes embedding incremental instead of a full re-embed.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, repo, number, title, body, state, labels, updated_at
            FROM issues
            WHERE embedded = false
            ORDER BY updated_at ASC
            LIMIT %s
            """,
            (limit,),
        )
        return list(cur.fetchall())


def mark_embedded(conn, issue_ids: list[int]):
    if not issue_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE issues SET embedded = true WHERE id = ANY(%s)",
            (issue_ids,),
        )
    conn.commit()


def save_fingerprint(conn, repo: str, fingerprint: dict):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO schema_snapshots (repo, fingerprint) VALUES (%s, %s)",
            (repo, json.dumps(fingerprint)),
        )
    conn.commit()


def get_latest_fingerprint(conn, repo: str) -> Optional[dict]:
    """
    Returns the most recently stored fingerprint for this repo, i.e. the
    baseline the CURRENT run should be compared against. None if this is
    the first run ever for this repo (nothing to compare against yet).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT fingerprint FROM schema_snapshots
            WHERE repo = %s
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            (repo,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def save_drift_alert(conn, repo: str, severity: str, description: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO drift_alerts (repo, severity, description)
            VALUES (%s, %s, %s)
            """,
            (repo, severity, description),
        )
    conn.commit()


def save_mapping_override(conn, repo: str, original_field: str, mapped_field: str, confidence: float):
    """
    Records a self-healed field mapping (e.g. 'state' -> 'status'). Always
    inserts a new row rather than updating in place -- keeps a full history
    of healing decisions over time, and get_active_mapping_overrides just
    reads the most recent one per field.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO field_mapping_overrides (repo, original_field, mapped_field, confidence)
            VALUES (%s, %s, %s, %s)
            """,
            (repo, original_field, mapped_field, confidence),
        )
    conn.commit()


def get_active_mapping_overrides(conn, repo: str) -> dict:
    """
    Returns {original_field: mapped_field} for the most recent healed
    mapping per field. This is what schema.py's from_github_payload
    consults to extract a field that was renamed at the source -- e.g. if
    'state' was auto-mapped to 'status', a payload with only 'status'
    still gets extracted correctly.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (original_field) original_field, mapped_field
            FROM field_mapping_overrides
            WHERE repo = %s
            ORDER BY original_field, created_at DESC
            """,
            (repo,),
        )
        rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}