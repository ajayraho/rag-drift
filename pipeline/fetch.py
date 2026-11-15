"""
Fetches issues from the GitHub API, validates each one through
GitHubIssue, and writes results to Postgres -- valid rows to `issues`,
invalid rows to `quarantine`.

This is the module Airflow calls. It's also runnable standalone for
local testing: `python -m pipeline.fetch`
"""

import logging
import os
import time
from datetime import datetime, timezone

import requests
from pydantic import ValidationError

from pipeline.db import (
    finish_run,
    get_active_mapping_overrides,
    get_comments_cursor,
    mark_issues_unembedded,
    upsert_comment,
    get_connection,
    get_last_synced_updated_at,
    insert_quarantine,
    start_run,
    upsert_issue,
)
from pipeline.drift import check_drift
from pipeline.log_utils import get_run_logger
from pipeline.schema import GitHubIssue, NotAnIssueError

GITHUB_API = "https://api.github.com"


def build_session(token: str | None) -> requests.Session:
    """
    A single Session reuses the underlying TCP connection across all
    paginated requests instead of renegotiating a new connection per page --
    meaningfully faster once you're pulling multiple pages every run.
    """
    session = requests.Session()
    # Pinning the REST API version means GitHub won't change response
    # shapes under us without notice -- the cheapest drift prevention.
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


def check_rate_limit(resp: requests.Response, log: logging.LoggerAdapter):
    """
    GitHub tells you your rate-limit status on every response via headers --
    no need for a separate API call. If we're about to run dry, sleep until
    the window resets instead of blowing up mid-run on a 403.
    """
    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset_at = resp.headers.get("X-RateLimit-Reset")
    if remaining is None:
        return

    remaining = int(remaining)
    if remaining <= 1 and reset_at:
        sleep_for = max(int(reset_at) - int(time.time()), 0) + 2
        log.warning("Rate limit nearly exhausted (remaining=%s). Sleeping %ss until reset.", remaining, sleep_for)
        time.sleep(sleep_for)
    elif remaining <= 5:
        log.info("Rate limit getting low: %s requests remaining this window.", remaining)


def fetch_issues_page(
    session: requests.Session,
    repo: str,
    page: int,
    since: datetime | None,
    log: logging.LoggerAdapter,
) -> list[dict]:
    params = {
        "state": "all",
        "per_page": 100,
        "page": page,
        "sort": "updated",
        "direction": "asc",
    }
    if since:
        params["since"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    resp = session.get(f"{GITHUB_API}/repos/{repo}/issues", params=params, timeout=30)
    check_rate_limit(resp, log)

    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        sleep_for = max(reset_at - int(time.time()), 0) + 2
        log.warning("Hit rate limit (403). Sleeping %ss and retrying page %s.", sleep_for, page)
        time.sleep(sleep_for)
        resp = session.get(f"{GITHUB_API}/repos/{repo}/issues", params=params, timeout=30)

    resp.raise_for_status()
    return resp.json()


def fetch_comments_page(session, repo: str, page: int, since: datetime | None, log) -> list[dict]:
    """
    One page of ALL comments in the repo, oldest-updated first. A single
    repo-wide endpoint (not one call per issue), so it's incremental and
    cheap exactly like the issues sync.
    """
    params = {"per_page": 100, "page": page, "sort": "updated", "direction": "asc"}
    if since:
        params["since"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = session.get(f"{GITHUB_API}/repos/{repo}/issues/comments", params=params, timeout=30)
    check_rate_limit(resp, log)
    resp.raise_for_status()
    return resp.json()


def parse_comment(raw: dict, repo: str) -> dict | None:
    """GitHub comment payload -> comments row, or None if it's unusable."""
    try:
        issue_number = int(raw["issue_url"].rstrip("/").rsplit("/", 1)[1])
        user = raw.get("user") or {}
        login = user.get("login") or ""
        return {
            "id": int(raw["id"]),
            "repo": repo,
            "issue_number": issue_number,
            "author": login,
            "is_bot": user.get("type") == "Bot" or login.endswith("[bot]"),
            "body": raw.get("body") or "",
            "created_at": raw["created_at"],
            "updated_at": raw["updated_at"],
        }
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        return None


def sync_comments(conn, session, repo: str, log, max_pages: int | None = None) -> int:
    """
    Pulls comments updated since the newest one we have, upserts them,
    and flags their issues for re-embedding (the thread changed).
    Returns the number of comments stored.
    """
    # 10 pages (1000 comments) per hourly run; raise COMMENTS_MAX_PAGES for
    # a one-off backfill (5000 req/hour token limit allows thousands).
    max_pages = max_pages or int(os.environ.get("COMMENTS_MAX_PAGES", "10"))
    since = get_comments_cursor(conn, repo)
    stored = 0
    for page in range(1, max_pages + 1):
        rows = fetch_comments_page(session, repo, page, since, log)
        touched = set()
        for raw in rows:
            comment = parse_comment(raw, repo)
            if comment is None:
                continue
            upsert_comment(conn, comment)
            touched.add(comment["issue_number"])
            stored += 1
        mark_issues_unembedded(conn, repo, sorted(touched))
        conn.commit()
        if len(rows) < 100:
            break
    log.info("Comments: stored %s (since=%s)", stored, since)
    return stored


def _max_updated_at(current: datetime | None, raw: dict) -> datetime | None:
    """
    Cursor candidate from a raw record. Called for valid issues AND for
    skipped PRs -- but deliberately NOT for quarantined records.

    PRs: the cursor used to advance only on valid issues, so a page made
    up entirely of PRs (common on busy repos -- /issues returns PRs too)
    never moved it, and every following run re-fetched the same PRs.

    Quarantined records: NOT advancing past them is what lets self-heal
    actually recover data. When a field gets renamed upstream, that run's
    records are quarantined and a mapping is learned; because the cursor
    stayed behind them, the NEXT run re-fetches them and they now parse.
    Advancing past them would strand them in quarantine until a replay
    mechanism exists (roadmap 4.6). Unparseable timestamps are ignored.
    """
    value = raw.get("updated_at")
    if not isinstance(value, str):
        return current
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return current
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts if current is None or ts > current else current


def run_ingestion(repo: str | None = None, token: str | None = None, max_pages: int = 5) -> dict:
    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")
    token = token or os.environ.get("GITHUB_TOKEN")

    fetched_count = 0
    valid_count = 0
    quarantined_count = 0
    skipped_count = 0
    newest_updated_at = None
    raw_payloads_sample = []  # feeds the schema drift fingerprint below

    session = build_session(token)

    with get_connection() as conn:
        run_id = start_run(conn, repo)
        log = get_run_logger(__name__, run_id)
        since = get_last_synced_updated_at(conn, repo)
        field_overrides = get_active_mapping_overrides(conn, repo)
        if field_overrides:
            log.info("Using %s self-healed field mapping override(s): %s", len(field_overrides), field_overrides)
        log.info("Starting run for repo=%s since=%s", repo, since)

        for page in range(1, max_pages + 1):
            issues_raw = fetch_issues_page(session, repo, page, since, log)
            if not issues_raw:
                break

            for raw in issues_raw:
                fetched_count += 1
                raw_payloads_sample.append(raw)
                try:
                    issue = GitHubIssue.from_github_payload(raw, repo=repo, field_overrides=field_overrides)
                except NotAnIssueError:
                    # Expected, by-design exclusion -- not an error, don't quarantine.
                    skipped_count += 1
                    newest_updated_at = _max_updated_at(newest_updated_at, raw)
                    continue
                except (ValidationError, ValueError, KeyError) as e:
                    quarantined_count += 1
                    insert_quarantine(conn, repo=repo, raw_payload=raw, error_reason=str(e))
                    log.warning("Quarantined issue id=%s: %s", raw.get("id"), e)
                    continue
                except Exception as e:
                    # Anything else raised while turning ONE payload into a
                    # GitHubIssue is still a problem with that record, not
                    # with the pipeline -- quarantine it (with the exception
                    # type, so it stands out) and keep going instead of
                    # failing the whole run. Deliberately scoped to parsing
                    # only: DB errors from upsert_issue below must still fail
                    # loudly, because quarantining those would hide an outage.
                    quarantined_count += 1
                    reason = f"unexpected {type(e).__name__} during extraction: {e}"
                    insert_quarantine(conn, repo=repo, raw_payload=raw, error_reason=reason)
                    log.exception("Quarantined issue id=%s after unexpected error", raw.get("id"))
                    continue

                upsert_issue(conn, issue)
                valid_count += 1
                newest_updated_at = _max_updated_at(newest_updated_at, raw)

            # Commit after each page (not after every single row, not only at
            # the very end): if page 3 of 5 throws, pages 1-2 are already
            # durably saved and next run's `since` won't have to re-walk them.
            # This is a deliberate checkpoint interval, not a leftover.
            conn.commit()
            log.info("Page %s: %s issues processed (running total valid=%s, quarantined=%s, skipped=%s)",
                      page, len(issues_raw), valid_count, quarantined_count, skipped_count)

            if len(issues_raw) < 100:
                break  # last page

        drift_result = check_drift(conn, repo, raw_payloads_sample)
        comments_count = sync_comments(conn, session, repo, log)

        finish_run(conn, run_id, fetched_count, valid_count, quarantined_count, skipped_count, newest_updated_at)

    result = {
        "run_id": run_id,
        "repo": repo,
        "fetched": fetched_count,
        "valid": valid_count,
        "quarantined": quarantined_count,
        "skipped": skipped_count,
        "newest_updated_at": str(newest_updated_at) if newest_updated_at else None,
        "comments": comments_count,
        "drift_alerts": drift_result["alerts"],
        "heal_decisions": drift_result["heal_decisions"],
    }
    get_run_logger(__name__, run_id).info("Run complete: %s", result)
    return result


if __name__ == "__main__":
    run_ingestion()