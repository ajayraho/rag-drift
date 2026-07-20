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
    headers = {"Accept": "application/vnd.github+json"}
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
        run_id = start_run(conn)
        log = get_run_logger(__name__, run_id)
        since = get_last_synced_updated_at(conn)
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
                    continue
                except (ValidationError, ValueError, KeyError) as e:
                    quarantined_count += 1
                    insert_quarantine(conn, repo=repo, raw_payload=raw, error_reason=str(e))
                    log.warning("Quarantined issue id=%s: %s", raw.get("id"), e)
                    continue

                upsert_issue(conn, issue)
                valid_count += 1

                if newest_updated_at is None or issue.updated_at > newest_updated_at:
                    newest_updated_at = issue.updated_at

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

        finish_run(conn, run_id, fetched_count, valid_count, quarantined_count, skipped_count, newest_updated_at)

    result = {
        "run_id": run_id,
        "repo": repo,
        "fetched": fetched_count,
        "valid": valid_count,
        "quarantined": quarantined_count,
        "skipped": skipped_count,
        "newest_updated_at": str(newest_updated_at) if newest_updated_at else None,
        "drift_alerts": drift_result["alerts"],
        "heal_decisions": drift_result["heal_decisions"],
    }
    get_run_logger(__name__, run_id).info("Run complete: %s", result)
    return result


if __name__ == "__main__":
    run_ingestion()