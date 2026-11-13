"""
Schema-drift detector.

Pydantic validation (schema.py) catches malformed CONTENT -- an issue
with a blank title, an unrecognized state value. It does NOT catch
STRUCTURAL drift in the raw source -- e.g. GitHub silently changing
`labels` from a list of objects to a list of strings, or dropping the
`user` field. Structural drift like that often doesn't raise a
validation error at all; it just silently starts feeding wrong/empty
data into the pipeline. That's the failure mode this module exists to
catch: comparing each run's raw payload "shape" against the previous
run's, and alerting on meaningful deviations.

Runnable standalone (against synthetic mutated payloads, for testing
the detector itself without waiting for a real drift event):
`python -m pipeline.drift`
"""

from pipeline.log_utils import get_run_logger

log = get_run_logger(__name__)

# Only the fields our pipeline actually depends on (see schema.py's
# from_github_payload). Fingerprinting GitHub's entire payload would be
# noisy -- we care about drift in the fields we build on, not every field
# GitHub happens to include.
CORE_FIELDS = ["id", "number", "title", "body", "state", "labels", "user", "created_at", "updated_at", "comments"]

# A field that's basically always present (>=90%) but drops to near-zero
# presence is treated as critical -- that's a field disappearing, not
# normal optional-field variance.
PRESENCE_DROP_THRESHOLD = 0.9
PRESENCE_CRITICAL_FLOOR = 0.1


def compute_fingerprint(raw_payloads: list[dict]) -> dict:
    """
    For each core field, records what fraction of payloads had it present
    (non-null), and what Python types were observed for its value. This is
    the "shape" of the source data for this run.
    """
    n = len(raw_payloads)
    fingerprint = {}
    for field in CORE_FIELDS:
        present_count = 0
        types_seen = set()
        for payload in raw_payloads:
            value = payload.get(field)
            if value is not None:
                present_count += 1
                types_seen.add(type(value).__name__)
        fingerprint[field] = {
            "presence_rate": round(present_count / n, 3) if n else 0.0,
            "types": sorted(types_seen),
        }
    return fingerprint


MIN_DRIFT_SAMPLE = 20


def detect_drift(current: dict, baseline: dict) -> list[dict]:
    """
    Compares this run's fingerprint against the prior run's. Returns a
    list of alerts -- empty if nothing meaningful changed.
    """
    alerts = []

    for field, base_info in baseline.items():
        curr_info = current.get(field)
        if curr_info is None:
            alerts.append({
                "field": field,
                "severity": "critical",
                "description": f"Field '{field}' is no longer tracked in the current fingerprint (unexpected).",
            })
            continue

        # A field that was reliably present before but has now nearly
        # vanished -- the clearest, most dangerous drift signal.
        if base_info["presence_rate"] >= PRESENCE_DROP_THRESHOLD and curr_info["presence_rate"] < PRESENCE_CRITICAL_FLOOR:
            alerts.append({
                "field": field,
                "severity": "critical",
                "description": (
                    f"Field '{field}' presence dropped from "
                    f"{base_info['presence_rate']:.0%} to {curr_info['presence_rate']:.0%} "
                    f"-- likely renamed or removed at the source."
                ),
            })

        # A field's JSON type changed -- e.g. labels went from list to str.
        base_types = set(base_info["types"])
        curr_types = set(curr_info["types"])
        new_types = curr_types - base_types
        if base_types and new_types:
            alerts.append({
                "field": field,
                "severity": "warning",
                "description": (
                    f"Field '{field}' is showing new value type(s) {sorted(new_types)} "
                    f"not seen in the prior baseline (which had {sorted(base_types)})."
                ),
            })

    return alerts


def check_drift(conn, repo: str, raw_payloads: list[dict]) -> dict:
    """
    Full drift-check cycle for one ingestion run:
      1. compute this run's fingerprint
      2. compare against the most recently stored fingerprint for this repo
      3. store this run's fingerprint (becomes the baseline for next time)
      4. persist + log any alerts
      5. attempt self-healing on critical "field disappeared" alerts --
         high-confidence mappings get persisted so schema.py's
         from_github_payload can use them starting next run
    """
    from pipeline.db import get_latest_fingerprint, save_drift_alert, save_fingerprint, save_mapping_override
    from pipeline.self_heal import attempt_self_heal

    if not raw_payloads:
        log.info("No raw payloads this run -- skipping drift check.")
        return {"alerts": [], "baseline_established": False, "heal_decisions": []}

    # Hourly runs often fetch a handful of records; one issue with a null
    # body would read as "body presence dropped to 0%". Too small a sample
    # to judge -- skip, and don't let it become the next baseline either.
    if len(raw_payloads) < MIN_DRIFT_SAMPLE:
        log.info("Only %s payloads (< %s) -- too few to judge drift, skipping.", len(raw_payloads), MIN_DRIFT_SAMPLE)
        return {"alerts": [], "baseline_established": False, "heal_decisions": []}

    current_fingerprint = compute_fingerprint(raw_payloads)
    baseline = get_latest_fingerprint(conn, repo)

    save_fingerprint(conn, repo, current_fingerprint)

    if baseline is None:
        log.info("No prior schema baseline for repo=%s -- this run establishes the baseline.", repo)
        return {"alerts": [], "baseline_established": True, "heal_decisions": []}

    alerts = detect_drift(current_fingerprint, baseline)
    for alert in alerts:
        save_drift_alert(conn, repo, alert["severity"], alert["description"])
        if alert["severity"] == "critical":
            log.error("SCHEMA DRIFT (critical): %s", alert["description"])
        else:
            log.warning("SCHEMA DRIFT (warning): %s", alert["description"])

    if not alerts:
        log.info("No schema drift detected this run (repo=%s, %s payloads sampled).", repo, len(raw_payloads))

    heal_decisions = attempt_self_heal(repo, alerts, raw_payloads, baseline, CORE_FIELDS)
    for decision in heal_decisions:
        if decision["status"] == "self_healed":
            save_mapping_override(conn, repo, decision["field"], decision["candidate_key"], decision["confidence"])

    return {"alerts": alerts, "baseline_established": False, "heal_decisions": heal_decisions}


if __name__ == "__main__":
    # Synthetic test: prove the detector actually fires, without waiting
    # for GitHub to change its API. Simulates a baseline run, then a
    # follow-up run where `labels` changed type and `user` disappeared.
    baseline_payloads = [
        {"id": i, "number": i, "title": f"issue {i}", "body": "some body", "state": "open",
         "labels": [{"name": "bug"}], "user": {"login": "someone"}, "created_at": "2024-01-01T00:00:00Z",
         "updated_at": "2024-01-01T00:00:00Z", "comments": 2}
        for i in range(20)
    ]
    drifted_payloads = [
        {"id": i, "number": i, "title": f"issue {i}", "body": "some body", "state": "open",
         "labels": "bug,enhancement",  # <- type changed: list -> str
         "created_at": "2024-01-01T00:00:00Z",           # <- 'user' field missing entirely
         "updated_at": "2024-01-01T00:00:00Z", "comments": 2}
        for i in range(20)
    ]

    baseline_fp = compute_fingerprint(baseline_payloads)
    drifted_fp = compute_fingerprint(drifted_payloads)

    print("Baseline fingerprint:", baseline_fp)
    print("\nDrifted fingerprint:", drifted_fp)

    alerts = detect_drift(drifted_fp, baseline_fp)
    print(f"\n{len(alerts)} alert(s) detected:")
    for a in alerts:
        print(f"  [{a['severity']}] {a['description']}")