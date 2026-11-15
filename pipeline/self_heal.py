"""
Self-healing schema adaptation, by EVIDENCE rather than by name.

When drift.py flags a field as "disappeared" (presence dropped to ~0),
we look for the field that now carries its data. Many issues in a new
batch are ones we already store (sorted by updated_at, re-fetched when
they change), so we know what the missing field's value SHOULD be for
them. Whichever new key holds exactly those values is the rename:

    'body' vanished; for 38 of 40 issues we already have, the new key
    'description' contains the exact body text we stored -> that's it.

  * evidence >= 90% over >= 5 known issues -> mapping auto-applied
  * weaker evidence -> queued for a human (mapping_reviews table / UI)
  * no candidate, or too few known issues to check -> human review,
    never a guess

This replaced fuzzy name matching (difflib), which judged by spelling and
happily mapped `labels` to `labels_url`.

Runnable standalone (synthetic drift scenario, randomized each run):
`python -m pipeline.self_heal`
"""

import random
from datetime import datetime, timezone

from pipeline.log_utils import get_run_logger

log = get_run_logger(__name__)

AUTO_APPLY_EVIDENCE = 0.9
REVIEW_MIN_EVIDENCE = 0.5
MIN_KNOWN_OVERLAP = 5

# GitHub payload field -> our `issues` column holding the same data.
FIELD_TO_COLUMN = {
    "title": "title", "body": "body", "state": "state", "number": "number",
    "labels": "labels", "user": "author", "comments": "comments_count",
    "created_at": "created_at", "updated_at": "updated_at",
}


def normalize(field: str, value):
    """Puts a raw payload value and a stored column value into one comparable form."""
    if value is None:
        return None
    try:
        if field == "labels":
            if isinstance(value, str):
                value = [x.strip() for x in value.split(",") if x.strip()]
            return tuple(sorted(v["name"] if isinstance(v, dict) else str(v) for v in value))
        if field == "user":
            return value.get("login") if isinstance(value, dict) else (value if isinstance(value, str) else None)
        if field in ("created_at", "updated_at"):
            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        if field in ("number", "comments"):
            return int(value)
        return str(value)
    except (TypeError, ValueError, AttributeError, KeyError):
        return None


def propose_mapping(missing_field: str, raw_payloads: list[dict], known: dict[int, object],
                    exclude_keys: set[str]) -> dict:
    """
    known: {issue_id: stored value of missing_field} for issues we already have.
    Returns the best candidate with its evidence, or a reason why there is none.
    """
    overlap = [p for p in raw_payloads if p.get("id") in known
               and normalize(missing_field, known[p["id"]]) not in (None, "", ())]
    if len(overlap) < MIN_KNOWN_OVERLAP:
        return {"candidate_key": None,
                "reason": f"only {len(overlap)} already-stored issues in this batch (need {MIN_KNOWN_OVERLAP}) to verify a rename"}

    keys = set().union(*(p.keys() for p in overlap)) - exclude_keys
    keys = {k for k in keys if not k.endswith("_url")}  # links, never the data itself

    best = None
    for key in sorted(keys):
        matched = sum(1 for p in overlap
                      if normalize(missing_field, p.get(key)) == normalize(missing_field, known[p["id"]]))
        if matched and (best is None or matched > best["matched"]):
            best = {"candidate_key": key, "matched": matched}

    if best is None:
        return {"candidate_key": None, "reason": "no field in the new payloads holds the values we stored"}
    best["overlap"] = len(overlap)
    best["confidence"] = round(best["matched"] / len(overlap), 3)
    return best


def attempt_self_heal(repo: str, alerts: list[dict], raw_payloads: list[dict],
                      known_values: dict[str, dict[int, object]], core_fields: list[str]) -> list[dict]:
    """known_values: {missing_field: {issue_id: stored value}} (see drift.check_drift)."""
    decisions = []
    for alert in alerts:
        if alert["severity"] != "critical":
            continue
        field = alert["field"]
        if field not in FIELD_TO_COLUMN:
            decisions.append({"field": field, "status": "needs_review",
                              "reason": f"'{field}' can't be verified against stored data"})
            continue

        proposal = propose_mapping(field, raw_payloads, known_values.get(field, {}), set(core_fields))
        decision = {"field": field, **proposal}
        evidence = proposal.get("confidence", 0)
        if proposal["candidate_key"] and evidence >= AUTO_APPLY_EVIDENCE:
            decision["status"] = "self_healed"
            log.info("Self-heal: '%s' -> '%s' (%s/%s known issues match).",
                     field, proposal["candidate_key"], proposal["matched"], proposal["overlap"])
        else:
            decision["status"] = "needs_review"
            if proposal["candidate_key"] and evidence < REVIEW_MIN_EVIDENCE:
                decision["reason"] = (f"best candidate '{proposal['candidate_key']}' matched only "
                                      f"{proposal['matched']}/{proposal['overlap']} known issues")
                decision["candidate_key"] = None  # too weak to even suggest
            elif proposal["candidate_key"]:
                decision["reason"] = (f"'{proposal['candidate_key']}' matched {proposal['matched']}/"
                                      f"{proposal['overlap']} known issues -- below auto-apply bar, queued for review")
            log.warning("Self-heal: '%s' needs review: %s", field, decision.get("reason"))
        decisions.append(decision)
    return decisions


# --- Synthetic demo (randomized each run) --------------------------------

# Only fields that either (a) are REQUIRED by GitHubIssue's schema (state,
# title) -- so losing them causes a real, visible extraction failure -- or
# (b) are optional fields that still matter (labels, user, comments, body).
# Required fields are always included in the drift so the extraction proof
# below is always meaningful; the optional field is a coin flip, for variety.
# Strong = plain rename (values identical -> evidence 100%, auto-healed).
REQUIRED_FIELD_STRONG_ALIASES = {"state": ["status", "lifecycle_state"], "title": ["title_text", "headline"]}
# "Weak" = renamed AND reformatted upstream: the values no longer match what
# we stored, so there's no evidence -- correctly left for a human.
REQUIRED_FIELD_WEAK_ALIASES = {"state": ["issue_status"], "title": ["summary"]}
WEAK_TRANSFORMS = {"state": str.upper, "title": lambda t: f"[{t}]"}

OPTIONAL_FIELD_ALIASES = {
    "labels": ["tags", "label_names"],
    "user": ["author", "reporter"],
    "comments": ["comment_count", "num_comments"],
    "body": ["description", "content"],
}

SAMPLE_TITLES = [
    "Crash when loading large files", "Feature request: dark mode support",
    "Memory leak in background worker", "Documentation is out of date",
    "Support for custom themes", "Unexpected exception on save",
    "Improve error messages for invalid input", "Add pagination to search results",
    "Slow performance with large datasets", "Broken link in README",
    "Race condition in concurrent writes", "Config file not respected on Windows",
    "Add retry logic for flaky network calls", "Typo in CLI help text",
    "Timezone handling is inconsistent", "Export feature produces malformed CSV",
    "Add support for custom plugins", "Startup time regression in latest release",
    "Missing validation on empty input", "Inconsistent naming across modules",
]


def _build_sample_payloads(n: int = 20) -> list[dict]:
    payloads = []
    for i in range(n):
        payloads.append({
            "id": 5000 + i,
            "number": 300 + i,
            "title": SAMPLE_TITLES[i % len(SAMPLE_TITLES)],
            "body": f"Steps to reproduce and context for issue #{300 + i}.",
            "state": "open" if i % 3 == 0 else "closed",
            "labels": [{"name": "bug"}] if i % 2 == 0 else [{"name": "enhancement"}],
            "user": {"login": f"contributor{i % 6}"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "comments": i % 5,
        })
    return payloads


def _pick_drift_plan() -> dict:
    """
    Randomly decides which field(s) drift THIS run and how. Always drifts
    exactly one REQUIRED field (state or title) so the extraction proof is
    always meaningful, and sometimes also one optional field for variety.

    The required field's outcome is explicitly weighted (65% strong alias /
    25% weak alias / 10% dropped outright) rather than left to fall out of
    a uniform random choice -- a uniform choice meant "needs review" won
    more often than not, which made the demo look broken rather than
    occasionally-honest. This keeps real variety (you'll still sometimes
    see a genuine "needs review" outcome) without it dominating.
    """
    plan = {}

    req_field = random.choice(list(REQUIRED_FIELD_STRONG_ALIASES.keys()))
    roll = random.random()
    if roll < 0.65:
        plan[req_field] = random.choice(REQUIRED_FIELD_STRONG_ALIASES[req_field])
    elif roll < 0.90:
        plan[req_field] = random.choice(REQUIRED_FIELD_WEAK_ALIASES[req_field])
    else:
        plan[req_field] = None  # dropped, no replacement -- genuinely needs review

    if random.random() < 0.6:
        opt_field = random.choice(list(OPTIONAL_FIELD_ALIASES.keys()))
        if random.random() < 0.7:
            plan[opt_field] = random.choice(OPTIONAL_FIELD_ALIASES[opt_field])
        else:
            plan[opt_field] = None

    return plan


def _apply_drift(payload: dict, plan: dict) -> dict:
    drifted = dict(payload)
    for field, new_name in plan.items():
        if field in drifted:
            value = drifted.pop(field)
            if new_name:
                if new_name in REQUIRED_FIELD_WEAK_ALIASES.get(field, []):
                    value = WEAK_TRANSFORMS[field](value)
                drifted[new_name] = value
    return drifted


# Unrelated to the batch-level schema drift above -- real data also has
# scattered, per-record quality problems independent of any schema change.
# Sprinkling a few of these in is what makes results NOT always a clean
# 0/20 or 20/20: the schema-drift fix is uniform across the batch (same
# field renamed the same way everywhere), but a record with its own,
# unrelated bad value stays broken regardless of whether that fix applied.
PER_RECORD_QUALITY_ISSUES = [
    ("title", ""),          # blank title -- fails the required min_length validator
    ("state", "archived"),  # invalid state value -- fails the state_must_be_known validator
]


def _apply_random_quality_issues(payloads: list[dict]) -> tuple[list[dict], dict[int, str]]:
    affected: dict[int, str] = {}
    num_issues = random.choice([0, 0, 1, 1, 2, 3])
    if num_issues == 0:
        return payloads, affected

    indices = random.sample(range(len(payloads)), k=min(num_issues, len(payloads)))
    for idx in indices:
        # Only pick an issue for a field that's actually still present in
        # this record -- if the batch drift already renamed/removed it,
        # setting it here would wrongly re-add a field that shouldn't exist.
        applicable = [(f, v) for f, v in PER_RECORD_QUALITY_ISSUES if f in payloads[idx]]
        if not applicable:
            continue
        field, bad_value = random.choice(applicable)
        payloads[idx] = dict(payloads[idx])
        payloads[idx][field] = bad_value
        affected[idx] = f"unrelated issue: '{field}' is {bad_value!r} in this specific record"
    return payloads, affected


def run_synthetic_demo() -> dict:
    from pydantic import ValidationError

    from pipeline.drift import CORE_FIELDS, compute_fingerprint, detect_drift
    from pipeline.schema import GitHubIssue

    baseline_payloads = _build_sample_payloads(20)
    drift_plan = _pick_drift_plan()
    drifted_payloads = [_apply_drift(p, drift_plan) for p in baseline_payloads]
    drifted_payloads, quality_issues = _apply_random_quality_issues(drifted_payloads)

    baseline_fp = compute_fingerprint(baseline_payloads)
    drifted_fp = compute_fingerprint(drifted_payloads)
    alerts = detect_drift(drifted_fp, baseline_fp)
    # The baseline records play the role of "issues we already store".
    known_values = {f: {p["id"]: p.get(f) for p in baseline_payloads} for f in CORE_FIELDS}
    decisions = attempt_self_heal("demo/repo", alerts, drifted_payloads, known_values, CORE_FIELDS)

    healed_overrides = {d["field"]: d["candidate_key"] for d in decisions if d["status"] == "self_healed"}

    def try_extract(payload: dict, overrides: dict) -> dict:
        try:
            issue = GitHubIssue.from_github_payload(payload, repo="demo/repo", field_overrides=overrides)
            return {"success": True, "detail": f'state = "{issue.state}" -- extracted successfully'}
        except ValidationError as e:
            missing = sorted({str(err["loc"][0]) for err in e.errors()})
            return {"success": False, "detail": f"ValidationError on {missing} -- this record would be quarantined"}

    # Runs the REAL extraction (with and without the healed mapping) against
    # EVERY one of the 20 drifted records, not just one cherry-picked
    # example -- this is what proves the fix generalizes across the batch.
    records = []
    for i, (expected, actual) in enumerate(zip(baseline_payloads, drifted_payloads)):
        records.append({
            "index": i,
            "number": expected["number"],
            "title": expected["title"],
            # Only the fields that actually changed this run, expected vs
            # actual key/value -- kept small on purpose, for compact cards.
            "changed_expected": {f: expected[f] for f in drift_plan if f in expected},
            "changed_actual": {
                new_name: actual[new_name]
                for field, new_name in drift_plan.items()
                if new_name and new_name in actual
            },
            "extraction": {
                "before": try_extract(actual, {}),
                "after": try_extract(actual, healed_overrides),
            },
            "quality_issue": quality_issues.get(i),
        })

    return {
        "drift_plan": drift_plan,
        "alerts": alerts,
        "heal_decisions": decisions,
        "records": records,
    }


if __name__ == "__main__":
    result = run_synthetic_demo()
    print(f"Drift plan this run: {result['drift_plan']}\n")
    print(f"{len(result['alerts'])} drift alert(s):")
    for a in result["alerts"]:
        print(f"  [{a['severity']}] {a['description']}")
    print(f"\n{len(result['heal_decisions'])} self-heal decision(s):")
    for d in result["heal_decisions"]:
        print(f"  [{d['status']}] {d}")
    print(f"\n{len(result['records'])} records checked. Sample (record 0):")
    print(f"  before: {result['records'][0]['extraction']['before']}")
    print(f"  after:  {result['records'][0]['extraction']['after']}")