"""
Self-healing schema adaptation.

When drift.py flags a field as "disappeared" (presence dropped to ~0),
this module tries to find its likely replacement in the new payload
shape -- e.g. GitHub renaming `state` to `status` -- by fuzzy-matching
key names and checking type compatibility against the pre-drift baseline.

High-confidence matches are auto-applied as a field mapping override
(persisted in Postgres), so the NEXT ingestion run correctly extracts
that field again without a human touching code. Low-confidence or
no-candidate cases are logged as needing human review instead of
guessing -- a wrong auto-mapping would be worse than no mapping.

Runnable standalone (synthetic drift scenario):
`python -m pipeline.self_heal`
"""

import difflib

from pipeline.log_utils import get_run_logger

log = get_run_logger(__name__)

# A field name has to be at least this similar to the missing field's name
# to be considered a candidate at all (difflib ratio, 0-1).
CANDIDATE_MIN_NAME_SIMILARITY = 0.5

# Combined confidence (name similarity + presence + type match) needed to
# auto-apply a mapping without human review. Deliberately high -- a wrong
# auto-healed mapping silently corrupts data in a DIFFERENT way than the
# drift it's supposed to fix, so we'd rather under-heal than over-heal.
AUTO_APPLY_CONFIDENCE_THRESHOLD = 0.75


def propose_mapping(
    missing_field: str,
    raw_payloads: list[dict],
    expected_types: list[str],
    exclude_keys: set[str],
) -> dict | None:
    """
    Looks across all keys present in raw_payloads (excluding ones already
    claimed by other core fields) for the best candidate replacement for
    missing_field, scoring on: key name similarity, how consistently the
    candidate key is present, and whether its value type matches what
    missing_field used to have (from the baseline fingerprint).
    """
    if not raw_payloads:
        return None

    all_keys = set()
    for payload in raw_payloads:
        all_keys.update(payload.keys())
    candidate_keys = all_keys - exclude_keys

    best = None
    for key in candidate_keys:
        name_similarity = difflib.SequenceMatcher(None, missing_field.lower(), key.lower()).ratio()
        if name_similarity < CANDIDATE_MIN_NAME_SIMILARITY:
            continue

        present = 0
        types_seen = set()
        for payload in raw_payloads:
            value = payload.get(key)
            if value is not None:
                present += 1
                types_seen.add(type(value).__name__)

        presence_rate = present / len(raw_payloads)
        type_match = bool(types_seen & set(expected_types)) if expected_types else True

        # Weighted blend: name similarity matters most (it's the strongest
        # signal a rename happened), presence rate rules out sparse/optional
        # fields, type match is a smaller boost/tiebreaker.
        confidence = round(name_similarity * 0.6 + presence_rate * 0.3 + (0.1 if type_match else 0.0), 3)

        if best is None or confidence > best["confidence"]:
            best = {
                "candidate_key": key,
                "name_similarity": round(name_similarity, 3),
                "presence_rate": round(presence_rate, 3),
                "type_match": type_match,
                "confidence": confidence,
            }

    return best


def attempt_self_heal(
    repo: str,
    alerts: list[dict],
    raw_payloads: list[dict],
    baseline_fingerprint: dict,
    core_fields: list[str],
) -> list[dict]:
    """
    For every critical "field disappeared" alert, tries to propose a
    replacement mapping. Returns a list of decisions -- each either
    'self_healed' (high confidence, auto-applied) or 'needs_review'
    (low/no confidence, left for a human).
    """
    decisions = []
    exclude_keys = set(core_fields)

    for alert in alerts:
        if alert["severity"] != "critical":
            continue

        field = alert["field"]
        expected_types = baseline_fingerprint.get(field, {}).get("types", [])
        proposal = propose_mapping(field, raw_payloads, expected_types, exclude_keys)

        if proposal is None:
            decisions.append({
                "field": field,
                "status": "needs_review",
                "reason": "no plausible replacement field found in the new payload shape",
            })
            log.warning("Self-heal: no candidate found for missing field '%s' -- needs human review.", field)
            continue

        decision = {"field": field, **proposal}
        if proposal["confidence"] >= AUTO_APPLY_CONFIDENCE_THRESHOLD:
            decision["status"] = "self_healed"
            log.info(
                "Self-heal: auto-mapping '%s' -> '%s' (confidence=%.2f). Will apply from next run.",
                field, proposal["candidate_key"], proposal["confidence"],
            )
        else:
            decision["status"] = "needs_review"
            log.warning(
                "Self-heal: best candidate for '%s' is '%s' but confidence too low (%.2f < %.2f) to auto-apply.",
                field, proposal["candidate_key"], proposal["confidence"], AUTO_APPLY_CONFIDENCE_THRESHOLD,
            )
        decisions.append(decision)

    return decisions


if __name__ == "__main__":
    # Synthetic demo: `state` gets renamed to `status` (should self-heal --
    # near-identical name, matching type, high presence), while `user`
    # disappears with NO replacement candidate at all (should need review,
    # not get force-mapped onto some unrelated field).
    from pipeline.drift import CORE_FIELDS, compute_fingerprint, detect_drift

    baseline_payloads = [
        {"id": i, "number": i, "title": f"issue {i}", "body": "some body", "state": "open",
         "labels": [{"name": "bug"}], "user": {"login": "someone"}, "created_at": "2024-01-01T00:00:00Z",
         "updated_at": "2024-01-01T00:00:00Z", "comments": 2}
        for i in range(20)
    ]
    drifted_payloads = [
        {"id": i, "number": i, "title": f"issue {i}", "body": "some body",
         "status": "open",  # <- renamed from "state"
         "labels": [{"name": "bug"}], "created_at": "2024-01-01T00:00:00Z",  # <- "user" gone, no replacement
         "updated_at": "2024-01-01T00:00:00Z", "comments": 2}
        for i in range(20)
    ]

    baseline_fp = compute_fingerprint(baseline_payloads)
    drifted_fp = compute_fingerprint(drifted_payloads)
    alerts = detect_drift(drifted_fp, baseline_fp)

    print(f"{len(alerts)} drift alert(s):")
    for a in alerts:
        print(f"  [{a['severity']}] {a['description']}")

    decisions = attempt_self_heal("demo/repo", alerts, drifted_payloads, baseline_fp, CORE_FIELDS)
    print(f"\n{len(decisions)} self-heal decision(s):")
    for d in decisions:
        print(f"  [{d['status']}] {d}")