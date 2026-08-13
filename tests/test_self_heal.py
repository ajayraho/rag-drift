"""
Unit tests for pipeline/self_heal.py -- the fuzzy-matching engine that
proposes field-mapping corrections, and the confidence threshold that
decides auto-heal vs. needs-review.
"""

from pipeline.self_heal import (
    AUTO_APPLY_CONFIDENCE_THRESHOLD,
    attempt_self_heal,
    propose_mapping,
)


def make_payloads(n=20, **field_overrides) -> list[dict]:
    payloads = []
    for i in range(n):
        p = {
            "id": i, "number": i, "title": f"issue {i}", "body": "body text",
            "state": "open", "labels": [{"name": "bug"}], "user": {"login": "someone"},
            "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
            "comments": i % 3,
        }
        for field, transform in field_overrides.items():
            transform(p)
        payloads.append(p)
    return payloads


class TestProposeMapping:
    def test_finds_close_rename(self):
        payloads = make_payloads()
        for p in payloads:
            p["status"] = p.pop("state")

        proposal = propose_mapping(
            "state", payloads, expected_types=["str"], exclude_keys={"id", "number", "title"}
        )
        assert proposal is not None
        assert proposal["candidate_key"] == "status"
        assert proposal["confidence"] >= AUTO_APPLY_CONFIDENCE_THRESHOLD

    def test_returns_none_when_field_genuinely_gone(self):
        payloads = make_payloads()
        for p in payloads:
            del p["user"]  # no replacement key introduced anywhere

        proposal = propose_mapping(
            "user", payloads, expected_types=["dict"], exclude_keys={"id", "number", "title"}
        )
        assert proposal is None

    def test_excluded_keys_are_never_proposed(self):
        # Even if "title" is textually similar-ish to some field, it must
        # never be proposed as a replacement -- it's a real, independent
        # core field, not a candidate rename target.
        payloads = make_payloads()
        for p in payloads:
            p["status"] = p.pop("state")

        proposal = propose_mapping(
            "state", payloads, expected_types=["str"],
            exclude_keys={"id", "number", "title", "body", "labels", "user",
                          "created_at", "updated_at", "comments", "state"},
        )
        assert proposal["candidate_key"] != "title"

    def test_dissimilar_key_names_score_low_confidence(self):
        payloads = make_payloads()
        for p in payloads:
            p["completely_unrelated_field_name"] = p.pop("state")

        proposal = propose_mapping(
            "state", payloads, expected_types=["str"], exclude_keys={"id", "number", "title"}
        )
        # Either no candidate at all, or one that scores below auto-apply.
        if proposal is not None:
            assert proposal["confidence"] < AUTO_APPLY_CONFIDENCE_THRESHOLD


class TestAttemptSelfHeal:
    CORE_FIELDS = ["id", "number", "title", "body", "state", "labels",
                   "user", "created_at", "updated_at", "comments"]

    def test_high_confidence_rename_is_self_healed(self):
        payloads = make_payloads()
        for p in payloads:
            p["status"] = p.pop("state")

        alerts = [{"field": "state", "severity": "critical", "description": "presence dropped"}]
        from pipeline.drift import compute_fingerprint
        baseline_fp = compute_fingerprint(make_payloads())

        decisions = attempt_self_heal("test/repo", alerts, payloads, baseline_fp, self.CORE_FIELDS)
        assert len(decisions) == 1
        assert decisions[0]["status"] == "self_healed"
        assert decisions[0]["candidate_key"] == "status"

    def test_no_candidate_needs_review_not_forced(self):
        payloads = make_payloads()
        for p in payloads:
            del p["user"]

        alerts = [{"field": "user", "severity": "critical", "description": "presence dropped"}]
        from pipeline.drift import compute_fingerprint
        baseline_fp = compute_fingerprint(make_payloads())

        decisions = attempt_self_heal("test/repo", alerts, payloads, baseline_fp, self.CORE_FIELDS)
        assert len(decisions) == 1
        assert decisions[0]["status"] == "needs_review"
        assert "candidate_key" not in decisions[0]  # no field mapping gets fabricated

    def test_non_critical_alerts_are_ignored(self):
        alerts = [{"field": "labels", "severity": "warning", "description": "type changed"}]
        from pipeline.drift import compute_fingerprint
        baseline_fp = compute_fingerprint(make_payloads())

        decisions = attempt_self_heal("test/repo", alerts, make_payloads(), baseline_fp, self.CORE_FIELDS)
        assert decisions == []