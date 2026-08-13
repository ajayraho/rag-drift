"""
Unit tests for pipeline/drift.py -- fingerprinting and diffing raw
payload shapes to catch structural drift Pydantic validation alone
wouldn't necessarily notice.
"""

from pipeline.drift import compute_fingerprint, detect_drift


def make_payloads(n=20, **field_overrides) -> list[dict]:
    payloads = []
    for i in range(n):
        p = {
            "id": i, "number": i, "title": f"issue {i}", "body": "body text",
            "state": "open" if i % 2 == 0 else "closed",
            "labels": [{"name": "bug"}], "user": {"login": "someone"},
            "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
            "comments": i % 3,
        }
        for field, transform in field_overrides.items():
            transform(p)
        payloads.append(p)
    return payloads


class TestComputeFingerprint:
    def test_fully_present_field_has_full_presence_rate(self):
        fp = compute_fingerprint(make_payloads())
        assert fp["state"]["presence_rate"] == 1.0
        assert fp["state"]["types"] == ["str"]

    def test_empty_payload_list_returns_zero_presence(self):
        fp = compute_fingerprint([])
        assert fp["state"]["presence_rate"] == 0.0

    def test_partially_present_field_has_partial_presence_rate(self):
        payloads = make_payloads(20)
        for p in payloads[:5]:
            del p["body"]
        fp = compute_fingerprint(payloads)
        assert fp["body"]["presence_rate"] == 0.75

    def test_mixed_types_are_all_recorded(self):
        payloads = make_payloads(10)
        payloads[0]["comments"] = "3"  # string instead of int, e.g. a source quirk
        fp = compute_fingerprint(payloads)
        assert "int" in fp["comments"]["types"]
        assert "str" in fp["comments"]["types"]


class TestDetectDrift:
    def test_no_drift_between_identical_fingerprints(self):
        payloads = make_payloads()
        fp = compute_fingerprint(payloads)
        alerts = detect_drift(fp, fp)
        assert alerts == []

    def test_field_disappearing_is_flagged_critical(self):
        baseline = compute_fingerprint(make_payloads())
        drifted_payloads = make_payloads()
        for p in drifted_payloads:
            del p["state"]
        drifted = compute_fingerprint(drifted_payloads)

        alerts = detect_drift(drifted, baseline)
        critical = [a for a in alerts if a["field"] == "state" and a["severity"] == "critical"]
        assert len(critical) == 1
        assert "presence dropped" in critical[0]["description"]

    def test_field_type_change_is_flagged_warning(self):
        baseline = compute_fingerprint(make_payloads())
        drifted_payloads = make_payloads()
        for p in drifted_payloads:
            p["labels"] = "bug,urgent"  # list -> string
        drifted = compute_fingerprint(drifted_payloads)

        alerts = detect_drift(drifted, baseline)
        warnings = [a for a in alerts if a["field"] == "labels" and a["severity"] == "warning"]
        assert len(warnings) == 1

    def test_minor_presence_fluctuation_is_not_flagged(self):
        # A field that's merely optional and sometimes absent shouldn't
        # trigger a false alarm every run just from natural variance.
        baseline = compute_fingerprint(make_payloads())
        slightly_different = make_payloads(20)
        del slightly_different[0]["body"]  # 95% presence, still high
        drifted = compute_fingerprint(slightly_different)

        alerts = detect_drift(drifted, baseline)
        assert not any(a["field"] == "body" for a in alerts)