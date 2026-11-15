"""Self-heal by value evidence: which new key holds the values we already store?"""

from pipeline.drift import CORE_FIELDS
from pipeline.self_heal import attempt_self_heal, normalize, propose_mapping, run_synthetic_demo


def _payloads(n=20, rename=None, transform=None):
    out = []
    for i in range(n):
        p = {"id": i, "number": i, "title": f"title {i}", "state": "open" if i % 2 else "closed",
             "labels": [{"name": "bug"}], "labels_url": f"https://x/{i}/labels",
             "user": {"login": f"u{i}"}, "created_at": "2024-01-01T00:00:00Z"}
        if rename:
            old, new = rename
            value = p.pop(old)
            p[new] = transform(value) if transform else value
        out.append(p)
    return out


KNOWN_STATE = {i: ("open" if i % 2 else "closed") for i in range(20)}
CRITICAL = lambda f: [{"field": f, "severity": "critical", "description": "gone"}]  # noqa: E731


def test_finds_rename_by_identical_values_even_with_unrelated_name():
    p = propose_mapping("state", _payloads(rename=("state", "lifecycle")), KNOWN_STATE, set(CORE_FIELDS))
    assert p["candidate_key"] == "lifecycle" and p["confidence"] == 1.0 and p["overlap"] == 20


def test_url_sibling_never_proposed():
    known = {i: ["bug"] for i in range(20)}
    p = propose_mapping("labels", _payloads(rename=("labels", "labels_gone")), known, set(CORE_FIELDS))
    assert p["candidate_key"] == "labels_gone"
    gone = [{k: v for k, v in x.items() if k != "labels"} for x in _payloads()]
    assert propose_mapping("labels", gone, known, set(CORE_FIELDS))["candidate_key"] is None


def test_too_few_known_issues_means_no_guess():
    p = propose_mapping("state", _payloads(rename=("state", "status")), {0: "closed", 1: "open"}, set(CORE_FIELDS))
    assert p["candidate_key"] is None and "need" in p["reason"]


def test_reformatted_values_have_no_evidence():
    p = propose_mapping("state", _payloads(rename=("state", "status"), transform=str.upper), KNOWN_STATE, set(CORE_FIELDS))
    assert p["candidate_key"] is None


def test_auto_apply_vs_review_vs_nothing():
    payloads = _payloads(rename=("state", "status"))
    healed = attempt_self_heal("o/r", CRITICAL("state"), payloads, {"state": KNOWN_STATE}, CORE_FIELDS)
    assert healed[0]["status"] == "self_healed" and healed[0]["candidate_key"] == "status"

    # 14/20 match (70%): plausible but not enough -> queued for review WITH the candidate
    for p in payloads[:6]:
        p["status"] = "weird"
    review = attempt_self_heal("o/r", CRITICAL("state"), payloads, {"state": KNOWN_STATE}, CORE_FIELDS)
    assert review[0]["status"] == "needs_review" and review[0]["candidate_key"] == "status"

    # 4/20 (20%): too weak to even suggest
    for p in payloads[:16]:
        p["status"] = "weird"
    weak = attempt_self_heal("o/r", CRITICAL("state"), payloads, {"state": KNOWN_STATE}, CORE_FIELDS)
    assert weak[0]["status"] == "needs_review" and weak[0]["candidate_key"] is None


def test_non_critical_alerts_ignored():
    assert attempt_self_heal("o/r", [{"field": "state", "severity": "warning"}], _payloads(), {}, CORE_FIELDS) == []


def test_normalize_puts_payload_and_db_values_in_same_form():
    from datetime import datetime, timezone
    assert normalize("labels", [{"name": "b"}, {"name": "a"}]) == normalize("labels", ["a", "b"]) == ("a", "b")
    assert normalize("user", {"login": "x"}) == normalize("user", "x")
    assert normalize("created_at", "2024-01-01T00:00:00Z") == normalize(
        "created_at", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert normalize("comments", "3") == 3 and normalize("comments", "x") is None


def test_demo_runs_and_heals_with_evidence():
    for _ in range(20):
        result = run_synthetic_demo()
        for d in result["heal_decisions"]:
            if d["status"] == "self_healed":
                assert d["matched"] >= 0.9 * d["overlap"]
