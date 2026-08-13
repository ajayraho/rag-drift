"""
Unit tests for pipeline/schema.py -- the Pydantic validation layer that
decides whether a raw GitHub payload is a well-formed issue, and the
field_overrides mechanism self-healing relies on.
"""

import pytest
from pydantic import ValidationError

from pipeline.schema import GitHubIssue, NotAnIssueError


def make_raw_issue(**overrides) -> dict:
    """A minimal, valid raw GitHub issue payload, with overrides applied."""
    base = {
        "id": 123,
        "number": 45,
        "title": "Something is broken",
        "body": "Steps to reproduce...",
        "state": "open",
        "labels": [{"name": "bug"}],
        "user": {"login": "someone"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "comments": 3,
    }
    base.update(overrides)
    return base


class TestValidIssueExtraction:
    def test_valid_payload_extracts_correctly(self):
        issue = GitHubIssue.from_github_payload(make_raw_issue(), repo="acme/widgets")
        assert issue.id == 123
        assert issue.number == 45
        assert issue.title == "Something is broken"
        assert issue.state == "open"
        assert issue.labels == ["bug"]
        assert issue.author == "someone"
        assert issue.comments_count == 3
        assert issue.repo == "acme/widgets"

    def test_labels_as_plain_strings_also_work(self):
        # GitHub's real format is list-of-dicts, but the extractor should
        # tolerate a list of plain strings too (defensive, not just happy-path).
        issue = GitHubIssue.from_github_payload(
            make_raw_issue(labels=["bug", "urgent"]), repo="acme/widgets"
        )
        assert issue.labels == ["bug", "urgent"]

    def test_labels_as_comma_string_are_split(self):
        # This is the exact shape a self-healed mapping onto a delimited-
        # string field would produce -- must not crash, must split sensibly.
        issue = GitHubIssue.from_github_payload(
            make_raw_issue(labels="bug, urgent"), repo="acme/widgets"
        )
        assert issue.labels == ["bug", "urgent"]

    def test_missing_optional_fields_default_sensibly(self):
        raw = make_raw_issue()
        del raw["body"]
        del raw["user"]
        issue = GitHubIssue.from_github_payload(raw, repo="acme/widgets")
        assert issue.body is None
        assert issue.author is None
        assert issue.state == "open"  # required field untouched, still fine


class TestQuarantineWorthyPayloads:
    """
    These are the payloads that SHOULD fail validation -- this is the
    core proof that quarantine has something real to catch, not just an
    unused code path.
    """

    def test_missing_required_state_raises(self):
        raw = make_raw_issue()
        del raw["state"]
        with pytest.raises(ValidationError) as exc_info:
            GitHubIssue.from_github_payload(raw, repo="acme/widgets")
        assert any(err["loc"] == ("state",) for err in exc_info.value.errors())

    def test_blank_title_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            GitHubIssue.from_github_payload(make_raw_issue(title="   "), repo="acme/widgets")
        assert any(err["loc"] == ("title",) for err in exc_info.value.errors())

    def test_missing_title_raises(self):
        raw = make_raw_issue()
        del raw["title"]
        with pytest.raises(ValidationError):
            GitHubIssue.from_github_payload(raw, repo="acme/widgets")

    def test_invalid_state_value_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            GitHubIssue.from_github_payload(make_raw_issue(state="archived"), repo="acme/widgets")
        errors = exc_info.value.errors()
        assert any("state" in str(err["loc"]) for err in errors)

    def test_missing_created_at_raises(self):
        raw = make_raw_issue()
        del raw["created_at"]
        with pytest.raises(ValidationError):
            GitHubIssue.from_github_payload(raw, repo="acme/widgets")

    def test_pull_request_is_rejected_as_not_an_issue(self):
        # PRs are an expected, by-design exclusion -- NOT a quarantine
        # case. This distinct exception type is what lets fetch.py route
        # PRs to a "skipped" counter instead of muddying quarantine stats.
        raw = make_raw_issue(pull_request={"url": "https://github.com/acme/widgets/pull/45"})
        with pytest.raises(NotAnIssueError):
            GitHubIssue.from_github_payload(raw, repo="acme/widgets")


class TestFieldOverrides:
    """
    Proves the self-healing mechanism's core promise: a healed mapping
    recovers extraction for a record that would otherwise be quarantined.
    """

    def test_renamed_field_fails_without_override(self):
        raw = make_raw_issue()
        raw["status"] = raw.pop("state")  # simulate the source renaming `state`
        with pytest.raises(ValidationError):
            GitHubIssue.from_github_payload(raw, repo="acme/widgets", field_overrides={})

    def test_renamed_field_recovers_with_override(self):
        raw = make_raw_issue()
        raw["status"] = raw.pop("state")
        issue = GitHubIssue.from_github_payload(
            raw, repo="acme/widgets", field_overrides={"state": "status"}
        )
        assert issue.state == "open"

    def test_override_pointing_at_missing_key_still_fails(self):
        # The override exists, but the key it points to genuinely isn't
        # present -- should fail cleanly, not crash with a KeyError.
        raw = make_raw_issue()
        del raw["state"]
        with pytest.raises(ValidationError):
            GitHubIssue.from_github_payload(
                raw, repo="acme/widgets", field_overrides={"state": "status"}
            )