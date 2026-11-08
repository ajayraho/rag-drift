"""
Pydantic schema for GitHub issue records.

This is the gatekeeper between "raw JSON from GitHub's API" and
"row in our Postgres issues table". Anything that fails validation
here gets routed to quarantine instead of crashing the pipeline.
"""

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class NotAnIssueError(Exception):
    """
    Raised for payloads that are expected, by-design exclusions -- not
    malformed data. Right now this only means "it's a PR, not an issue".
    Kept separate from ValidationError/ValueError so the caller can route
    these to a `skipped` counter instead of `quarantine`. Mixing the two
    would make quarantine counts noisy and useless for drift alerting --
    PR volume naturally fluctuates and isn't a sign anything is broken.
    """
    pass


class GitHubIssue(BaseModel):
    id: int
    repo: str
    number: int
    title: str = Field(..., min_length=1)
    body: Optional[str] = None
    state: str
    labels: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    author: Optional[str] = None
    comments_count: int = 0

    @field_validator("state")
    @classmethod
    def state_must_be_known(cls, v: str) -> str:
        allowed = {"open", "closed"}
        if v not in allowed:
            raise ValueError(f"unexpected issue state: {v!r} (expected one of {allowed})")
        return v

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title is blank after stripping whitespace")
        return v

    def content_hash(self) -> str:
        """
        Hash of the fields that matter for "did this issue meaningfully change".
        Used on Day 2 to decide whether a doc needs re-embedding.
        We deliberately exclude comments_count from the hash inputs that would
        cause churn on every comment -- title/body/state/labels are what actually
        change the semantic content an agent would retrieve.
        """
        payload = f"{self.title}|{self.body or ''}|{self.state}|{','.join(sorted(self.labels))}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_github_payload(
        cls, raw: dict, repo: str, field_overrides: dict[str, str] | None = None
    ) -> "GitHubIssue":
        """
        Maps GitHub's raw API JSON shape into our schema.
        This is the ONE place that knows GitHub's field names, so if GitHub
        renames/restructures something, this is the single point of failure
        (and the thing our drift detector watches).

        field_overrides is a learned mapping from self_heal.py, e.g.
        {"state": "status"} if the drift detector previously figured out
        that `state` got renamed to `status` at the source. When the
        expected key is missing, we fall back to the override's key
        before giving up -- this is what makes self-healing actually take
        effect on the NEXT ingestion run after it's learned.
        """
        # GitHub's issues endpoint also returns pull requests -- they have a
        # "pull_request" key when it's actually a PR, not a plain issue.
        if "pull_request" in raw:
            raise NotAnIssueError("payload is a pull request, not an issue")

        overrides = field_overrides or {}

        def get(field_name: str, default=None):
            if field_name in raw and raw[field_name] is not None:
                return raw[field_name]
            override_key = overrides.get(field_name)
            if override_key and override_key in raw:
                return raw[override_key]
            return default

        try:
            raw_labels = get("labels")
            if isinstance(raw_labels, str):
                # Handles a healed mapping onto a field that stores labels as a
                # delimited string rather than a list -- e.g. "bug,enhancement".
                labels = [x.strip() for x in raw_labels.split(",") if x.strip()]
            elif raw_labels:
                labels = [l["name"] if isinstance(l, dict) else l for l in raw_labels]
            else:
                labels = []

            user = get("user")
            if user is not None and not isinstance(user, dict):
                raise ValueError(f"'user' should be an object, got {type(user).__name__}")
            author = (user or {}).get("login")
        except (TypeError, AttributeError) as e:
            # Shape surprises while EXTRACTING (labels is an int, a label is
            # a list, ...) used to escape as TypeError/AttributeError, which
            # the caller didn't catch -- one odd record crashed the whole run.
            # They're data problems, so surface them as ValueError -> quarantine.
            raise ValueError(f"unexpected payload shape: {type(e).__name__}: {e}") from e

        return cls(
            id=get("id"),
            repo=repo,
            number=get("number"),
            title=get("title"),
            body=get("body"),
            state=get("state"),
            labels=labels,
            created_at=get("created_at"),
            updated_at=get("updated_at"),
            author=author,
            comments_count=get("comments", 0),
        )
