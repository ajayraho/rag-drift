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
    labels: list[str] = Field(default_factory=list)   # why not labels: list[str] = []  because mutable default arguments are a bad idea in Python. If we used labels: list[str] = [], then all instances of GitHubIssue would share the same list object as the default value for labels. This could lead to unexpected behavior if one instance modifies the list, as it would affect all other instances that use the default value. Using default_factory=list ensures that each instance gets its own separate list. 
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
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()   # sha256 because it's fast and collision-resistant enough for our purposes

    @classmethod   # like a static method
    def from_github_payload(cls, raw: dict, repo: str) -> "GitHubIssue":
        """
        Maps GitHub's raw API JSON shape into our schema.
        This is the ONE place that knows GitHub's field names, so if GitHub
        renames/restructures something, this is the single point of failure
        (and the thing our drift detector on Day 3 will be watching).
        """
        # GitHub's issues endpoint also returns pull requests -- they have a
        # "pull_request" key when it's actually a PR, not a plain issue.
        if "pull_request" in raw:
            raise NotAnIssueError("payload is a pull request, not an issue")

        return cls(
            id=raw["id"],
            repo=repo,
            number=raw["number"],
            title=raw["title"],
            body=raw.get("body"),
            state=raw["state"],
            labels=[l["name"] if isinstance(l, dict) else l for l in raw.get("labels", [])],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            author=(raw.get("user") or {}).get("login"),
            comments_count=raw.get("comments", 0),
        )