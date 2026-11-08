"""
The golden question set: a fixed list of questions per repo, each tied
to the issue(s) that SHOULD come back when you search for it. It's what
turns "the answers look okay" into a number we can compare across
changes (new embedding model, chunking, hybrid search, ...).

How it's built (explainable in one breath): take closed issues that
actually got discussed, have an LLM write the question a developer
would type when they hit that problem, and record the source issue as
the correct answer. A human then reviews the file -- fix or delete bad
questions, set "reviewed": true.

Files live in pipeline/golden/<owner>__<name>.jsonl (inside the package
so Airflow and the API image both see them with no extra mounts), one
JSON object per line:

  {"question": "...", "expected": [1234], "source_issue": 1234, "reviewed": false}

`expected` is a list so a reviewer can add duplicates/related issues
that are equally correct answers.

CLI:
  python -m pipeline.golden_set generate --repo owner/name [--n 40]
  python -m pipeline.golden_set stats    --repo owner/name
"""

import hashlib
import json
import random
import sys
import time
from pathlib import Path

from pipeline.log_utils import extract_text, get_run_logger
from pipeline.repo_name import validate_repo

log = get_run_logger(__name__)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

QUESTION_PROMPT = """Below is a closed GitHub issue from the {repo} repository.

Write ONE question that a developer who is running into this same problem would \
type into a search box for this project's issues. Rules:
- Describe the problem in your own words; do NOT copy the issue title.
- Do not mention the issue number, the author, or that it is closed.
- Keep it under 25 words.

Title: {title}

Body:
{body}

Respond with ONLY a JSON object: {{"question": "<the question>"}}"""


def golden_path(repo: str) -> Path:
    owner, name = validate_repo(repo).split("/")
    return GOLDEN_DIR / f"{owner}__{name}.jsonl"


def load_golden_set(repo: str) -> list[dict]:
    """Returns [] if this repo has no golden set yet. Skips blank lines."""
    path = golden_path(repo)
    if not path.exists():
        return []
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("question") or not item.get("expected"):
            raise ValueError(f"{path.name}:{line_no}: every entry needs 'question' and a non-empty 'expected'")
        item["expected"] = [int(n) for n in item["expected"]]
        items.append(item)
    return items


def golden_set_version(repo: str) -> str | None:
    """
    Short content hash of the golden file. Logged with every eval run so a
    score is always tied to the exact question set it was measured on --
    editing the set changes the version, and old/new scores aren't
    silently compared.
    """
    path = golden_path(repo)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def save_golden_set(repo: str, items: list[dict]) -> Path:
    path = golden_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def _parse_question(text: str) -> str | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        question = json.loads(text).get("question", "")
    except (json.JSONDecodeError, AttributeError):
        return None
    question = " ".join(str(question).split())
    return question or None


def fetch_candidate_issues(conn, repo: str, limit: int, exclude: set[int], seed: int = 42) -> list[dict]:
    """
    Closed issues that are good question sources: a real description
    (200-6000 chars) and at least one comment (someone engaged with it).
    Shuffled with a fixed seed so re-running picks the same sample.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT number, title, body FROM issues
            WHERE repo = %s AND state = 'closed' AND comments_count >= 1
              AND length(coalesce(body, '')) BETWEEN 200 AND 6000
            ORDER BY number
            """,
            (repo,),
        )
        rows = [{"number": r[0], "title": r[1], "body": r[2]} for r in cur.fetchall()]
    rows = [r for r in rows if r["number"] not in exclude]
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def generate_golden_set(repo: str, n: int = 40, llm=None, conn=None, min_interval: float | None = None) -> dict:
    """
    Drafts up to `n` NEW questions and appends them to the repo's golden
    file (existing entries, reviewed or not, are kept untouched). New
    entries start with "reviewed": false.

    llm/conn are injectable for tests; by default this uses the Gemini
    model from agent.py and a normal pipeline DB connection.
    """
    from pipeline.db import get_connection

    repo = validate_repo(repo)
    existing = load_golden_set(repo)
    already_used = {item.get("source_issue") for item in existing}

    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        from pipeline.agent import GEMINI_MODEL, _get_api_key

        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, google_api_key=_get_api_key())
    if min_interval is None:
        from pipeline.eval import _MIN_INTERVAL_SECONDS as min_interval  # free-tier pacing

    if conn is None:
        with get_connection() as own_conn:
            candidates = fetch_candidate_issues(own_conn, repo, n, already_used)
    else:
        candidates = fetch_candidate_issues(conn, repo, n, already_used)

    added, failed = [], 0
    for i, issue in enumerate(candidates):
        if i:
            time.sleep(min_interval)
        prompt = QUESTION_PROMPT.format(repo=repo, title=issue["title"], body=issue["body"][:1500])
        try:
            question = _parse_question(extract_text(llm.invoke(prompt).content))
        except Exception as e:
            log.warning("Question generation failed for #%s: %s", issue["number"], e)
            question = None
        if not question:
            failed += 1
            continue
        added.append({
            "question": question,
            "expected": [issue["number"]],
            "source_issue": issue["number"],
            "reviewed": False,
        })
        log.info("Drafted question for #%s: %s", issue["number"], question)

    path = save_golden_set(repo, existing + added)
    return {"repo": repo, "path": str(path), "added": len(added), "failed": failed,
            "total": len(existing) + len(added), "candidates": len(candidates)}


def stats(repo: str) -> dict:
    items = load_golden_set(repo)
    reviewed = sum(1 for i in items if i.get("reviewed"))
    return {"repo": repo, "path": str(golden_path(repo)), "total": len(items),
            "reviewed": reviewed, "version": golden_set_version(repo)}


def _cli(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] not in ("generate", "stats") or "--repo" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    repo = argv[argv.index("--repo") + 1]
    if argv[0] == "generate":
        n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 40
        print(json.dumps(generate_golden_set(repo, n=n), indent=2))
        print("\nNext: open the file, fix/delete weak questions, set \"reviewed\": true.")
    else:
        print(json.dumps(stats(repo), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
