"""
Retrieval-quality eval: a small fixed set of test questions, scored by
an LLM-as-judge on whether the retrieved issues are actually relevant.

This is intentionally NOT an end-to-end answer-quality eval (that would
need hand-labeled ground-truth answers, which we don't have for a live
GitHub repo). Scoring retrieval relevance is the right-sized eval for
what this project needs to detect: "did the pipeline hand the agent
good context to work with" -- which is exactly what degrades silently
when a source's structure or content drifts.

Runnable standalone: `python -m pipeline.eval`
"""

import json
import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline.agent import GEMINI_MODEL, _get_api_key, retrieve_issues
from pipeline.log_utils import extract_text, get_run_logger

log = get_run_logger(__name__)

# Free-tier Gemini quota is 5 requests/minute. Ten test questions = ten
# judge calls in a loop -- without pacing, that blows through the quota
# around question 7-8 (as it did in testing), and unlike the agent (where
# fewer, spaced-out calls happen naturally), eval fires calls back-to-back
# by construction. Configurable via env var since a paid tier would allow
# a much higher rate.
GEMINI_RPM_LIMIT = int(os.environ.get("GEMINI_RPM_LIMIT", "5"))
_MIN_INTERVAL_SECONDS = 60.0 / GEMINI_RPM_LIMIT
_MAX_INTERVAL_SECONDS = 120.0  # cap so backoff can't grow unboundedly on repeated 429s
_MAX_RETRIES_PER_QUESTION = 4

# Fixed question set so eval scores are comparable run-over-run. Kept
# generic (not tied to specific issue numbers) since the underlying data
# changes as ingestion pulls in new issues.
TEST_QUESTIONS = [
    "What are some issues related to embeddings?",
    "Are there any issues about vector store integrations?",
    "What issues mention FAISS?",
    "Are there issues requesting support for new LLM providers?",
    "What issues are related to prompt templates?",
    "Are there any issues about chains or agents?",
    "What issues mention Hugging Face?",
    "Are there issues about saving or loading models?",
    "What issues discuss API rate limits or errors?",
    "Are there issues about documentation improvements?",
]

JUDGE_PROMPT = """You are evaluating a search system for a GitHub issue tracker.

Question: {question}

Retrieved results:
{results}

On a scale of 1-5, how relevant are these retrieved results to the question?
1 = completely irrelevant, 5 = highly relevant and directly answers the question.
A score of 3 given to genuinely empty results ("No relevant issues found") is
acceptable IF the question is plausible but the corpus may not cover it --
don't penalize honest "no results" as harshly as actively wrong results.

Respond with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


def _is_rate_limit_error(e: Exception) -> bool:
    text = str(e)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def _judge(llm, question: str, results: list[dict]) -> dict:
    if not results:
        results_text = "No relevant issues found."
    else:
        results_text = "\n".join(f"- Issue #{r['number']}: {r['snippet'][:150]}" for r in results)

    prompt = JUDGE_PROMPT.format(question=question, results=results_text)
    response = llm.invoke(prompt)

    text = extract_text(response.content)
    # Strip markdown code fences if the model wraps its JSON in them
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(text)
        return {"score": int(parsed["score"]), "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("Judge response wasn't valid JSON, defaulting to score=0: %s", text[:200])
        return {"score": 0, "reason": f"judge parse failure: {e}"}


def run_eval() -> dict:
    api_key = _get_api_key()
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=api_key)

    results = []
    last_call_at = 0.0
    # Adaptive: starts at the theoretical min interval for GEMINI_RPM_LIMIT,
    # but DOUBLES (up to _MAX_INTERVAL_SECONDS) the first time a 429 is
    # actually hit, and stays doubled for the rest of the run rather than
    # resetting -- if the quota was tighter than advertised once, it's
    # likely to be again, so keep the safer pace instead of immediately
    # racing back to the theoretical minimum.
    current_interval = _MIN_INTERVAL_SECONDS

    for question in TEST_QUESTIONS:
        elapsed = time.monotonic() - last_call_at
        if elapsed < current_interval:
            wait = current_interval - elapsed
            log.info("Pacing: waiting %.1fs before next judge call (interval=%.1fs)", wait, current_interval)
            time.sleep(wait)

        retrieved = retrieve_issues(question, n_results=5)

        judgment = None
        for attempt in range(1, _MAX_RETRIES_PER_QUESTION + 1):
            try:
                judgment = _judge(llm, question, retrieved)
                last_call_at = time.monotonic()
                break
            except Exception as e:
                if not _is_rate_limit_error(e) or attempt == _MAX_RETRIES_PER_QUESTION:
                    raise

                current_interval = min(current_interval * 2, _MAX_INTERVAL_SECONDS)
                log.warning(
                    "Rate limited (attempt %s/%s). Doubling pacing interval to %.1fs and retrying.",
                    attempt, _MAX_RETRIES_PER_QUESTION, current_interval,
                )
                time.sleep(current_interval)
                last_call_at = time.monotonic()

        results.append({
            "question": question,
            "retrieved_count": len(retrieved),
            "score": judgment["score"],
            "reason": judgment["reason"],
        })
        log.info("Eval: %r -> score=%s (%s)", question, judgment["score"], judgment["reason"])

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "avg_score": round(avg_score, 2),
        "num_questions": len(TEST_QUESTIONS),
        "results": results,
    }
    log.info("Eval complete: avg_score=%s across %s questions", summary["avg_score"], summary["num_questions"])
    return summary


if __name__ == "__main__":
    summary = run_eval()
    print(json.dumps(summary, indent=2))