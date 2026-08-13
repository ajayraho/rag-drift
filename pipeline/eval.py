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

# Free-tier Gemini quota is tight. Ten test questions = ten judge calls in
# a loop -- without pacing, that can blow through the quota partway
# through. Configurable via env var since a paid tier would allow a much
# higher rate.
GEMINI_RPM_LIMIT = int(os.environ.get("GEMINI_RPM_LIMIT", "5"))
_MIN_INTERVAL_SECONDS = 60.0 / GEMINI_RPM_LIMIT
_MAX_INTERVAL_SECONDS = 120.0
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
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(text)
        return {"score": int(parsed["score"]), "reason": parsed.get("reason", "")}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("Judge response wasn't valid JSON, defaulting to score=0: %s", text[:200])
        return {"score": 0, "reason": f"judge parse failure: {e}"}


def run_eval(repo: str | None = None) -> dict:
    # Same repo-scoping fix as agent.py: without this, eval's retrieval
    # would search across every ingested repo's issues mixed together,
    # scoring relevance against the wrong data.
    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")

    api_key = _get_api_key()
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=api_key)

    results = []
    last_call_at = 0.0
    current_interval = _MIN_INTERVAL_SECONDS

    for question in TEST_QUESTIONS:
        elapsed = time.monotonic() - last_call_at
        if elapsed < current_interval:
            wait = current_interval - elapsed
            log.info("Pacing: waiting %.1fs before next judge call (interval=%.1fs)", wait, current_interval)
            time.sleep(wait)

        retrieved = retrieve_issues(question, n_results=5, repo=repo)

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
        "repo": repo,
        "avg_score": round(avg_score, 2),
        "num_questions": len(TEST_QUESTIONS),
        "results": results,
    }
    log.info("Eval complete: avg_score=%s across %s questions", summary["avg_score"], summary["num_questions"])
    return summary


# --- Full RAG eval: retrieval + faithfulness + answer relevance -----------
# Separate from run_eval() above on purpose: that one only exercises
# retrieval (cheap, no agent/generation call), so it can afford all 10
# questions on the daily schedule. This one runs the FULL agent (which
# itself costs 1-2 model calls to decide whether to search and to compose
# an answer) plus one combined judge call -- roughly 3x the quota cost per
# question. Kept to a smaller question set and manually triggered (not on
# the automated daily DAG) so a single run doesn't exhaust the whole day's
# quota by itself.
FULL_EVAL_QUESTIONS = TEST_QUESTIONS[:5]

COMBINED_JUDGE_PROMPT = """You are evaluating a RAG (retrieval-augmented generation) \
system for a GitHub issue tracker.

Question: {question}

Retrieved context (issues the system found):
{results}

Generated answer:
{answer}

Score three things, each on a 1-5 scale:
1. retrieval_relevance: are the retrieved issues actually relevant to the question? \
(a score of 3 for genuinely empty results is acceptable if the question is plausible \
but the corpus may not cover it -- don't penalize honest "no results" as harshly as \
actively wrong results)
2. faithfulness: does the answer ONLY use information present in the retrieved \
context above, without inventing facts, issue numbers, or details not actually there?
3. answer_relevance: does the answer actually address what was asked?

Respond with ONLY a JSON object:
{{"retrieval_relevance": <int 1-5>, "faithfulness": <int 1-5>, "answer_relevance": <int 1-5>, "reason": "<one sentence covering all three>"}}"""


def _judge_full(llm, question: str, sources: list[dict], answer: str) -> dict:
    if not sources:
        results_text = "No relevant issues found."
    else:
        results_text = "\n".join(f"- Issue #{s['number']}: {s['snippet'][:150]}" for s in sources)

    prompt = COMBINED_JUDGE_PROMPT.format(question=question, results=results_text, answer=answer)
    response = llm.invoke(prompt)

    text = extract_text(response.content)
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(text)
        return {
            "retrieval_relevance": int(parsed["retrieval_relevance"]),
            "faithfulness": int(parsed["faithfulness"]),
            "answer_relevance": int(parsed["answer_relevance"]),
            "reason": parsed.get("reason", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("Full-eval judge response wasn't valid JSON, defaulting to 0s: %s", text[:200])
        return {"retrieval_relevance": 0, "faithfulness": 0, "answer_relevance": 0, "reason": f"judge parse failure: {e}"}


def run_full_eval_stream(repo: str | None = None):
    """
    Generator version of the full eval -- yields progress events as each
    question is processed, instead of running silently for ~1 minute.
    Shape of yielded dicts:
      {"type": "step", "detail": "..."}
      {"type": "final", "repo": ..., "avg_*": ..., "results": [...]}
      {"type": "error", "detail": "..."}

    Runs the actual agent (retrieval + generation, the real end-to-end
    path a user hits) for each question in FULL_EVAL_QUESTIONS, then
    scores retrieval relevance, faithfulness, and answer relevance in one
    combined judge call per question. This is the eval that actually
    checks whether the AGENT'S ANSWER is trustworthy, not just whether
    search returned relevant documents -- an answer can cite plausible-
    looking issue numbers while inventing details not in the retrieved
    context, and only a faithfulness check catches that.
    """
    from pipeline.agent import ask

    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")

    try:
        api_key = _get_api_key()
    except RuntimeError as e:
        yield {"type": "error", "detail": str(e)}
        return

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=api_key)

    results = []
    last_call_at = 0.0
    current_interval = _MIN_INTERVAL_SECONDS
    total = len(FULL_EVAL_QUESTIONS)

    for idx, question in enumerate(FULL_EVAL_QUESTIONS, start=1):
        yield {"type": "step", "detail": f'({idx}/{total}) Asking the agent: "{question}"'}

        elapsed = time.monotonic() - last_call_at
        if elapsed < current_interval:
            wait = current_interval - elapsed
            yield {"type": "step", "detail": f"Pacing {wait:.0f}s before the agent call (free-tier rate limit)"}
            time.sleep(wait)

        try:
            agent_result = ask(question, repo=repo)
        except Exception as e:
            log.warning("Agent failed on eval question %r: %s", question, e)
            yield {"type": "step", "detail": f"({idx}/{total}) Agent call failed: {e}"}
            results.append({
                "question": question, "answer": f"(agent error: {e})", "sources": [],
                "retrieval_relevance": 0, "faithfulness": 0, "answer_relevance": 0,
                "reason": "agent call failed, not scored",
            })
            last_call_at = time.monotonic()
            continue
        last_call_at = time.monotonic()

        yield {
            "type": "step",
            "detail": f"({idx}/{total}) Got an answer citing {len(agent_result['sources'])} source(s). Judging retrieval, faithfulness, and relevance...",
        }

        elapsed = time.monotonic() - last_call_at
        if elapsed < current_interval:
            wait = current_interval - elapsed
            yield {"type": "step", "detail": f"Pacing {wait:.0f}s before the judge call"}
            time.sleep(wait)

        judgment = _judge_full(llm, question, agent_result["sources"], agent_result["answer"])
        last_call_at = time.monotonic()

        results.append({
            "question": question,
            "answer": agent_result["answer"],
            "sources": agent_result["sources"],
            "retrieval_relevance": judgment["retrieval_relevance"],
            "faithfulness": judgment["faithfulness"],
            "answer_relevance": judgment["answer_relevance"],
            "reason": judgment["reason"],
        })
        yield {
            "type": "step",
            "detail": f"({idx}/{total}) Scored: retrieval={judgment['retrieval_relevance']}/5 · "
                      f"faithfulness={judgment['faithfulness']}/5 · relevance={judgment['answer_relevance']}/5",
        }

    def _avg(key):
        vals = [r[key] for r in results if key in r]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    yield {
        "type": "final",
        "repo": repo,
        "num_questions": total,
        "avg_retrieval_relevance": _avg("retrieval_relevance"),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_relevance": _avg("answer_relevance"),
        "results": results,
    }


def run_full_eval(repo: str | None = None) -> dict:
    """Blocking wrapper for CLI/non-streaming callers -- drains run_full_eval_stream()."""
    for event in run_full_eval_stream(repo):
        if event["type"] == "final":
            return {k: v for k, v in event.items() if k != "type"}
        if event["type"] == "error":
            raise RuntimeError(event["detail"])
    raise RuntimeError("Eval stream ended without a final result.")


if __name__ == "__main__":
    summary = run_eval()
    print(json.dumps(summary, indent=2))