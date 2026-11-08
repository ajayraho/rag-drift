"""
Evaluation, in two parts -- both run against the repo's golden question
set (see golden_set.py), so scores are comparable run over run:

1. Retrieval eval (run_retrieval_eval) -- cheap, objective, NO LLM.
   For each golden question, search the index and check where the
   expected issue landed: Hit@5 and MRR (see metrics.py). Runs daily
   and after any retrieval change. This replaced an LLM judge rating
   "how relevant do these results look" on 10 hard-coded LangChain
   questions, which was subjective, cost quota, and meaningless for
   any other repo.

2. Answer eval (run_full_eval_stream) -- runs the real agent on a few
   golden questions and has an LLM judge score the ANSWER on:
     faithfulness     -- uses only what's in the sources it cited
     answer_relevance -- actually addresses the question
   Plus "found_expected": did the agent's own sources include the
   expected issue (objective, no judge). Manually triggered from the UI
   because each question costs ~3 LLM calls on a free-tier quota.

Judge failures (unparseable JSON) and agent errors are COUNTED and
excluded from averages. They used to be scored 0, which silently
dragged averages down and looked like a quality regression.

Runnable standalone: `python -m pipeline.eval --repo owner/name`
"""

import json
import os
import sys
import time

from pipeline.golden_set import golden_set_version, load_golden_set
from pipeline.log_utils import extract_text, get_run_logger
from pipeline.metrics import first_hit_rank, hit_at_k, mean, reciprocal_rank

log = get_run_logger(__name__)

# Free-tier Gemini quota is tight -- pace LLM calls. Configurable since a
# paid tier would allow a much higher rate.
GEMINI_RPM_LIMIT = int(os.environ.get("GEMINI_RPM_LIMIT", "5"))
_MIN_INTERVAL_SECONDS = 60.0 / GEMINI_RPM_LIMIT
_MAX_INTERVAL_SECONDS = 120.0
_MAX_RETRIES = 4

HIT_K = 5          # "did the right issue make the top 5?"
SEARCH_DEPTH = 10  # how many results we fetch to compute MRR
FULL_EVAL_QUESTIONS = 5


class NoGoldenSetError(RuntimeError):
    pass


def _questions_for(repo: str) -> list[dict]:
    items = load_golden_set(repo)
    if not items:
        raise NoGoldenSetError(
            f"No golden question set for {repo}. Create one with: "
            f"python -m pipeline.golden_set generate --repo {repo}"
        )
    return items


# --- 1. Retrieval eval -----------------------------------------------------

def run_retrieval_eval(repo: str, retrieve=None) -> dict:
    """
    retrieve(query, n_results, repo) -> list of issue dicts with "number";
    injectable for tests, defaults to the agent's real retrieve_issues.
    """
    if retrieve is None:
        from pipeline.agent import retrieve_issues as retrieve

    items = _questions_for(repo)
    per_question = []
    for item in items:
        ranked = [r["number"] for r in retrieve(item["question"], n_results=SEARCH_DEPTH, repo=repo)]
        per_question.append({
            "question": item["question"],
            "expected": item["expected"],
            "rank": first_hit_rank(ranked, item["expected"]),
            "hit": hit_at_k(ranked, item["expected"], HIT_K),
            "rr": reciprocal_rank(ranked, item["expected"]),
            "returned": ranked,
        })

    summary = {
        "repo": repo,
        "golden_set_version": golden_set_version(repo),
        "num_questions": len(per_question),
        "num_reviewed": sum(1 for i in items if i.get("reviewed")),
        f"hit_at_{HIT_K}": round(mean([q["hit"] for q in per_question]), 3),
        "mrr": round(mean([q["rr"] for q in per_question]), 3),
        "misses": [q for q in per_question if not q["hit"]],
        "results": per_question,
    }
    log.info("Retrieval eval for %s: hit@%s=%s mrr=%s over %s questions",
             repo, HIT_K, summary[f"hit_at_{HIT_K}"], summary["mrr"], summary["num_questions"])
    return summary


# --- 2. Answer eval ----------------------------------------------------------

JUDGE_PROMPT = """You are evaluating an assistant that answers questions about a \
project's GitHub issues.

Question: {question}

Sources the assistant retrieved:
{sources}

Assistant's answer:
{answer}

Score each on a 1-5 scale:
1. faithfulness: does the answer ONLY use information present in the sources \
above, without inventing facts, issue numbers, or details? (If there are no \
sources and the answer honestly says nothing relevant was found, that is a 5.)
2. answer_relevance: does the answer actually address what was asked?

Respond with ONLY a JSON object:
{{"faithfulness": <int 1-5>, "answer_relevance": <int 1-5>, "reason": "<one sentence>"}}"""


def _is_rate_limit_error(e: Exception) -> bool:
    text = str(e)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def parse_judge_response(text: str) -> dict | None:
    """Scores dict, or None if the judge's output can't be used."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        scores = {"faithfulness": int(parsed["faithfulness"]),
                  "answer_relevance": int(parsed["answer_relevance"])}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    if not all(1 <= v <= 5 for v in scores.values()):
        return None
    scores["reason"] = str(parsed.get("reason", ""))
    return scores


def _judge(llm, question: str, sources: list[dict], answer: str) -> dict | None:
    sources_text = (
        "\n".join(f"- Issue #{s['number']}: {s['snippet'][:300]}" for s in sources)
        if sources else "(none)"
    )
    response = llm.invoke(JUDGE_PROMPT.format(question=question, sources=sources_text, answer=answer))
    result = parse_judge_response(extract_text(response.content))
    if result is None:
        log.warning("Judge output unusable for %r", question)
    return result


def _default_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    from pipeline.agent import GEMINI_MODEL, _get_api_key

    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=_get_api_key())


def run_full_eval_stream(repo: str | None = None, ask_fn=None, llm=None, min_interval: float | None = None):
    """
    Yields {"type": "step"|"final"|"error", ...} events (the UI renders
    them live). ask_fn/llm/min_interval are injectable for tests.
    """
    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")
    if ask_fn is None:
        from pipeline.agent import ask as ask_fn
    interval = _MIN_INTERVAL_SECONDS if min_interval is None else min_interval

    try:
        items = _questions_for(repo)
        llm = llm or _default_llm()
    except (NoGoldenSetError, RuntimeError) as e:
        yield {"type": "error", "detail": str(e)}
        return

    # Prefer reviewed questions; stable order so runs are comparable.
    items = sorted(items, key=lambda i: not i.get("reviewed"))[:FULL_EVAL_QUESTIONS]
    total = len(items)
    results = []
    last_call_at = 0.0

    def pace():
        nonlocal last_call_at
        wait = interval - (time.monotonic() - last_call_at)
        if wait > 0:
            time.sleep(wait)

    for idx, item in enumerate(items, start=1):
        question = item["question"]
        yield {"type": "step", "detail": f'({idx}/{total}) Asking the agent: "{question}"'}
        pace()
        try:
            agent_result = ask_fn(question, repo=repo)
        except Exception as e:
            last_call_at = time.monotonic()
            yield {"type": "step", "detail": f"({idx}/{total}) Agent call failed: {e}"}
            results.append({"question": question, "status": "agent_error", "answer": f"(agent error: {e})",
                            "sources": [], "found_expected": False, "reason": "agent call failed, not scored"})
            continue
        last_call_at = time.monotonic()

        found = first_hit_rank([s["number"] for s in agent_result["sources"]], item["expected"]) is not None
        yield {"type": "step", "detail": f"({idx}/{total}) Got an answer citing {len(agent_result['sources'])} "
                                         f"source(s){' incl. the expected issue' if found else ''}. Judging..."}

        judgment = None
        for attempt in range(1, _MAX_RETRIES + 1):
            pace()
            try:
                judgment = _judge(llm, question, agent_result["sources"], agent_result["answer"])
                last_call_at = time.monotonic()
                break
            except Exception as e:
                last_call_at = time.monotonic()
                if not _is_rate_limit_error(e) or attempt == _MAX_RETRIES:
                    log.warning("Judge call failed for %r: %s", question, e)
                    break
                interval = min(interval * 2, _MAX_INTERVAL_SECONDS)
                yield {"type": "step", "detail": f"Rate limited; slowing down to one call per {interval:.0f}s"}

        row = {"question": question, "answer": agent_result["answer"], "sources": agent_result["sources"],
               "expected": item["expected"], "found_expected": found}
        if judgment is None:
            row.update(status="judge_failed", reason="judge output unusable, not scored")
            yield {"type": "step", "detail": f"({idx}/{total}) Judge output unusable -- counted, not scored"}
        else:
            row.update(status="scored", **judgment)
            yield {"type": "step", "detail": f"({idx}/{total}) faithfulness={judgment['faithfulness']}/5 · "
                                             f"relevance={judgment['answer_relevance']}/5"}
        results.append(row)

    scored = [r for r in results if r["status"] == "scored"]
    answered = [r for r in results if r["status"] != "agent_error"]
    yield {
        "type": "final",
        "repo": repo,
        "golden_set_version": golden_set_version(repo),
        "num_questions": total,
        "num_scored": len(scored),
        "judge_failures": sum(1 for r in results if r["status"] == "judge_failed"),
        "agent_errors": sum(1 for r in results if r["status"] == "agent_error"),
        "found_expected_rate": round(mean([float(r["found_expected"]) for r in answered]), 3),
        "avg_faithfulness": round(mean([r["faithfulness"] for r in scored]), 2),
        "avg_answer_relevance": round(mean([r["answer_relevance"] for r in scored]), 2),
        "results": results,
    }


def run_full_eval(repo: str | None = None, **kwargs) -> dict:
    """Blocking wrapper -- drains run_full_eval_stream()."""
    for event in run_full_eval_stream(repo, **kwargs):
        if event["type"] == "final":
            return {k: v for k, v in event.items() if k != "type"}
        if event["type"] == "error":
            raise RuntimeError(event["detail"])
    raise RuntimeError("Eval stream ended without a final result.")


if __name__ == "__main__":
    target = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else os.environ.get(
        "GITHUB_REPO", "langchain-ai/langchain")
    summary = run_retrieval_eval(target)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
