"""Golden-set eval: retrieval metrics (no LLM) and the answer eval."""

import pytest

from pipeline import eval as ev
from pipeline import golden_set


@pytest.fixture
def golden(tmp_path, monkeypatch):
    monkeypatch.setattr(golden_set, "GOLDEN_DIR", tmp_path)
    golden_set.save_golden_set("o/r", [
        {"question": "q1", "expected": [10], "reviewed": True},
        {"question": "q2", "expected": [20, 21], "reviewed": True},
        {"question": "q3", "expected": [30], "reviewed": False},
    ])
    return tmp_path


def test_retrieval_eval_scores_hit_at_5_and_mrr(golden):
    returned = {
        "q1": [10, 1, 2],                 # rank 1
        "q2": [1, 2, 3, 21],              # rank 4 (second acceptable answer)
        "q3": [1, 2, 3, 4, 5, 30],        # rank 6 -> outside top 5
    }
    calls = []

    def fake_retrieve(query, n_results, repo):
        calls.append((query, n_results, repo))
        return [{"number": n} for n in returned[query]]

    s = ev.run_retrieval_eval("o/r", retrieve=fake_retrieve)

    assert s["hit_at_5"] == round(2 / 3, 3)
    assert s["mrr"] == round((1 + 1 / 4 + 1 / 6) / 3, 3)
    assert [m["question"] for m in s["misses"]] == ["q3"]
    assert s["num_questions"] == 3 and s["num_reviewed"] == 2
    assert s["golden_set_version"]
    assert all(c[1] == ev.SEARCH_DEPTH and c[2] == "o/r" for c in calls)


def test_retrieval_eval_without_golden_set_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(golden_set, "GOLDEN_DIR", tmp_path)
    with pytest.raises(ev.NoGoldenSetError):
        ev.run_retrieval_eval("o/r", retrieve=lambda *a, **k: [])


@pytest.mark.parametrize("text,ok", [
    ('{"faithfulness": 5, "answer_relevance": 4, "reason": "fine"}', True),
    ('```json\n{"faithfulness": "3", "answer_relevance": 3}\n```', True),
    ('{"faithfulness": 9, "answer_relevance": 4}', False),   # out of range
    ('{"faithfulness": 5}', False),                         # missing key
    ("The answer is good.", False),
])
def test_parse_judge_response(text, ok):
    assert (ev.parse_judge_response(text) is not None) == ok


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, prompt):
        reply = self.replies.pop(0)

        class Msg:
            content = reply
        return Msg()


def test_full_eval_counts_failures_instead_of_scoring_zero(golden):
    def fake_ask(question, repo):
        if question == "q3":
            raise RuntimeError("timeout")
        number = 10 if question == "q1" else 99
        return {"answer": f"answer to {question}", "sources": [{"number": number, "snippet": "s"}]}

    llm = FakeLLM([
        '{"faithfulness": 5, "answer_relevance": 4, "reason": "ok"}',  # q1
        "not json at all",                                            # q2 -> judge failure
    ])
    final = ev.run_full_eval("o/r", ask_fn=fake_ask, llm=llm, min_interval=0)

    assert final["num_questions"] == 3
    assert final["num_scored"] == 1
    assert final["judge_failures"] == 1 and final["agent_errors"] == 1
    # averages only over scored questions -- not dragged down by failures
    assert final["avg_faithfulness"] == 5 and final["avg_answer_relevance"] == 4
    # found_expected over answered questions: q1 yes, q2 no
    assert final["found_expected_rate"] == 0.5
    statuses = [r["status"] for r in final["results"]]
    assert statuses == ["scored", "judge_failed", "agent_error"]


def test_full_eval_prefers_reviewed_questions(golden, monkeypatch):
    monkeypatch.setattr(ev, "FULL_EVAL_QUESTIONS", 2)
    asked = []

    def fake_ask(question, repo):
        asked.append(question)
        return {"answer": "a", "sources": []}

    llm = FakeLLM(['{"faithfulness": 5, "answer_relevance": 5}'] * 2)
    ev.run_full_eval("o/r", ask_fn=fake_ask, llm=llm, min_interval=0)
    assert asked == ["q1", "q2"]


def test_full_eval_without_golden_set_yields_error_event(tmp_path, monkeypatch):
    monkeypatch.setattr(golden_set, "GOLDEN_DIR", tmp_path)
    events = list(ev.run_full_eval_stream("o/r", ask_fn=lambda *a, **k: None, llm=object()))
    assert events[-1]["type"] == "error" and "golden_set generate" in events[-1]["detail"]
