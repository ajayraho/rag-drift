"""Eval results land in MLflow with the context needed to compare runs."""

import pytest

mlflow = pytest.importorskip("mlflow")

from pipeline import mlflow_tracking as mt  # noqa: E402


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(mt, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # artifacts go to ./mlruns under tmp
    mlflow.set_tracking_uri(uri)
    return uri


def _runs():
    return mlflow.search_runs(experiment_names=[mt.MLFLOW_EXPERIMENT], output_format="list")


def test_retrieval_eval_logs_scores_and_dataset_version(tracking, monkeypatch):
    summary = {"repo": "o/r", "golden_set_version": "abc123def456", "num_questions": 3,
               "num_reviewed": 2, "hit_at_5": 0.667, "mrr": 0.472, "misses": [], "results": []}
    monkeypatch.setattr(mt, "run_retrieval_eval", lambda repo: summary)

    mt.run_eval_only("o/r")

    (run,) = _runs()
    assert run.data.metrics["hit_at_5"] == 0.667 and run.data.metrics["mrr"] == 0.472
    assert run.data.params["golden_set_version"] == "abc123def456"
    assert run.data.params["embedding_model"] == mt.EMBEDDING_MODEL
    assert run.data.tags["run_type"] == "retrieval_eval"


def test_eval_all_repos_skips_repos_without_a_golden_set(tracking, monkeypatch):
    def fake_eval(repo):
        if repo == "no/golden":
            raise mt.NoGoldenSetError("none")
        return {"repo": repo, "golden_set_version": "v", "num_questions": 1, "num_reviewed": 1,
                "hit_at_5": 1.0, "mrr": 1.0, "misses": [], "results": []}

    monkeypatch.setattr(mt, "run_retrieval_eval", fake_eval)
    result = mt.run_eval_all_repos(["has/golden", "no/golden"])

    assert result["failed"] == {}
    assert [r.data.params["repo"] for r in _runs()] == ["has/golden"]


def test_answer_eval_is_logged_and_never_raises(tracking, monkeypatch):
    summary = {"repo": "o/r", "golden_set_version": "v1", "avg_faithfulness": 4.5,
               "avg_answer_relevance": 4.0, "found_expected_rate": 0.8, "num_scored": 4,
               "judge_failures": 1, "agent_errors": 0}
    assert mt.log_full_eval(summary)
    (run,) = _runs()
    assert run.data.metrics["avg_faithfulness"] == 4.5 and run.data.metrics["judge_failures"] == 1

    monkeypatch.setattr(mt.mlflow, "start_run", lambda: (_ for _ in ()).throw(RuntimeError("db locked")))
    assert mt.log_full_eval(summary) is None  # swallowed, logged
