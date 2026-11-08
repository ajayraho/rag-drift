"""
Orchestrates a full pipeline run (ingest -> embed -> eval) as a single
MLflow run, logging params, metrics, and artifacts so every run is
auditable and comparable over time.

Runnable standalone:
  python -m pipeline.mlflow_tracking                    # full pipeline, default repo
  python -m pipeline.mlflow_tracking --repo owner/name   # full pipeline, specific repo
  python -m pipeline.mlflow_tracking --eval-only --repo owner/name  # retrieval eval, one repo
  python -m pipeline.mlflow_tracking --eval-only --all-repos        # retrieval eval, every repo with a golden set
  python -m pipeline.mlflow_tracking --all-repos         # ingest+embed every tracked repo
"""

import json
import os
import tempfile

import mlflow

from pipeline.agent import GEMINI_MODEL
from pipeline.embed import EMBEDDING_MODEL, run_embedding
from pipeline.eval import HIT_K, SEARCH_DEPTH, NoGoldenSetError, run_retrieval_eval
from pipeline.fetch import run_ingestion
from pipeline.log_utils import get_run_logger

log = get_run_logger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "rag-drift-pipeline"


def run_full_pipeline(repo: str | None = None, run_eval_step: bool = True) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")

    with mlflow.start_run() as run:
        log.info("Starting MLflow run %s for repo=%s", run.info.run_id, repo)

        mlflow.log_params({
            "repo": repo,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": GEMINI_MODEL,
        })
        mlflow.set_tag("run_type", "full_pipeline")

        ingestion_result = run_ingestion(repo=repo)
        mlflow.log_metrics({
            "fetched_count": ingestion_result["fetched"],
            "valid_count": ingestion_result["valid"],
            "quarantined_count": ingestion_result["quarantined"],
            "skipped_count": ingestion_result["skipped"],
        })

        drift_alerts = ingestion_result.get("drift_alerts", [])
        critical_drift = sum(1 for a in drift_alerts if a["severity"] == "critical")
        warning_drift = sum(1 for a in drift_alerts if a["severity"] == "warning")
        mlflow.log_metrics({
            "drift_alerts_critical": critical_drift,
            "drift_alerts_warning": warning_drift,
        })
        if drift_alerts:
            mlflow.set_tag("schema_drift_detected", "true")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(drift_alerts, f, indent=2)
                drift_path = f.name
            mlflow.log_artifact(drift_path, artifact_path="drift")
            os.unlink(drift_path)

        embedding_result = run_embedding()
        mlflow.log_metrics({
            "newly_embedded_count": embedding_result["embedded_count"],
            "total_vector_count": embedding_result["collection_count"],
        })

        mlflow.set_tags({
            "newest_synced_at": str(ingestion_result.get("newest_updated_at")),
            "had_quarantine": str(ingestion_result["quarantined"] > 0),
        })

        result = {
            "mlflow_run_id": run.info.run_id,
            "ingestion": ingestion_result,
            "embedding": embedding_result,
        }

        if run_eval_step:
            try:
                eval_summary = run_retrieval_eval(repo)
            except NoGoldenSetError as e:
                log.info("Skipping eval step: %s", e)
            else:
                _log_retrieval_eval(eval_summary)
                result["eval"] = eval_summary

        log.info("MLflow run %s complete: %s", run.info.run_id,
                  {k: v for k, v in result.items() if k not in ("ingestion", "embedding", "eval")})
        return result


def resolve_tracked_repos() -> list[str]:
    """
    Repos for a scheduled refresh: every repo ever ingested, plus
    GITHUB_REPO (so a fresh deployment still has something to ingest).
    Names read back from the DB are re-validated -- this list is passed
    straight back into the ingestion code.
    """
    from pipeline.db import get_connection, get_tracked_repos
    from pipeline.repo_name import is_valid_repo

    default_repo = os.environ.get("GITHUB_REPO", "langchain-ai/langchain")
    with get_connection() as conn:
        tracked = get_tracked_repos(conn)

    repos = []
    for repo in [default_repo, *tracked]:
        if repo in repos:
            continue
        if not is_valid_repo(repo):
            log.warning("Skipping tracked repo with an invalid name: %r", repo)
            continue
        repos.append(repo)
    return repos


def run_all_repos(repos: list[str], runner=None) -> dict:
    """
    Runs the ingest -> embed pipeline once per repo, each as its own
    MLflow run (per-repo metrics stay comparable over time).

    The hourly schedule used to refresh only GITHUB_REPO -- a repo added
    from the UI was ingested once and then went stale forever.

    One repo failing (deleted repo, transient GitHub 5xx) must not starve
    the others, so failures are collected and the caller decides the exit
    code after every repo has had its turn.
    """
    runner = runner or (lambda r: run_full_pipeline(repo=r, run_eval_step=False))
    succeeded, failed = [], {}
    for repo in repos:
        try:
            runner(repo)
            succeeded.append(repo)
        except Exception as e:
            log.exception("Pipeline failed for repo=%s; continuing with the rest", repo)
            failed[repo] = f"{type(e).__name__}: {e}"
    return {"succeeded": succeeded, "failed": failed}


def _log_json_artifact(data: dict, artifact_path: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2, default=str)
        path = f.name
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    finally:
        os.unlink(path)


def _log_retrieval_eval(summary: dict) -> None:
    """
    Everything needed to compare two retrieval eval runs fairly: the
    scores, AND what they were measured with (question-set version,
    embedding model, search depth). Called inside an active MLflow run.
    """
    mlflow.log_params({
        "golden_set_version": summary["golden_set_version"],
        "embedding_model": EMBEDDING_MODEL,
        "search_depth": SEARCH_DEPTH,
    })
    mlflow.log_metrics({
        f"hit_at_{HIT_K}": summary[f"hit_at_{HIT_K}"],
        "mrr": summary["mrr"],
        "eval_num_questions": summary["num_questions"],
        "eval_num_reviewed": summary["num_reviewed"],
    })
    _log_json_artifact(summary, "eval")


def run_eval_only(repo: str | None = None) -> dict:
    """Retrieval eval for one repo as its own MLflow run (no LLM calls)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    repo = repo or os.environ.get("GITHUB_REPO", "langchain-ai/langchain")

    summary = run_retrieval_eval(repo)  # raises NoGoldenSetError before any run is created
    with mlflow.start_run() as run:
        mlflow.set_tags({"run_type": "retrieval_eval", "repo": repo})
        mlflow.log_param("repo", repo)
        _log_retrieval_eval(summary)
        log.info("Retrieval eval run %s for %s: hit@%s=%s mrr=%s", run.info.run_id, repo,
                 HIT_K, summary[f"hit_at_{HIT_K}"], summary["mrr"])
        return {"mlflow_run_id": run.info.run_id, "eval": summary}


def run_eval_all_repos(repos: list[str]) -> dict:
    """Daily DAG entrypoint: every tracked repo that HAS a golden set."""
    def runner(repo):
        try:
            run_eval_only(repo)
        except NoGoldenSetError:
            log.info("No golden set for %s yet -- skipping eval", repo)
    return run_all_repos(repos, runner=runner)


def log_full_eval(summary: dict) -> str | None:
    """
    Logs a UI-triggered answer eval (faithfulness / answer relevance) as
    its own MLflow run. Best-effort: a logging failure must never turn a
    finished, displayed eval into an error, so it's caught and logged.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run() as run:
            mlflow.set_tags({"run_type": "answer_eval", "repo": summary["repo"]})
            mlflow.log_params({
                "repo": summary["repo"],
                "golden_set_version": summary["golden_set_version"],
                "llm_model": GEMINI_MODEL,
            })
            mlflow.log_metrics({
                "avg_faithfulness": summary["avg_faithfulness"],
                "avg_answer_relevance": summary["avg_answer_relevance"],
                "found_expected_rate": summary["found_expected_rate"],
                "num_scored": summary["num_scored"],
                "judge_failures": summary["judge_failures"],
                "agent_errors": summary["agent_errors"],
            })
            # No artifact here on purpose: the experiment's artifact root
            # is a path inside the Airflow container, which this (API)
            # process can't write to. The per-question details are shown
            # in the UI; params + metrics are what we compare over time.
            return run.info.run_id
    except Exception:
        log.exception("Could not log answer eval to MLflow (eval result itself is fine)")
        return None


def _parse_cli_args(argv: list[str]) -> dict:
    """Tiny manual parser -- avoids pulling in argparse for three flags."""
    parsed = {"repo": None, "eval_only": False, "no_eval": False, "all_repos": False}
    i = 0
    while i < len(argv):
        if argv[i] == "--repo" and i + 1 < len(argv):
            parsed["repo"] = argv[i + 1]
            i += 2
        elif argv[i] == "--eval-only":
            parsed["eval_only"] = True
            i += 1
        elif argv[i] == "--no-eval":
            parsed["no_eval"] = True
            i += 1
        elif argv[i] == "--all-repos":
            parsed["all_repos"] = True
            i += 1
        else:
            i += 1
    return parsed


if __name__ == "__main__":
    import sys

    from pipeline.repo_name import validate_repo

    args = _parse_cli_args(sys.argv[1:])
    if args["repo"] is not None:
        # Last line of defence: the value may come from Airflow dag_run.conf.
        try:
            args["repo"] = validate_repo(args["repo"])
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)

    if args["eval_only"]:
        if args["all_repos"]:
            summary = run_eval_all_repos(resolve_tracked_repos())
            print(json.dumps(summary, indent=2))
            sys.exit(1 if summary["failed"] else 0)
        try:
            result = run_eval_only(args["repo"])
        except NoGoldenSetError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        ev = result["eval"]
        print(json.dumps({"mlflow_run_id": result["mlflow_run_id"], "repo": ev["repo"],
                          f"hit_at_{HIT_K}": ev[f"hit_at_{HIT_K}"], "mrr": ev["mrr"],
                          "num_questions": ev["num_questions"]}, indent=2))
    elif args["all_repos"]:
        summary = run_all_repos(resolve_tracked_repos())
        print(json.dumps(summary, indent=2))
        # Non-zero if ANY repo failed, so Airflow marks the task failed
        # (and retries) -- but only after every repo got its turn.
        sys.exit(1 if summary["failed"] else 0)
    else:
        result = run_full_pipeline(repo=args["repo"], run_eval_step=not args["no_eval"])
        printable = {k: v for k, v in result.items() if k != "eval"}
        print(json.dumps(printable, indent=2, default=str))
        if "eval" in result:
            print(f"\nRetrieval eval: hit@{HIT_K}={result['eval'][f'hit_at_{HIT_K}']} mrr={result['eval']['mrr']}")
