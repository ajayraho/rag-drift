"""
Orchestrates a full pipeline run (ingest -> embed -> eval) as a single
MLflow run, logging params, metrics, and artifacts so every run is
auditable and comparable over time.

Runnable standalone:
  python -m pipeline.mlflow_tracking                    # full pipeline, default repo
  python -m pipeline.mlflow_tracking --repo owner/name   # full pipeline, specific repo
  python -m pipeline.mlflow_tracking --eval-only         # eval only, default repo
"""

import json
import os
import tempfile

import mlflow

from pipeline.agent import GEMINI_MODEL
from pipeline.embed import EMBEDDING_MODEL, run_embedding
from pipeline.eval import run_eval
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
            eval_summary = run_eval()
            mlflow.log_metric("retrieval_eval_avg_score", eval_summary["avg_score"])
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(eval_summary, f, indent=2)
                eval_path = f.name
            mlflow.log_artifact(eval_path, artifact_path="eval")
            os.unlink(eval_path)
            result["eval"] = eval_summary

        log.info("MLflow run %s complete: %s", run.info.run_id,
                  {k: v for k, v in result.items() if k not in ("ingestion", "embedding", "eval")})
        return result


def run_eval_only() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.set_tag("run_type", "eval_only")
        mlflow.log_param("llm_model", GEMINI_MODEL)

        eval_summary = run_eval()
        mlflow.log_metric("retrieval_eval_avg_score", eval_summary["avg_score"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(eval_summary, f, indent=2)
            eval_path = f.name
        mlflow.log_artifact(eval_path, artifact_path="eval")
        os.unlink(eval_path)

        log.info("Eval-only MLflow run %s complete: avg_score=%s", run.info.run_id, eval_summary["avg_score"])
        return {"mlflow_run_id": run.info.run_id, "eval": eval_summary}


def _parse_cli_args(argv: list[str]) -> dict:
    """Tiny manual parser -- avoids pulling in argparse for three flags."""
    parsed = {"repo": None, "eval_only": False, "no_eval": False}
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
        result = run_eval_only()
        print(json.dumps({"mlflow_run_id": result["mlflow_run_id"]}, indent=2))
        print(f"\nRetrieval eval avg score: {result['eval']['avg_score']} / 5")
    else:
        result = run_full_pipeline(repo=args["repo"], run_eval_step=not args["no_eval"])
        printable = {k: v for k, v in result.items() if k != "eval"}
        print(json.dumps(printable, indent=2, default=str))
        if "eval" in result:
            print(f"\nRetrieval eval avg score: {result['eval']['avg_score']} / 5")