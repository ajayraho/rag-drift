"""
Orchestrates a full pipeline run (ingest -> embed -> eval) as a single
MLflow run, logging params, metrics, and artifacts so every run is
auditable and comparable over time. This is the module Airflow's
scheduled DAG calls -- ingestion and embedding on every run, eval only
when explicitly requested (it costs LLM calls against a tight free-tier
quota, so it's not something we want firing every hour).

Runnable standalone: `python -m pipeline.mlflow_tracking`
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

        # Params: things that describe the CONFIGURATION of this run, not
        # its outcome. Useful for later filtering ("show me all runs that
        # used embedding model X") if we ever change models.
        mlflow.log_params({
            "repo": repo,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": GEMINI_MODEL,
        })
        mlflow.set_tag("run_type", "full_pipeline")

        # Stage 1: ingest
        ingestion_result = run_ingestion(repo=repo)
        mlflow.log_metrics({
            "fetched_count": ingestion_result["fetched"],
            "valid_count": ingestion_result["valid"],
            "quarantined_count": ingestion_result["quarantined"],
            "skipped_count": ingestion_result["skipped"],
        })

        # Schema drift: surfaced as its own metrics so a run with drift is
        # visible directly in the MLflow runs table, not just buried in logs.
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

        # Stage 2: embed (only embeds what's new/changed -- see embed.py)
        embedding_result = run_embedding()
        mlflow.log_metrics({
            "newly_embedded_count": embedding_result["embedded_count"],
            "total_vector_count": embedding_result["collection_count"],
        })

        # "What changed this run" at a glance, without opening the DB.
        mlflow.set_tags({
            "newest_synced_at": str(ingestion_result.get("newest_updated_at")),
            "had_quarantine": str(ingestion_result["quarantined"] > 0),
        })

        result = {
            "mlflow_run_id": run.info.run_id,
            "ingestion": ingestion_result,
            "embedding": embedding_result,
        }

        # Stage 3: retrieval-quality eval -- optional, since it costs a
        # judge LLM call per test question against the same tight free-tier
        # quota the agent uses. Run on a slower cadence than ingestion.
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
    """
    Runs just the retrieval-quality eval as its own MLflow run, decoupled
    from ingestion. Used by a separate, less-frequent Airflow DAG (daily,
    not hourly) since eval costs ~10 judge LLM calls against the same
    tight free-tier quota the agent uses -- running it every ingestion
    tick would be wasteful and would compete with real agent traffic.
    """
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


if __name__ == "__main__":
    import sys

    if "--eval-only" in sys.argv:
        result = run_eval_only()
        print(json.dumps({"mlflow_run_id": result["mlflow_run_id"]}, indent=2))
        print(f"\nRetrieval eval avg score: {result['eval']['avg_score']} / 5")
    else:
        result = run_full_pipeline()
        printable = {k: v for k, v in result.items() if k != "eval"}
        print(json.dumps(printable, indent=2, default=str))
        if "eval" in result:
            print(f"\nRetrieval eval avg score: {result['eval']['avg_score']} / 5")