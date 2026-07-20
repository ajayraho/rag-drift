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

        # Stage 1: ingest
        ingestion_result = run_ingestion(repo=repo)
        mlflow.log_metrics({
            "fetched_count": ingestion_result["fetched"],
            "valid_count": ingestion_result["valid"],
            "quarantined_count": ingestion_result["quarantined"],
            "skipped_count": ingestion_result["skipped"],
        })

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


if __name__ == "__main__":
    result = run_full_pipeline()
    printable = {k: v for k, v in result.items() if k != "eval"}
    print(json.dumps(printable, indent=2, default=str))
    if "eval" in result:
        print(f"\nRetrieval eval avg score: {result['eval']['avg_score']} / 5")