"""
Airflow DAG: runs the full ingest -> embed pipeline on a schedule,
tracked as an MLflow run.

Uses BashOperator to shell out to a separate virtualenv
(/opt/pipeline_venv) rather than importing pipeline code directly into
Airflow's own Python process. Airflow 2.9.3 and our pipeline's LLM stack
(langchain, mlflow, chromadb) need incompatible dependency versions --
importing pipeline modules directly into Airflow's process caused a real
RecursionError in Airflow's own logging code from a transitively-
upgraded pendulum. Full isolation avoids that entirely.

Eval is intentionally NOT run here -- see rag_pipeline_eval.py, a
separate daily DAG -- since eval costs LLM calls against a tight
free-tier quota and shouldn't fire every hour.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "rag-pipeline",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ingest_github_issues",
    description="Incrementally pull, validate, embed GitHub issues; tracked in MLflow",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,  # prevents overlapping runs from racing on the same
                         # `since` high-water mark and double-fetching
    tags=["rag-pipeline", "ingestion"],
) as dag:

    pipeline_task = BashOperator(
        task_id="run_full_pipeline",
        bash_command="cd /opt/airflow && /opt/pipeline_venv/bin/python -m pipeline.mlflow_tracking",
    )