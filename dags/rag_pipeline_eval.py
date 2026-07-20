"""
Airflow DAG: runs the retrieval-quality eval on a daily schedule,
separate from ingestion (which runs hourly) -- eval costs ~10 judge LLM
calls against a tight free-tier quota, so running it every hour would
compete with real agent traffic for the same quota.

Uses BashOperator to shell out to the isolated pipeline virtualenv --
see ingest_github_issues.py for why.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "rag-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="rag_pipeline_eval",
    description="Daily retrieval-quality eval (LLM-as-judge), tracked in MLflow",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag-pipeline", "eval"],
) as dag:

    eval_task = BashOperator(
        task_id="run_retrieval_eval",
        bash_command="cd /opt/airflow && /opt/pipeline_venv/bin/python -m pipeline.mlflow_tracking --eval-only",
    )