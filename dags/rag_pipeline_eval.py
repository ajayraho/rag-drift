"""
Airflow DAG: daily retrieval eval for every tracked repo that has a
golden question set (pipeline/golden/<owner>__<name>.jsonl).

Scores Hit@5 and MRR -- did search return the issue each question is
about, and how high. No LLM calls at all (it used to burn ~10 judge
calls a day on hard-coded LangChain questions), so it's free to run as
often as we like. Repos without a golden set are skipped, not failed.

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
    description="Daily golden-set retrieval eval (Hit@5, MRR) per repo, tracked in MLflow",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag-pipeline", "eval"],
) as dag:

    eval_task = BashOperator(
        task_id="run_retrieval_eval",
        bash_command="cd /opt/airflow && /opt/pipeline_venv/bin/python -m pipeline.mlflow_tracking --eval-only --all-repos",
    )