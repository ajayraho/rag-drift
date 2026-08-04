"""
Airflow DAG: runs the full ingest -> embed pipeline on a schedule (or
on-demand, triggered from the web UI for an arbitrary repo).

Reads an optional `repo` key from the triggering dag_run's conf -- when
the UI triggers this DAG for e.g. "facebook/react", it's passed through
Jinja templating into the bash command as --repo facebook/react. Falls
back to the GITHUB_REPO env var (and mlflow_tracking's own default) for
the normal scheduled runs, which don't set any conf.

Uses BashOperator to shell out to a separate virtualenv
(/opt/pipeline_venv) rather than importing pipeline code directly into
Airflow's own Python process -- see module history for why.
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
    max_active_runs=1,
    tags=["rag-pipeline", "ingestion"],
) as dag:

    pipeline_task = BashOperator(
        task_id="run_full_pipeline",
        bash_command=(
            "cd /opt/airflow && /opt/pipeline_venv/bin/python -m pipeline.mlflow_tracking --no-eval "
            "{% if dag_run.conf and dag_run.conf.get('repo') %}--repo {{ dag_run.conf.get('repo') }}{% endif %}"
        ),
    )