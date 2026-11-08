"""
Airflow DAG: runs the full ingest -> embed pipeline on a schedule (or
on-demand, triggered from the web UI for an arbitrary repo).

Scheduled runs refresh EVERY tracked repo (--all-repos). Reads an
optional `repo` key from the triggering dag_run's conf -- when
the UI triggers this DAG for e.g. "facebook/react", it's passed through
Jinja templating into the task's environment (PIPELINE_REPO) and
handed to the pipeline as --repo facebook/react. Falls
back to refreshing all tracked repos for the normal scheduled runs,
which don't set any conf.

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

    # Two modes:
    #   * UI-triggered (conf has "repo"): ingest just that repo.
    #   * Scheduled (no conf): --all-repos, i.e. every repo ever ingested
    #     plus GITHUB_REPO. Previously the schedule only refreshed
    #     GITHUB_REPO, so UI-added repos went stale after their first run.
    #
    # The repo is passed through the task's ENVIRONMENT, never templated
    # into the command string. It used to be --repo {{ dag_run.conf.get('repo') }}
    # inside bash_command, so a conf value like "a/b; rm -rf /" became
    # shell syntax. Now bash only ever sees "$PIPELINE_REPO" as one quoted
    # word, whatever it contains -- and the pipeline CLI validates it
    # again against pipeline/repo_name.py before using it.
    pipeline_task = BashOperator(
        task_id="run_full_pipeline",
        bash_command=(
            'cd /opt/airflow && '
            'if [ -n "$PIPELINE_REPO" ]; then '
            '/opt/pipeline_venv/bin/python -m pipeline.mlflow_tracking --no-eval --repo "$PIPELINE_REPO"; '
            'else '
            '/opt/pipeline_venv/bin/python -m pipeline.mlflow_tracking --no-eval --all-repos; '
            'fi'
        ),
        env={"PIPELINE_REPO": "{{ (dag_run.conf or {}).get('repo', '') }}"},
        append_env=True,  # keep POSTGRES_*, CHROMA_*, GITHUB_TOKEN, ...
        # Scheduled runs now loop over every tracked repo -- give it room.
        execution_timeout=timedelta(minutes=50),
    )
