"""
Airflow DAG: pulls GitHub issues on a schedule, validates them,
lands valid rows in Postgres, quarantines genuinely malformed ones.

This DAG is deliberately thin -- it just calls run_ingestion() from
pipeline/fetch.py. Airflow's job is scheduling/retries/observability,
not owning the business logic. Keeping the logic testable and runnable
standalone (python -m pipeline.fetch) outside Airflow is why we built
it that way from the start.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "rag-pipeline",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _run_ingestion(**context):
    # Imported inside the callable, not at module top-level: Airflow parses
    # every DAG file repeatedly to build its UI, so heavy/DB-touching imports
    # at module scope slow that down. Keeping them inside the task callable
    # is the standard Airflow pattern.
    from pipeline.fetch import run_ingestion

    result = run_ingestion()

    # Push to XCom so downstream tasks (or a future drift-check task on Day 3)
    # can read this run's stats without re-querying Postgres.
    context["ti"].xcom_push(key="ingestion_result", value=result)

    if result["quarantined"] > 0:
        print(f"WARNING: {result['quarantined']} records quarantined this run. Check the quarantine table.")

    return result


with DAG(
    dag_id="ingest_github_issues",
    description="Incrementally pull, validate, and land GitHub issues in Postgres",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag-pipeline", "ingestion"],
) as dag:

    ingest_task = PythonOperator(
        task_id="fetch_and_validate_issues",
        python_callable=_run_ingestion,
    )