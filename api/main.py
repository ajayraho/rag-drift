"""
FastAPI wrapper around the RAG agent. Exposes /ask (blocking), /ask/stream
(SSE, powers the live UI trace), /health, /demo/drift-simulation,
/trigger-ingestion (+ status polling, calls Airflow's REST API to run
ingestion against an arbitrary repo), and serves the static chat UI at /.

Run locally: uvicorn api.main:app --reload --port 8000
"""

import json
import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.agent import ask, ask_stream, get_agent
from pipeline.log_utils import get_run_logger
from pipeline.self_heal import run_synthetic_demo

log = get_run_logger(__name__)

AIRFLOW_BASE_URL = os.environ.get("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
AIRFLOW_API_USERNAME = os.environ.get("AIRFLOW_API_USERNAME", "api_trigger")
AIRFLOW_API_PASSWORD = os.environ.get("AIRFLOW_API_PASSWORD", "")
INGESTION_DAG_ID = "ingest_github_issues"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_agent()
        log.info("Agent pre-warmed successfully at startup.")
    except RuntimeError as e:
        log.error("Startup failed: %s", e)
        raise
    yield


app = FastAPI(title="RAG Drift Pipeline API", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    repo: str | None = None


class SourceIssue(BaseModel):
    number: int
    state: str
    labels: str
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceIssue]


class TriggerIngestionRequest(BaseModel):
    repo: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        result = ask(request.question, repo=request.repo)
    except Exception as e:
        log.exception("Agent failed to answer question")
        raise HTTPException(status_code=502, detail=f"Agent error: {e}")
    return result


@app.post("/ask/stream")
def ask_stream_endpoint(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    def event_generator():
        for event in ask_stream(request.question, repo=request.repo):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/demo/drift-simulation")
def drift_simulation_endpoint():
    try:
        result = run_synthetic_demo()
    except Exception as e:
        log.exception("Drift simulation failed")
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")
    return result


def _valid_repo_format(repo: str) -> bool:
    parts = repo.strip().split("/")
    return len(parts) == 2 and all(p.strip() for p in parts)


@app.post("/trigger-ingestion")
def trigger_ingestion_endpoint(request: TriggerIngestionRequest):
    """
    Triggers the ingest_github_issues Airflow DAG for an arbitrary repo,
    via Airflow's REST API -- lets the UI point the whole pipeline
    (ingest -> validate -> drift-check -> embed) at any public GitHub
    repo on demand, not just the one hardcoded in docker-compose.
    """
    repo = request.repo.strip()
    if not _valid_repo_format(repo):
        raise HTTPException(status_code=400, detail="repo must be in 'owner/name' format, e.g. facebook/react")

    try:
        resp = requests.post(
            f"{AIRFLOW_BASE_URL}/api/v1/dags/{INGESTION_DAG_ID}/dagRuns",
            auth=(AIRFLOW_API_USERNAME, AIRFLOW_API_PASSWORD),
            json={"conf": {"repo": repo}},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        log.exception("Failed to trigger Airflow DAG")
        raise HTTPException(status_code=502, detail=f"Could not reach Airflow: {e}")

    data = resp.json()
    return {"dag_run_id": data["dag_run_id"], "state": data.get("state", "queued"), "repo": repo}


@app.get("/trigger-ingestion/status/{dag_run_id}")
def trigger_ingestion_status_endpoint(dag_run_id: str):
    """Poll target for the UI -- dag_run_id must be URL-encoded by the caller (it contains ':' and '+')."""
    try:
        resp = requests.get(
            f"{AIRFLOW_BASE_URL}/api/v1/dags/{INGESTION_DAG_ID}/dagRuns/{dag_run_id}",
            auth=(AIRFLOW_API_USERNAME, AIRFLOW_API_PASSWORD),
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        log.exception("Failed to poll Airflow DAG run status")
        raise HTTPException(status_code=502, detail=f"Could not reach Airflow: {e}")

    data = resp.json()
    return {"dag_run_id": dag_run_id, "state": data.get("state", "unknown")}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
