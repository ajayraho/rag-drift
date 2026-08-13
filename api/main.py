"""
FastAPI wrapper around the RAG agent. Exposes /ask (blocking), /ask/stream
(SSE, powers the live UI trace), /health, /repos (for the UI's repo
autocomplete), /demo/drift-simulation, /trigger-ingestion (+ status
polling, calls Airflow's REST API to run ingestion against an arbitrary
repo), and serves the static chat UI at /.

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
from pipeline.embed import get_chroma_collection
from pipeline.log_utils import get_run_logger
from pipeline.eval import run_full_eval, run_full_eval_stream
from pipeline.self_heal import run_synthetic_demo

log = get_run_logger(__name__)

AIRFLOW_BASE_URL = os.environ.get("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
AIRFLOW_API_USERNAME = os.environ.get("AIRFLOW_API_USERNAME", "api_trigger")
AIRFLOW_API_PASSWORD = os.environ.get("AIRFLOW_API_PASSWORD", "")
INGESTION_DAG_ID = "ingest_github_issues"
DEFAULT_REPO = os.environ.get("GITHUB_REPO", "langchain-ai/langchain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm both the default single-repo agent (the common case, since
    # the UI's repo field starts pre-populated) and the search-all-repos
    # agent (repo=None) -- cheap, since this just builds the LLM client +
    # graph, no API call happens yet. Explicit repo here matters: get_agent()
    # with no args now means "search all repos", not "use the default".
    try:
        get_agent(DEFAULT_REPO)
        get_agent(None)
        log.info("Agent pre-warmed successfully at startup (default repo + search-all).")
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
    repo: str
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


@app.get("/repos")
def list_repos_endpoint():
    """
    Returns every repo with embedded, searchable data -- powers the UI's
    repo-field autocomplete. Reads Chroma directly rather than Postgres:
    keeps the API decoupled from the ingestion pipeline's database, and
    "has embedded data" is the more relevant signal here than "was ever
    ingested" (a repo could be ingested but not yet embedded).
    """
    try:
        collection = get_chroma_collection()
        data = collection.get(include=["metadatas"])
        repos = sorted({m["repo"] for m in data["metadatas"] if m.get("repo")})
    except Exception as e:
        log.exception("Failed to list repos")
        raise HTTPException(status_code=500, detail=f"Could not list repos: {e}")
    return {"repos": repos}


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


class EvalRequest(BaseModel):
    repo: str | None = None


@app.post("/eval/run")
def eval_run_endpoint(request: EvalRequest):
    """Blocking variant -- simple to curl/test, no live progress."""
    try:
        result = run_full_eval(repo=request.repo)
    except Exception as e:
        log.exception("Full eval failed")
        raise HTTPException(status_code=500, detail=f"Eval error: {e}")
    return result


@app.post("/eval/run/stream")
def eval_run_stream_endpoint(request: EvalRequest):
    """
    SSE variant -- streams {"type": "step"|"final"|"error", ...} events as
    each question is asked and judged, so the UI can show live progress
    instead of a blind ~1 minute wait.
    """

    def event_generator():
        for event in run_full_eval_stream(repo=request.repo):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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