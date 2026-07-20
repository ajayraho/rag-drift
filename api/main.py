"""
FastAPI wrapper around the RAG agent. Exposes a single /ask endpoint.

Run locally: uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.agent import ask, get_agent
from pipeline.log_utils import get_run_logger

log = get_run_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build (and cache, via agent.py's lru_cache) the agent once at startup
    # instead of on the first incoming request -- so the first real user
    # doesn't pay the cold-start cost of spinning up the Gemini client and
    # loading the embedding model. Also fails loudly here if GEMINI_API_KEY
    # is missing, rather than accepting traffic and failing every request.
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