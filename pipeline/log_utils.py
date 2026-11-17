"""
Shared logging setup. Every pipeline stage (ingestion, embedding, and
later the drift detector) tags its log lines with a run_id, so logs
from any stage can be correlated to a specific run -- and later, to a
specific MLflow run.
"""

import logging

_configured = False


class _RunIdFilter(logging.Filter):
    """
    Our format string requires %(run_id)s on every record. Records that
    come from our own get_run_logger() have it (via LoggerAdapter), but
    records from third-party libraries (httpx, chromadb, sentence-
    transformers' internal HF Hub calls, etc.) don't -- without this
    filter, the formatter raises on any library log line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return True


def _configure_root_logging():
    global _configured
    if _configured:
        return

    root = logging.getLogger()

    if root.handlers:
        # Something else (Airflow, uvicorn, pytest, ...) already configured
        # root logging before our code ran. Adding our OWN StreamHandler on
        # top of that is dangerous in Airflow specifically: Airflow
        # redirects sys.stdout/stderr through its own logger during task
        # execution, so a new handler writing to that redirected stream
        # re-enters the logging system, which triggers our handler again,
        # forever -- this caused a real RecursionError in production here.
        # Instead, just attach our run_id-safety filter to the existing
        # handlers and leave their formatting alone.
        for handler in root.handlers:
            handler.addFilter(_RunIdFilter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s run_id=%(run_id)s] %(message)s")
        )
        handler.addFilter(_RunIdFilter())
        root.addHandler(handler)

    root.setLevel(logging.INFO)

    # These libraries are chatty at INFO level (every HTTP call, every
    # tokenizer load) and it's not signal we care about in our pipeline logs.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


_configure_root_logging()


def get_run_logger(name: str, run_id: int | str = "-") -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger(name), {"run_id": run_id})


def extract_text(content) -> str:
    """
    As of recent LangChain/Gemini versions, message.content isn't always a
    plain string -- it can be a list of typed content blocks (text,
    reasoning, citations, etc.), matching how modern models actually
    structure responses. This normalizes either shape back to plain text.
    Shared here since both the agent (displaying answers) and the eval
    module (parsing judge output as JSON) need the same normalization.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)