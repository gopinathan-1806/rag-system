"""Metrics and structured logging. Prometheus text format at /metrics, JSON logs on stdout."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("cc_http_requests_total", "HTTP requests", ["method", "route", "status"])
REQUEST_LATENCY = Histogram(
    "cc_http_request_seconds", "HTTP request latency", ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)  # fmt: skip
RETRIEVAL_LATENCY = Histogram("cc_retrieval_seconds", "Embed query + hybrid search")
LLM_FIRST_TOKEN = Histogram(
    "cc_llm_first_token_seconds", "Time to first streamed token",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 10, 20),
)  # fmt: skip
LLM_TOKENS = Counter("cc_llm_tokens_total", "LLM tokens", ["direction"])
GUARDRAIL_BLOCKS = Counter("cc_guardrail_blocks_total", "Requests blocked", ["rule"])
CITATION_CHECKS = Counter("cc_citation_checks_total", "Answer citation outcomes", ["outcome"])
INGESTED = Counter("cc_documents_ingested_total", "Documents ingested", ["kind"])
LLM_ERRORS = Counter("cc_llm_errors_total", "LLM failures")
ACTIVE_SESSIONS = Gauge("cc_active_sessions", "Sessions held in memory")


class JsonFormatter(logging.Formatter):
    _SKIP = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in self._SKIP})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our request middleware
