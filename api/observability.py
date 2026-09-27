"""Structured request logging: one JSON line per request, carrying the
request ID, route, status, total time, and whatever fields the handler
attached along the way (per-stage timings, confidence, fallback) via
log_fields().
"""

import json
import logging
import sys
import time
from contextvars import ContextVar

logger = logging.getLogger("rag.api")

# The dict for the in-flight request. Handlers mutate it via log_fields();
# the middleware owns creating it and logging it once the response is out.
_request_fields: ContextVar[dict | None] = ContextVar("request_fields", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "msg": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Send rag.api logs to stderr as JSON. Idempotent, so reloading the app
    (or building it twice in tests) never stacks duplicate handlers."""
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def start_request() -> tuple[dict, object]:
    fields: dict = {}
    return fields, _request_fields.set(fields)


def end_request(token) -> None:
    _request_fields.reset(token)


def log_fields(**fields) -> None:
    """Attach fields to the current request's log line (no-op outside a request)."""
    current = _request_fields.get()
    if current is not None:
        current.update(fields)
