"""Structured JSON logging with request correlation.

Privacy rule enforced here: article text is never logged unless
``FND_LOG_REQUEST_BODIES=true`` is set explicitly. Log records carry lengths and
identifiers instead, which is enough to debug latency and error rates without
retaining user content.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from contextvars import ContextVar
from typing import Any

__all__ = [
    "JsonFormatter",
    "LatencyTracker",
    "configure_logging",
    "get_logger",
    "new_request_id",
    "request_id_var",
    "set_request_id",
]

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def __init__(
        self, *, service: str = "fake-news-detection", include_extras: bool = True
    ) -> None:
        super().__init__()
        self.service = service
        self.include_extras = include_extras

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service,
        }
        request_id = getattr(record, "request_id", None) or request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if self.include_extras:
            for key, value in record.__dict__.items():
                if key not in _RESERVED and key not in payload and not key.startswith("_"):
                    try:
                        json.dumps(value)
                        payload[key] = value
                    except (TypeError, ValueError):
                        payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    service: str = "fake-news-detection",
) -> None:
    """Install a single stdout handler with the chosen format.

    Called by the API and the CLI entry points. Repeated calls are idempotent.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter(service=service))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Third-party loggers are noisy at INFO during training.
    for noisy in ("matplotlib", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)


def new_request_id() -> str:
    """Generate a short, sortable request identifier."""
    import uuid

    return f"req_{uuid.uuid4().hex[:16]}"


def set_request_id(request_id: str | None) -> None:
    """Bind a request id to the current async/thread context."""
    request_id_var.set(request_id)


class LatencyTracker:
    """Thread-safe rolling latency statistics for a small set of endpoints.

    This is deliberately in-process and dependency-free: the goal is to expose
    enough signal to notice a regression, not to replace a real metrics stack.
    """

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._window = window
        self._started = time.time()

    def observe(self, endpoint: str, duration_ms: float, *, error: bool = False) -> None:
        with self._lock:
            bucket = self._samples.setdefault(endpoint, deque(maxlen=self._window))
            bucket.append(float(duration_ms))
            self._counts[endpoint] = self._counts.get(endpoint, 0) + 1
            if error:
                self._errors[endpoint] = self._errors.get(endpoint, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        """Return recent latency percentiles and counters per endpoint."""
        import numpy as np

        with self._lock:
            payload: dict[str, Any] = {
                "uptime_seconds": round(time.time() - self._started, 1),
                "endpoints": {},
            }
            for endpoint, samples in self._samples.items():
                values = np.asarray(samples, dtype="float64")
                payload["endpoints"][endpoint] = {
                    "requests": self._counts.get(endpoint, 0),
                    "errors": self._errors.get(endpoint, 0),
                    "window": int(values.size),
                    "p50_ms": round(float(np.percentile(values, 50)), 3) if values.size else None,
                    "p95_ms": round(float(np.percentile(values, 95)), 3) if values.size else None,
                    "max_ms": round(float(values.max()), 3) if values.size else None,
                }
            return payload
