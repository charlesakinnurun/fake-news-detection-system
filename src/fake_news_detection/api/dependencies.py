"""Shared FastAPI dependencies and middleware.

The app state holds a single :class:`~fake_news_detection.api.service.ModelService`
plus a :class:`~fake_news_detection.monitoring.logging.LatencyTracker`. Both are
created during the lifespan start-up hook and accessed through the dependency
functions below, which keeps request handlers free of global mutable state.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from fake_news_detection.api.service import ModelService
from fake_news_detection.config import ServiceSettings, get_service_settings
from fake_news_detection.monitoring.logging import LatencyTracker, new_request_id, set_request_id

LOGGER = logging.getLogger(__name__)

# Starlette renamed these constants across versions; prefer the new names.
HTTP_413_CONTENT_TOO_LARGE = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)

__all__ = [
    "ClientRateLimiter",
    "RequestContextMiddleware",
    "RequestSizeLimitMiddleware",
    "get_latency_tracker",
    "get_model_service",
    "get_settings",
]


def get_settings() -> ServiceSettings:
    """FastAPI dependency returning the service settings."""
    settings = get_service_settings()
    if not isinstance(settings, ServiceSettings):  # pragma: no cover - defensive
        raise TypeError("service settings are misconfigured")
    return settings


def get_model_service(request: Request) -> ModelService:
    """FastAPI dependency returning the process-wide model service."""
    service = getattr(request.app.state, "model_service", None)
    if service is None:  # pragma: no cover - only before lifespan start-up
        raise RuntimeError("model service is not initialised")
    return service


def get_latency_tracker(request: Request) -> LatencyTracker:
    """FastAPI dependency returning the in-process latency tracker."""
    tracker = getattr(request.app.state, "latency_tracker", None)
    if tracker is None:  # pragma: no cover - only before lifespan start-up
        raise RuntimeError("latency tracker is not initialised")
    return tracker


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a request id and record per-endpoint latency."""

    def __init__(
        self, app: Any, *, tracker: LatencyTracker, path_filter: tuple[str, ...] = ()
    ) -> None:
        super().__init__(app)
        self.tracker = tracker
        self.path_filter = path_filter

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or new_request_id()
        set_request_id(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error = False
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = status_code >= 500
        except Exception:
            error = True
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            path = request.url.path
            if path not in self.path_filter:
                self.tracker.observe(path, duration_ms, error=error)
            LOGGER.info(
                "request",
                extra={
                    "http_method": request.method,
                    "http_path": path,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 3),
                    "client": request.client.host if request.client else None,
                },
            )

        response.headers["X-Request-ID"] = request_id
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared body exceeds the configured limit."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "detail": (
                                f"request body exceeds FND_MAX_REQUEST_BYTES={self.max_bytes} bytes"
                            )
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "invalid Content-Length header"},
                )
        return await call_next(request)


class ClientRateLimiter(BaseHTTPMiddleware):
    """Simple in-process token bucket per client.

    Suitable for a single instance or as a second line of defence behind an
    ingress. For a multi-replica deployment, enforce the limit at the gateway or
    with a shared Redis counter instead - this limiter is per process by design
    and its state is not shared.
    """

    def __init__(
        self, app: Any, *, requests: int, window_seconds: int, enabled: bool = True
    ) -> None:
        super().__init__(app)
        self.requests = max(int(requests), 1)
        self.window = max(int(window_seconds), 1)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def _allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.requests:
                retry_after = max(self.window - (now - bucket[0]), 0.0)
                return False, retry_after
            bucket.append(now)
            if len(self._hits) > 10_000:  # bound memory for many distinct clients
                self._hits = {k: v for k, v in self._hits.items() if v}
            return True, 0.0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled or request.url.path in {"/health", "/ready", "/metrics"}:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        allowed, retry_after = self._allow(client)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return await call_next(request)
