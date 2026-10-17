"""FastAPI application: routes, lifespan and error handling.

Endpoints
---------
``GET  /health``         liveness - process is up
``GET  /ready``          readiness - model is loaded and usable
``GET  /metadata``       safe model metadata (no paths, no secrets)
``GET  /metrics``        in-process latency counters
``POST /predict``        classify one article
``POST /predict/batch``  classify a bounded batch
``GET  /``               service banner with links to the docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fake_news_detection import get_version
from fake_news_detection.api.dependencies import (
    ClientRateLimiter,
    RequestContextMiddleware,
    RequestSizeLimitMiddleware,
    get_latency_tracker,
    get_model_service,
    get_settings,
)
from fake_news_detection.api.schemas import (
    AttributionResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
    PredictRequest,
    ReadyResponse,
)
from fake_news_detection.api.service import ModelService, ServiceNotReadyError
from fake_news_detection.config import ServiceSettings
from fake_news_detection.monitoring.logging import LatencyTracker, configure_logging

LOGGER = logging.getLogger(__name__)

# Starlette renamed this constant across versions; prefer the new name when present.
HTTP_422_UNPROCESSABLE = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)

DESCRIPTION = """\
Classify a news headline and/or article body as resembling **REAL** or **FAKE**
examples from the ISOT Fake and Real News dataset.

The response is a **model classification**, not a fact-check. A high probability
means the text resembles the labelled class in the training corpus; it is not
evidence that a claim is true or false. See the repository model card for known
biases, dataset artifacts and out-of-scope uses.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at start-up and release it on shutdown."""
    settings: ServiceSettings = get_settings()
    configure_logging(settings.log_level, json_output=True, service=settings.app_name)

    tracker = getattr(app.state, "latency_tracker", None) or LatencyTracker()
    app.state.latency_tracker = tracker
    service = ModelService(settings)
    app.state.model_service = service

    LOGGER.info(
        "starting %s v%s",
        settings.app_name,
        get_version(),
        extra={"artifact": settings.artifact_dir},
    )
    loaded = service.load()
    if not loaded and settings.fail_fast_on_missing_model:  # pragma: no cover - opt-in
        raise RuntimeError(f"model could not be loaded: {service.error}")
    if not loaded:
        LOGGER.warning(
            "model is not available; /ready will report 503 until artifacts are present "
            "(train with `python scripts/train.py`)"
        )
    try:
        yield
    finally:
        LOGGER.info("shutting down %s", settings.app_name)


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    """Application factory. Tests build isolated apps from this function."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Fake News Detection API",
        description=DESCRIPTION,
        version=get_version(),
        lifespan=lifespan,
        contact={"name": "Fake News Detection contributors"},
        license_info={"name": "MIT"},
    )

    # A single tracker instance is shared between the middleware and the
    # /metrics route so the two report the same numbers.
    tracker = LatencyTracker()
    app.state.latency_tracker = tracker
    app.state.model_service = None

    # Middleware order: last added runs first. Size limits and rate limits must
    # run before any model work happens.
    app.add_middleware(
        RequestContextMiddleware,
        tracker=tracker,
        path_filter=("/health", "/ready", "/metrics"),
    )
    app.add_middleware(
        ClientRateLimiter,
        requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
        enabled=settings.rate_limit_enabled,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    _register_routes(app, settings)
    _register_error_handlers(app)
    return app


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI, settings: ServiceSettings) -> None:
    @app.get("/", tags=["service"], summary="Service banner")
    async def root() -> dict[str, Any]:
        return {
            "service": settings.app_name,
            "version": get_version(),
            "docs": "/docs",
            "openapi": "/openapi.json",
            "endpoints": [
                "/health",
                "/ready",
                "/metadata",
                "/metrics",
                "/predict",
                "/predict/batch",
            ],
            "disclaimer": settings.disclaimer,
        }

    @app.get("/health", tags=["service"], response_model=HealthResponse, summary="Liveness probe")
    async def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=get_version(),
            service=settings.app_name,
            uptime_seconds=service.uptime_seconds,
            model_loaded=service.ready,
        )

    @app.get(
        "/ready",
        tags=["service"],
        response_model=ReadyResponse,
        summary="Readiness probe",
        responses={503: {"description": "Model is not loaded"}},
    )
    async def ready(service: ModelService = Depends(get_model_service)) -> JSONResponse:
        if service.ready:
            payload = ReadyResponse(ready=True, model_loaded=True, detail=None)
            return JSONResponse(status_code=status.HTTP_200_OK, content=payload.model_dump())
        payload = ReadyResponse(
            ready=False,
            model_loaded=False,
            detail=service.error or "the model has not been loaded",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump()
        )

    @app.get(
        "/metadata",
        tags=["service"],
        response_model=MetadataResponse,
        summary="Model metadata",
        responses={503: {"description": "Model is not loaded"}},
    )
    async def metadata(service: ModelService = Depends(get_model_service)) -> MetadataResponse:
        if not service.ready or service.metadata is None:
            raise ServiceNotReadyError(service.error or "model is not loaded")
        info = service.metadata
        return MetadataResponse(
            model_name=info.model_name,
            model_version=info.model_version,
            created_at=info.created_at,
            labels=info.labels,
            positive_label=info.positive_label,
            decision_threshold=service.decision_threshold,
            text_field=info.text_field,
            calibration=info.calibration,
            training=info.training,
            dataset=info.dataset,
            validation_metrics=info.validation_metrics,
            test_metrics=info.test_metrics,
            attribution_supported=service.attribution_supported(),
            disclaimer=settings.disclaimer,
        )

    @app.get("/metrics", tags=["service"], summary="In-process latency counters")
    async def metrics(tracker: LatencyTracker = Depends(get_latency_tracker)) -> dict[str, Any]:
        return tracker.snapshot()

    @app.post(
        "/predict",
        tags=["prediction"],
        response_model=PredictionResponse,
        summary="Classify one article",
        responses={
            400: {"description": "Service-level input rejection after schema validation"},
            422: {"description": "Request failed schema validation (empty, oversized or unknown)"},
            503: {"description": "Model is not loaded"},
        },
    )
    async def predict(
        payload: PredictRequest,
        request: Request,
        service: ModelService = Depends(get_model_service),
    ) -> PredictionResponse:
        return _to_response(
            service.predict(payload.title, payload.text, explain=payload.explain), request, settings
        )

    @app.post(
        "/predict/batch",
        tags=["prediction"],
        response_model=BatchPredictResponse,
        summary="Classify a batch of articles",
        responses={
            400: {"description": "Service-level input rejection after schema validation"},
            422: {"description": "Request failed schema validation (empty, oversized or unknown)"},
            503: {"description": "Model is not loaded"},
        },
    )
    async def predict_batch(
        payload: BatchPredictRequest,
        request: Request,
        service: ModelService = Depends(get_model_service),
    ) -> BatchPredictResponse:
        records = [{"title": item.title or "", "text": item.text or ""} for item in payload.items]
        explain = any(item.explain for item in payload.items)
        predictions = service.predict_batch(records, explain=explain)
        request_id = getattr(request.state, "request_id", None)
        return BatchPredictResponse(
            predictions=[_to_response(p, request, settings) for p in predictions],
            model_version=predictions[0].model_version if predictions else "",
            disclaimer=settings.disclaimer,
            request_id=request_id,
        )


def _to_response(
    prediction: Any, request: Request, settings: ServiceSettings
) -> PredictionResponse:
    attribution = None
    if prediction.attribution is not None:
        attribution = AttributionResponse(**prediction.attribution)
    return PredictionResponse(
        prediction=prediction.label,
        probabilities=prediction.probabilities,
        score=prediction.score,
        positive_label=prediction.positive_label,
        decision_threshold=prediction.decision_threshold,
        model_version=prediction.model_version,
        model_name=prediction.model_name,
        request_id=getattr(request.state, "request_id", None),
        disclaimer=settings.disclaimer,
        attribution=attribution,
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceNotReadyError)
    async def _not_ready(_: Request, exc: ServiceNotReadyError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc), "hint": "check GET /ready"},
        )

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Reduce the errors to JSON-safe fields: `ctx` can hold the original
        # exception object, which is neither serialisable nor useful to a client.
        detail = [
            {
                "loc": [str(item) for item in error.get("loc", ())],
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE,
            content={"detail": detail},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        LOGGER.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "internal server error",
                "request_id": getattr(request.state, "request_id", None),
            },
        )


app = create_app()
