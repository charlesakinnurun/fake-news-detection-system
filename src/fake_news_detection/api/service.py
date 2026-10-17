"""The model service: artifact lifecycle and prediction orchestration.

The model is loaded exactly once during application start-up. Request handling
never reads the model from disk, never downloads anything and never retrains.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from fake_news_detection.config import ServiceSettings
from fake_news_detection.models.predict import (
    EmptyInputError,
    InferenceError,
    Prediction,
    Predictor,
)
from fake_news_detection.models.registry import ArtifactError, ModelMetadata

LOGGER = logging.getLogger(__name__)

__all__ = ["ModelService", "ServiceNotReadyError"]


class ServiceNotReadyError(RuntimeError):
    """Raised when a prediction is requested before the model is available."""


class ModelService:
    """Owns the loaded predictor and exposes a small, testable surface."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._predictor: Predictor | None = None
        self._metadata: ModelMetadata | None = None
        self._error: str | None = None
        self._loaded_at: float | None = None
        self._lock = threading.Lock()
        self._started_at = time.time()

    # -- lifecycle -----------------------------------------------------------
    @property
    def ready(self) -> bool:
        """Whether a model is loaded and able to serve predictions."""
        return self._predictor is not None

    @property
    def error(self) -> str | None:
        """Human-readable reason the model is unavailable (``None`` when ready)."""
        return self._error

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the service object was created."""
        return round(time.time() - self._started_at, 1)

    def load(self) -> bool:
        """Load the model artifacts. Returns ``True`` on success.

        Failures are captured rather than raised so that the process can start and
        report ``/ready == false``; this is what Kubernetes-style probes expect.
        """
        with self._lock:
            artifact = Path(self.settings.artifact_path)
            metadata = Path(self.settings.metadata_path)
            try:
                self._predictor = Predictor.from_artifacts(
                    artifact,
                    metadata,
                    decision_threshold=self.settings.decision_threshold,
                )
                self._metadata = self._predictor.metadata
                self._error = None
                self._loaded_at = time.time()
                LOGGER.info(
                    "model ready: %s v%s (%d candidate classes, threshold=%.3f)",
                    self._metadata.model_name,
                    self._metadata.model_version,
                    len(self._metadata.labels),
                    self._predictor.threshold,
                )
                return True
            except (ArtifactError, InferenceError, FileNotFoundError, OSError) as error:
                self._predictor = None
                self._metadata = None
                self._error = f"{type(error).__name__}: {error}"
                LOGGER.error("model could not be loaded: %s", self._error)
                return False
            except Exception as error:  # pragma: no cover - unexpected deserialisation issue
                self._predictor = None
                self._metadata = None
                self._error = f"{type(error).__name__}: {error}"
                LOGGER.exception("unexpected error while loading the model")
                return False

    # -- introspection -------------------------------------------------------
    @property
    def metadata(self) -> ModelMetadata | None:
        """Loaded model metadata, if any."""
        return self._metadata

    @property
    def decision_threshold(self) -> float:
        """Effective decision threshold."""
        return self._predictor.threshold if self._predictor else 0.5

    def attribution_supported(self) -> bool:
        """Whether the loaded model supports signed linear attribution."""
        if self._predictor is None:
            return False
        return self._predictor.attribution_supported()

    # -- inference -----------------------------------------------------------
    def predict(
        self, title: str | None, text: str | None, *, explain: bool | None = None
    ) -> Prediction:
        """Predict a single article, raising :class:`ServiceNotReadyError` if unloaded."""
        predictor = self._require_predictor()
        try:
            return predictor.predict(title, text, explain=explain)
        except EmptyInputError as error:
            raise ValueError(str(error)) from error

    def predict_batch(
        self, records: list[dict[str, Any]], *, explain: bool | None = None
    ) -> list[Prediction]:
        """Predict a batch of articles in one vectorised call."""
        predictor = self._require_predictor()
        cleaned = [
            {"title": record.get("title") or "", "text": record.get("text") or ""}
            for record in records
        ]
        try:
            return predictor.predict_many(cleaned, explain=explain)
        except EmptyInputError as error:
            raise ValueError(str(error)) from error

    def _require_predictor(self) -> Predictor:
        if self._predictor is None:
            raise ServiceNotReadyError(
                self._error or "the model is not loaded; check /ready for details"
            )
        return self._predictor
