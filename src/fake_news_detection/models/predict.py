"""Inference: turning raw headline/body text into a structured prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fake_news_detection.features.tfidf import (
    explain_document,
    supports_attribution,
    unwrap_pipeline,
)
from fake_news_detection.models.registry import ModelBundle, ModelMetadata, load_bundle

LOGGER = logging.getLogger(__name__)

__all__ = ["EmptyInputError", "InferenceError", "Prediction", "Predictor"]


class InferenceError(RuntimeError):
    """Base class for inference-time problems."""


class EmptyInputError(InferenceError):
    """Raised when neither the headline nor the body carries usable text."""


@dataclass(slots=True)
class Prediction:
    """One structured prediction."""

    label: str
    positive_label: str
    score: float
    probabilities: dict[str, float]
    decision_threshold: float
    model_version: str
    model_name: str
    attribution: dict[str, Any] | None = field(default=None)

    @property
    def confidence(self) -> float:
        """Probability assigned to the predicted label."""
        return float(self.probabilities.get(self.label, self.score))

    def to_dict(self, *, round_to: int | None = 4) -> dict[str, Any]:
        """JSON-friendly view with optional rounding."""
        probabilities = (
            {key: round(float(value), round_to) for key, value in self.probabilities.items()}
            if round_to is not None
            else {key: float(value) for key, value in self.probabilities.items()}
        )
        return {
            "prediction": self.label,
            "probabilities": probabilities,
            "score": round(float(self.score), round_to) if round_to is not None else self.score,
            "decision_threshold": self.decision_threshold,
            "model_version": self.model_version,
            "model_name": self.model_name,
        }


class Predictor:
    """Wraps a fitted pipeline with input validation and label mapping."""

    def __init__(
        self,
        pipeline: Any,
        metadata: ModelMetadata,
        *,
        decision_threshold: float | None = None,
        explain: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self.metadata = metadata
        self.labels = list(metadata.labels)
        self.positive_label = metadata.positive_label
        self.threshold = float(
            metadata.decision_threshold if decision_threshold is None else decision_threshold
        )
        if not 0.0 < self.threshold < 1.0:
            raise InferenceError(f"decision threshold must be in (0, 1), got {self.threshold}")
        self.explain = explain
        self._classes = self._resolve_classes()
        # Fail fast rather than at the first request.
        self._pipeline.predict_proba(
            pd.DataFrame({"title": ["warmup"], "text": ["warmup article body"]})
        )

    # -- construction --------------------------------------------------------
    @classmethod
    def from_bundle(
        cls, bundle: ModelBundle, *, decision_threshold: float | None = None, explain: bool = False
    ) -> Predictor:
        """Build a predictor from an already-loaded bundle."""
        return cls(
            bundle.pipeline,
            bundle.metadata,
            decision_threshold=decision_threshold,
            explain=explain,
        )

    @classmethod
    def from_artifacts(
        cls,
        artifact_path: str | Path,
        metadata_path: str | Path,
        *,
        decision_threshold: float | None = None,
        explain: bool = False,
        verify_checksum: bool = True,
    ) -> Predictor:
        """Load the artifacts and build a predictor in one step."""
        bundle = load_bundle(artifact_path, metadata_path, verify_checksum=verify_checksum)
        return cls.from_bundle(bundle, decision_threshold=decision_threshold, explain=explain)

    # -- public API ----------------------------------------------------------
    def predict(
        self, title: str | None = None, text: str | None = None, *, explain: bool | None = None
    ) -> Prediction:
        """Predict a single document."""
        frame = pd.DataFrame({"title": [title or ""], "text": [text or ""]})
        return self._predict_frame(frame, explain=explain)[0]

    def predict_many(
        self,
        records: list[dict[str, str | None]],
        *,
        explain: bool | None = None,
    ) -> list[Prediction]:
        """Predict a batch of documents in a single vectorised call."""
        if not records:
            return []
        frame = pd.DataFrame(
            {
                "title": [record.get("title") or "" for record in records],
                "text": [record.get("text") or "" for record in records],
            }
        )
        return self._predict_frame(frame, explain=explain)

    # -- internals -----------------------------------------------------------
    def _resolve_classes(self) -> list[str]:
        classes = list(getattr(self._pipeline, "classes_", self.labels))
        if set(classes) != set(self.labels):
            raise InferenceError(
                f"model classes {classes} do not match metadata labels {self.labels}; "
                "the artifact and its metadata are inconsistent"
            )
        return [str(c) for c in classes]

    def _predict_frame(
        self, frame: pd.DataFrame, *, explain: bool | None = None
    ) -> list[Prediction]:
        if (frame["title"].str.strip() == "").all() and (frame["text"].str.strip() == "").all():
            raise EmptyInputError("provide a headline, an article body, or both")

        proba = np.asarray(self._pipeline.predict_proba(frame), dtype=np.float64)
        positive_index = self._classes.index(self.positive_label)
        scores = proba[:, positive_index]

        want_explanation = self.explain if explain is None else explain
        attribution: list[dict[str, Any] | None] = [None] * len(frame)
        if want_explanation:
            combined = self._combined_texts(frame)
            attribution = list(
                explain_document(self._pipeline, combined, positive_label=self.positive_label)
            )

        results: list[Prediction] = []
        for row, score in enumerate(scores.tolist()):
            probabilities = {
                str(label): float(proba[row, index]) for index, label in enumerate(self._classes)
            }
            label = self.positive_label if score >= self.threshold else self._other_label()
            results.append(
                Prediction(
                    label=label,
                    positive_label=self.positive_label,
                    score=float(score),
                    probabilities=probabilities,
                    decision_threshold=self.threshold,
                    model_version=self.metadata.model_version,
                    model_name=self.metadata.model_name,
                    attribution=attribution[row] if want_explanation else None,
                )
            )
        return results

    def _combined_texts(self, frame: pd.DataFrame) -> list[str]:
        """Reproduce the model's own text combination for attribution."""
        pipeline = unwrap_pipeline(self._pipeline)
        combiner = pipeline.named_steps.get("text") if pipeline is not None else None
        if combiner is None:
            return (frame["title"].fillna("") + " " + frame["text"].fillna("")).tolist()
        return [str(item) for item in combiner.transform(frame)]

    def attribution_supported(self) -> bool:
        """Whether the loaded model exposes a linear decision function."""
        pipeline = unwrap_pipeline(self._pipeline)
        if pipeline is None:
            return False
        return supports_attribution(pipeline.steps[-1][1], self.positive_label)

    def _other_label(self) -> str:
        others = [label for label in self._classes if label != self.positive_label]
        return others[0] if others else self.positive_label
