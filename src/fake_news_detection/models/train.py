"""Training orchestration: fit candidates, compare them, pick a winner.

Discipline enforced by this module:

* every candidate sees only the training split;
* the comparison metric is computed on the validation split;
* cross-validation uses ``StratifiedGroupKFold`` over duplicate clusters so a
  near-duplicate cannot sit on both sides of a fold boundary;
* the test split is never touched here.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from fake_news_detection.config import PreprocessingConfig, SelectionConfig, TrainingConfig
from fake_news_detection.evaluation.metrics import (
    MetricsResult,
    evaluate_predictions,
    latency_benchmark,
)
from fake_news_detection.models.baseline import build_pipeline
from fake_news_detection.utils import timer

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CandidateOutcome",
    "cross_validate_candidate",
    "fit_candidate",
    "measure_artifact_size_mb",
    "rank_candidates",
    "select_best",
    "stratified_subsample",
    "train_candidates",
]

# Metrics where a *lower* value is better; everything else is maximised.
LOWER_IS_BETTER = {
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_mean",
    "artifact_size_mb",
    "fit_seconds",
}


@dataclass(slots=True)
class CandidateOutcome:
    """Everything measured for one candidate model."""

    name: str
    pipeline: Any
    validation_metrics: MetricsResult
    fit_seconds: float
    latency: dict[str, float] = field(default_factory=dict)
    artifact_size_mb: float = 0.0
    cross_validation: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        metrics = self.validation_metrics.to_dict()
        return {
            "name": self.name,
            "macro_f1": metrics["macro_f1"],
            "f1_positive": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "fit_seconds": round(self.fit_seconds, 3),
            "latency_ms_p50": self.latency.get("latency_ms_p50"),
            "latency_ms_p95": self.latency.get("latency_ms_p95"),
            "artifact_size_mb": self.artifact_size_mb,
            "cross_validation": self.cross_validation,
            "notes": list(self.notes),
        }


def fit_candidate(
    name: str,
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    training_config: TrainingConfig,
    preprocessing: PreprocessingConfig,
) -> tuple[Any, float]:
    """Fit one candidate on the training split and return ``(pipeline, seconds)``."""
    pipeline = build_pipeline(name, training_config, preprocessing)
    with timer() as elapsed:
        pipeline.fit(train_frame, y_train)
    LOGGER.info("fitted %-22s in %6.1fs", name, elapsed.seconds)
    return pipeline, elapsed.seconds


def measure_artifact_size_mb(pipeline: Any) -> float:
    """Serialise the pipeline the same way the registry does and measure it."""
    import joblib

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.joblib"
        joblib.dump(pipeline, path, compress=3)
        size = path.stat().st_size
    return round(size / (1024 * 1024), 4)


def cross_validate_candidate(
    name: str,
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    training_config: TrainingConfig,
    preprocessing: PreprocessingConfig,
    *,
    positive_label: str,
    labels: list[str],
) -> dict[str, Any]:
    """Leakage-safe grouped cross-validation on a bounded subsample."""
    cv_config = training_config.cv
    if not cv_config.enabled:
        return {"enabled": False}

    frame = train_frame
    y_used = y_train
    if cv_config.max_samples and len(frame) > cv_config.max_samples:
        frame, y_used = stratified_subsample(
            train_frame, y_train, cv_config.max_samples, training_config.seed
        )

    groups = (
        frame["dup_cluster"].to_numpy() if "dup_cluster" in frame.columns else np.arange(len(frame))
    )
    splitter = StratifiedGroupKFold(
        n_splits=cv_config.folds, shuffle=True, random_state=training_config.seed
    )

    scores: list[float] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(frame, y_used, groups)):
        pipeline = build_pipeline(name, training_config, preprocessing)
        pipeline.fit(frame.iloc[train_index], y_used[train_index])
        proba = pipeline.predict_proba(frame.iloc[test_index])
        classes = list(pipeline.classes_)
        positive_index = classes.index(positive_label)
        metrics = evaluate_predictions(
            y_used[test_index],
            proba[:, positive_index],
            positive_label=positive_label,
            labels=labels,
            threshold=0.5,
        )
        scores.append(float(metrics.macro_f1))
        LOGGER.info("  cv fold %d/%d macro_f1=%.4f", fold + 1, cv_config.folds, metrics.macro_f1)

    return {
        "enabled": True,
        "folds": cv_config.folds,
        "samples": len(frame),
        "macro_f1_mean": round(float(np.mean(scores)), 5),
        "macro_f1_std": round(float(np.std(scores)), 5),
        "macro_f1_folds": [round(value, 5) for value in scores],
    }


def stratified_subsample(
    frame: pd.DataFrame, y: np.ndarray, max_samples: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Take a label-stratified subsample, keeping the label mix intact.

    A plain ``head(n)`` is *not* safe here: the corpus concatenates two source
    files, so row order correlates with the label.
    """
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    labels = np.asarray(y).astype(str)
    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        share = round(max_samples * len(index) / len(frame))
        share = max(2, min(share, len(index)))
        keep.extend(rng.choice(index, size=share, replace=False).tolist())
    keep_sorted = sorted(keep)
    return frame.iloc[keep_sorted].reset_index(drop=True), np.asarray(y)[keep_sorted]


def _stratified_subsample(
    frame: pd.DataFrame, y: np.ndarray, max_samples: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Backwards-compatible alias for :func:`stratified_subsample`."""
    return stratified_subsample(frame, y, max_samples, seed)


def _latency_samples(frame: pd.DataFrame, sample_size: int, seed: int) -> list[dict[str, str]]:
    """Draw a deterministic *random* sample of documents for latency measurement.

    Taking ``head(n)`` would be biased: the corpus concatenates two source files,
    so row order is correlated with the label and with document length.
    """
    if frame.empty:
        return []
    count = max(1, min(int(sample_size), len(frame)))
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(frame), size=count, replace=False)
    subset = frame.iloc[np.sort(positions)]
    return [
        {"title": str(row.title), "text": str(row.text)} for row in subset.itertuples(index=False)
    ]


def train_candidates(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    training_config: TrainingConfig,
    preprocessing: PreprocessingConfig,
    *,
    candidates: list[str],
    labels: list[str],
    positive_label: str,
    run_cross_validation: bool = True,
) -> list[CandidateOutcome]:
    """Fit, evaluate and profile every candidate on the training/validation splits."""
    y_train = train_frame["label"].to_numpy()
    y_validation = validation_frame["label"].to_numpy()
    latency_samples = _latency_samples(
        validation_frame, training_config.selection.latency_sample_size, training_config.seed
    )

    outcomes: list[CandidateOutcome] = []
    for name in candidates:
        pipeline, fit_seconds = fit_candidate(
            name, train_frame, y_train, training_config, preprocessing
        )
        proba = pipeline.predict_proba(validation_frame)
        classes = list(pipeline.classes_)
        scores = proba[:, classes.index(positive_label)]
        metrics = evaluate_predictions(
            y_validation,
            scores,
            positive_label=positive_label,
            labels=labels,
            threshold=training_config.threshold.default,
        )

        def _predict(record: dict[str, str], _pipeline: Any = pipeline) -> Any:
            return _pipeline.predict_proba(
                pd.DataFrame({"title": [record["title"]], "text": [record["text"]]})
            )

        latency = latency_benchmark(_predict, latency_samples)
        outcome = CandidateOutcome(
            name=name,
            pipeline=pipeline,
            validation_metrics=metrics,
            fit_seconds=fit_seconds,
            latency=latency,
            artifact_size_mb=measure_artifact_size_mb(pipeline),
        )
        if run_cross_validation:
            outcome.cross_validation = cross_validate_candidate(
                name,
                train_frame,
                y_train,
                training_config,
                preprocessing,
                positive_label=positive_label,
                labels=labels,
            )
        outcomes.append(outcome)
        LOGGER.info(
            "%-22s val macro_f1=%.4f  p95=%.1fms  size=%.1fMiB",
            name,
            metrics.macro_f1,
            latency.get("latency_ms_p95", float("nan")),
            outcome.artifact_size_mb,
        )
    return outcomes


def rank_candidates(
    outcomes: list[CandidateOutcome], selection: SelectionConfig
) -> list[CandidateOutcome]:
    """Sort candidates by the configured metric and tie-breakers (best first)."""
    keys = [selection.primary_metric, *selection.tie_breakers]

    def sort_key(outcome: CandidateOutcome) -> list[float]:
        payload = outcome.to_dict()
        values: list[float] = []
        for key in keys:
            raw = payload.get(key)
            value = float("inf") if raw is None else float(raw)
            if key in LOWER_IS_BETTER:
                values.append(value)
            else:
                values.append(-value)
        return values

    # ``sort`` is stable, so equal candidates keep their original order.
    return sorted(outcomes, key=sort_key)


def select_best(outcomes: list[CandidateOutcome], selection: SelectionConfig) -> CandidateOutcome:
    """Return the winning candidate according to the configured criterion."""
    if not outcomes:
        raise ValueError("no candidates were trained")
    ranked = rank_candidates(outcomes, selection)
    best = ranked[0]
    non_dummy = [outcome for outcome in ranked if outcome.name != "dummy"]
    if non_dummy and best.name == "dummy":
        LOGGER.warning(
            "the dummy baseline out-ranked every real model on %s; check the feature "
            "configuration and the label distribution",
            selection.primary_metric,
        )
    LOGGER.info(
        "selected %s (%s=%.5f)",
        best.name,
        selection.primary_metric,
        best.to_dict().get(selection.primary_metric, float("nan")),
    )
    return best


def clone_pipeline(pipeline: Any) -> Any:
    """Return an unfitted copy of a fitted pipeline."""
    return clone(pipeline)


def validation_frame_latency(
    pipeline: Any, frame: pd.DataFrame, sample_size: int
) -> dict[str, float]:
    """Convenience wrapper used by the evaluation script."""
    samples = [
        {"title": str(row.title), "text": str(row.text)}
        for row in frame.head(max(sample_size, 1)).itertuples(index=False)
    ]

    def _predict(record: dict[str, str]) -> Any:
        return pipeline.predict_proba(
            pd.DataFrame({"title": [record["title"]], "text": [record["text"]]})
        )

    return latency_benchmark(_predict, samples)
