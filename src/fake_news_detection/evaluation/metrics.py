"""Binary classification metrics, threshold tuning and latency measurement.

Convention used everywhere in this project
------------------------------------------
``positive_label`` defaults to ``FAKE``. Every binary metric (precision, recall,
F1, ROC-AUC, PR-AUC) therefore refers to *detecting the FAKE class*. This is
stated explicitly in every report because "high recall" is meaningless without
knowing which class it refers to.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics

from fake_news_detection.utils import percentile, timer

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MetricsResult",
    "artifact_size_mb",
    "binary_labels",
    "evaluate_predictions",
    "latency_benchmark",
    "threshold_sweep",
    "tune_threshold",
]

_EPSILON = 1e-12


def _finite(value: float | None) -> float | None:
    """Return ``value`` when it is a finite float, else ``None``.

    ``cohen_kappa_score`` and ``matthews_corrcoef`` are undefined when only one
    class is present; a ``NaN`` would serialise to invalid JSON, so it is
    reported as ``null`` instead.
    """
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def binary_labels(y_true: np.ndarray, positive_label: str) -> np.ndarray:
    """Map string/object labels onto ``1 == positive_label``."""
    return (np.asarray(y_true).astype(str) == str(positive_label)).astype(int)


@dataclass(slots=True)
class MetricsResult:
    """All metrics computed for one (labels, scores) pair at one threshold."""

    n_samples: int
    n_positive: int
    n_negative: int
    positive_label: str
    threshold: float
    accuracy: float
    balanced_accuracy: float | None
    precision: float
    recall: float
    f1: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float | None
    average_precision: float | None
    brier_score: float | None
    log_loss: float | None
    cohen_kappa: float | None
    matthews_corrcoef: float | None
    confusion_matrix: list[list[int]]
    confusion_matrix_normalized: list[list[float]]
    per_class: dict[str, dict[str, float]]
    labels_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: str = "FAKE",
    labels: list[str] | None = None,
    threshold: float = 0.5,
) -> MetricsResult:
    """Compute the full metric suite for ``P(positive_label)`` scores."""
    labels = labels or ["FAKE", "REAL"]
    y_true = np.asarray(y_true).astype(str)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_binary = (y_true == str(positive_label)).astype(int)
    y_pred = (y_score >= threshold).astype(int)
    negative_label = _other_label(labels, positive_label)

    n_positive = int(y_binary.sum())
    n_negative = int(len(y_binary) - n_positive)
    both_present = n_positive > 0 and n_negative > 0

    precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
        y_binary, y_pred, average="binary", zero_division=0
    )
    macro_f1 = sk_metrics.f1_score(y_binary, y_pred, average="macro", zero_division=0)
    weighted_f1 = sk_metrics.f1_score(y_binary, y_pred, average="weighted", zero_division=0)

    roc_auc = float(sk_metrics.roc_auc_score(y_binary, y_score)) if both_present else None
    average_precision = (
        float(sk_metrics.average_precision_score(y_binary, y_score)) if both_present else None
    )
    brier_score = (
        float(sk_metrics.brier_score_loss(y_binary, np.clip(y_score, _EPSILON, 1 - _EPSILON)))
        if both_present
        else None
    )
    log_loss = (
        float(
            sk_metrics.log_loss(y_binary, np.clip(y_score, _EPSILON, 1 - _EPSILON), labels=[0, 1])
        )
        if both_present
        else None
    )

    matrix = sk_metrics.confusion_matrix(y_binary, y_pred, labels=[0, 1])
    normalized = sk_metrics.confusion_matrix(y_binary, y_pred, labels=[0, 1], normalize="true")
    # sklearn orders binary rows as [negative, positive]. Reorder to the
    # declared ``labels`` order so that reports and plots cannot silently
    # swap the classes.
    binary_index = {str(negative_label): 0, str(positive_label): 1}
    order = [binary_index[label] for label in labels if label in binary_index]
    if len(order) == matrix.shape[0]:
        matrix = matrix[np.ix_(order, order)]
        normalized = normalized[np.ix_(order, order)]

    report = sk_metrics.classification_report(
        y_true,
        np.where(y_pred == 1, str(positive_label), negative_label),
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        str(label): {
            "precision": round(float(report[label]["precision"]), 5),
            "recall": round(float(report[label]["recall"]), 5),
            "f1": round(float(report[label]["f1-score"]), 5),
            "support": int(report[label]["support"]),
        }
        for label in labels
        if label in report
    }

    return MetricsResult(
        n_samples=len(y_binary),
        n_positive=n_positive,
        n_negative=n_negative,
        positive_label=str(positive_label),
        threshold=float(threshold),
        accuracy=float(sk_metrics.accuracy_score(y_binary, y_pred)),
        balanced_accuracy=_finite(sk_metrics.balanced_accuracy_score(y_binary, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        roc_auc=roc_auc,
        average_precision=average_precision,
        brier_score=brier_score,
        log_loss=log_loss,
        cohen_kappa=_finite(sk_metrics.cohen_kappa_score(y_binary, y_pred)),
        matthews_corrcoef=_finite(sk_metrics.matthews_corrcoef(y_binary, y_pred)),
        confusion_matrix=matrix.astype(int).tolist(),
        confusion_matrix_normalized=[[round(float(v), 5) for v in row] for row in normalized],
        per_class=per_class,
        labels_order=list(labels),
    )


def _other_label(labels: list[str], positive_label: str) -> str:
    others = [label for label in labels if label != positive_label]
    return others[0] if others else positive_label


def tune_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: str,
    grid: list[float],
    metric: str = "f1_positive",
) -> dict[str, Any]:
    """Select a decision threshold on the validation split only."""
    sweep = threshold_sweep(
        y_true, y_score, positive_label=positive_label, grid=grid, metric=metric
    )
    if not sweep["thresholds"]:
        return {"threshold": 0.5, "metric": metric, "best_value": None, "curve": sweep}
    values = np.asarray(
        [entry[metric] if entry[metric] is not None else -1.0 for entry in sweep["thresholds"]],
        dtype=np.float64,
    )
    best_index = int(np.argmax(values))
    return {
        "threshold": float(sweep["thresholds"][best_index]["threshold"]),
        "metric": metric,
        "best_value": float(values[best_index]),
        "curve": sweep,
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: str,
    grid: list[float],
    metric: str = "f1_positive",
) -> dict[str, Any]:
    """Evaluate precision/recall/F1 across a threshold grid."""
    y_binary = binary_labels(y_true, positive_label)
    entries: list[dict[str, Any]] = []
    for threshold in grid:
        predicted = (y_score >= threshold).astype(int)
        precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
            y_binary, predicted, average="binary", zero_division=0
        )
        tn, fp, fn, _ = sk_metrics.confusion_matrix(y_binary, predicted, labels=[0, 1]).ravel()
        entries.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": round(float(precision), 5),
                "recall": round(float(recall), 5),
                "f1_positive": round(float(f1), 5),
                "balanced_accuracy": round(
                    float(sk_metrics.balanced_accuracy_score(y_binary, predicted)), 5
                ),
                "youden": round(float(recall + (tn / max(tn + fp, 1)) - 1.0), 5),
                "predicted_positive": int(predicted.sum()),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )
    return {
        "metric": metric,
        "positive_label": str(positive_label),
        "thresholds": entries,
    }


def artifact_size_mb(path: str | Path) -> float:
    """Return the artifact size in MiB (0.0 when missing)."""
    path = Path(path)
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 4)


def latency_benchmark(
    predict_one: Callable[[dict[str, str]], Any],
    samples: list[dict[str, str]],
    *,
    warmup: int = 3,
    batch_size: int = 1,
) -> dict[str, float]:
    """Measure per-request inference latency on a small sample of documents.

    ``predict_one`` receives a ``{"title": ..., "text": ...}`` mapping so that the
    measurement includes the same input handling the API performs.
    """
    if not samples:
        return {"samples": 0}
    for record in samples[:warmup]:
        predict_one(record)

    durations: list[float] = []
    with timer() as elapsed:
        for record in samples:
            start = time.perf_counter()
            predict_one(record)
            durations.append((time.perf_counter() - start) * 1000.0)

    return {
        "samples": len(samples),
        "batch_size": batch_size,
        "latency_ms_mean": round(float(np.mean(durations)), 4),
        "latency_ms_p50": round(percentile(durations, 50), 4),
        "latency_ms_p95": round(percentile(durations, 95), 4),
        "latency_ms_max": round(float(np.max(durations)), 4),
        "throughput_per_second": round(len(durations) / max(elapsed.seconds, _EPSILON), 3),
    }
