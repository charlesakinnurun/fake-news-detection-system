"""Confidence calibration diagnostics.

``predict_proba`` output is *not* automatically a trustworthy confidence
estimate. This module measures how far the scores are from calibrated and fits
an optional post-hoc calibrator.

Important interpretation caveat, repeated in the model card: a calibrated
probability estimates ``P(label | text)`` **under the evaluation distribution of
this dataset**. It is not the probability that a claim is objectively true or
false.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from fake_news_detection.evaluation.metrics import binary_labels

LOGGER = logging.getLogger(__name__)

__all__ = [
    "calibration_summary",
    "fit_calibrator",
    "reliability_curve",
    "supports_probabilities",
]

_EPSILON = 1e-12


def supports_probabilities(model: Any) -> bool:
    """Return ``True`` when the model exposes ``predict_proba``."""
    return hasattr(model, "predict_proba")


def reliability_curve(
    y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10
) -> dict[str, Any]:
    """Return equal-width reliability-diagram data for the positive class."""
    y_score = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, n_bins - 1)

    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        bins.append(
            {
                "bin": index,
                "lower": round(float(edges[index]), 4),
                "upper": round(float(edges[index + 1]), 4),
                "count": count,
                "mean_predicted": round(float(y_score[mask].mean()), 5) if count else None,
                "observed_frequency": round(float(y_true[mask].mean()), 5) if count else None,
            }
        )
    return {"n_bins": n_bins, "bins": bins, "samples": len(y_score)}


def expected_calibration_error(rel_curve: dict[str, Any]) -> tuple[float, float]:
    """Compute ECE and MCE from a reliability curve."""
    total = max(rel_curve["samples"], 1)
    ece = 0.0
    mce = 0.0
    for entry in rel_curve["bins"]:
        if not entry["count"]:
            continue
        gap = abs(float(entry["observed_frequency"]) - float(entry["mean_predicted"]))
        ece += gap * entry["count"] / total
        mce = max(mce, gap)
    return round(ece, 5), round(mce, 5)


def calibration_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: str = "FAKE",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Brier score, log loss, reliability curve and ECE/MCE for the scores."""
    y_binary = binary_labels(y_true, positive_label)
    y_score = np.clip(np.asarray(y_score, dtype=np.float64), _EPSILON, 1 - _EPSILON)
    curve = reliability_curve(y_binary, y_score, n_bins=n_bins)
    ece, mce = expected_calibration_error(curve)
    return {
        "positive_label": str(positive_label),
        "brier_score": round(float(brier_score_loss(y_binary, y_score)), 5),
        "log_loss": round(float(log_loss(y_binary, y_score, labels=[0, 1])), 5),
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
        "reliability_curve": curve,
        "note": (
            "Calibration is measured on the split named in the report. It describes agreement "
            "between predicted probabilities and observed label frequencies in that distribution; "
            "it says nothing about factual truth."
        ),
    }


def fit_calibrator(
    pipeline: Any,
    validation_frame: Any,
    validation_labels: np.ndarray,
    *,
    method: str = "isotonic",
) -> tuple[Any, dict[str, Any]]:
    """Wrap a fitted pipeline in a post-hoc calibrator trained on validation data.

    Returns ``(model, info)``. When the estimator has no ``predict_proba`` (or the
    calibration fit fails) the original pipeline is returned unchanged and
    ``info["applied"]`` is ``False`` - the code never pretends a model is
    calibrated when it is not.
    """
    info: dict[str, Any] = {"requested": method, "applied": False}
    if not supports_probabilities(pipeline):
        info["reason"] = "estimator does not expose predict_proba"
        return pipeline, info

    try:  # sklearn >= 1.6 recommends freezing the fitted estimator explicitly.
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(pipeline), method=method)
    except ImportError:  # pragma: no cover - older sklearn
        calibrator = CalibratedClassifierCV(pipeline, method=method, cv="prefit")

    try:
        calibrator.fit(validation_frame, validation_labels)
    except Exception as error:  # pragma: no cover - defensive
        LOGGER.warning("calibration failed (%s); keeping uncalibrated model", error)
        info["reason"] = f"calibration failed: {error}"
        return pipeline, info

    info["applied"] = True
    info["method"] = method
    info["fitted_on"] = "validation split"
    info["n_samples"] = len(validation_labels)
    return calibrator, info
