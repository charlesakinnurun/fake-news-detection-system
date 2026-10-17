"""Slice-based error analysis for a fitted model.

The goal is to find *where* the classifier fails, not to claim that it
generalises. Every slice reports its own support, and slices with small support
are flagged rather than interpreted.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

LOGGER = logging.getLogger(__name__)

__all__ = [
    "confidence_profile",
    "length_bucket_analysis",
    "misclassified_examples",
    "slice_metrics",
]

# Below this many samples a slice is reported but explicitly marked unreliable.
MIN_RELIABLE_SLICE = 30


def _slice_frame(frame: pd.DataFrame, scores: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    working = frame.copy()
    working["_score"] = scores
    working["_pred"] = preds
    return working


def slice_metrics(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    column: str,
    positive_label: str,
    labels: list[str],
    max_slices: int = 25,
) -> dict[str, Any]:
    """Compute per-group performance for a categorical column."""
    if column not in frame.columns:
        return {"column": column, "available": False, "slices": []}

    working = _slice_frame(frame, y_score, y_pred)
    truth = (working["label"].astype(str) == str(positive_label)).astype(int)

    slices: list[dict[str, Any]] = []
    groups = working[column].astype(str).fillna("")
    for value, index in groups.groupby(groups).groups.items():
        index = pd.Index(index)
        subset_truth = truth.loc[index].to_numpy()
        subset_pred = working.loc[index, "_pred"].to_numpy()
        subset_score = working.loc[index, "_score"].to_numpy()
        n = len(index)
        entry: dict[str, Any] = {
            "value": str(value),
            "n": int(n),
            "support_positive": int(subset_truth.sum()),
            "accuracy": round(float(sk_metrics.accuracy_score(subset_truth, subset_pred)), 5),
            "reliable": bool(n >= MIN_RELIABLE_SLICE and 0 < subset_truth.sum() < n),
        }
        if entry["reliable"]:
            entry["precision"] = round(
                float(sk_metrics.precision_score(subset_truth, subset_pred, zero_division=0)),
                5,
            )
            entry["recall"] = round(
                float(sk_metrics.recall_score(subset_truth, subset_pred, zero_division=0)), 5
            )
            entry["f1"] = round(
                float(sk_metrics.f1_score(subset_truth, subset_pred, zero_division=0)), 5
            )
            entry["roc_auc"] = round(float(sk_metrics.roc_auc_score(subset_truth, subset_score)), 5)
        slices.append(entry)

    slices.sort(key=lambda item: (-item["n"], item["value"]))
    return {
        "column": column,
        "available": True,
        "min_reliable_support": MIN_RELIABLE_SLICE,
        "slices": slices[:max_slices],
    }


def length_bucket_analysis(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: str,
    buckets: int = 5,
) -> dict[str, Any]:
    """Performance as a function of document length."""
    if "row_length" not in frame.columns:
        return {"available": False}
    working = _slice_frame(frame, y_score, y_pred)
    truth = (working["label"].astype(str) == str(positive_label)).astype(int)

    try:
        working["_bucket"] = pd.qcut(working["row_length"], q=buckets, duplicates="drop")
    except ValueError:  # pragma: no cover - degenerate input
        return {"available": False}

    slices: list[dict[str, Any]] = []
    for interval, index in working.groupby("_bucket", observed=True).groups.items():
        index = pd.Index(index)
        subset_truth = truth.loc[index].to_numpy()
        subset_pred = working.loc[index, "_pred"].to_numpy()
        entry: dict[str, Any] = {
            "range": f"{int(interval.left)}-{int(interval.right)} chars",
            "n": len(index),
            "accuracy": round(float(sk_metrics.accuracy_score(subset_truth, subset_pred)), 5),
            "reliable": bool(len(index) >= MIN_RELIABLE_SLICE),
        }
        if entry["reliable"] and 0 < subset_truth.sum() < len(index):
            entry["precision"] = round(
                float(sk_metrics.precision_score(subset_truth, subset_pred, zero_division=0)), 5
            )
            entry["recall"] = round(
                float(sk_metrics.recall_score(subset_truth, subset_pred, zero_division=0)), 5
            )
        slices.append(entry)

    slices.sort(key=lambda item: item["n"], reverse=True)
    return {"available": True, "buckets": slices}


def confidence_profile(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: str,
    edges: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.01),
) -> dict[str, Any]:
    """Accuracy as a function of how confident the model was."""
    working = _slice_frame(frame, y_score, y_pred)
    truth = (working["label"].astype(str) == str(positive_label)).astype(int)
    confidence = np.where(y_score >= 0.5, y_score, 1.0 - y_score)
    correct = (working["_pred"].to_numpy() == truth.to_numpy()).astype(int)

    buckets: list[dict[str, Any]] = []
    for lower, upper in itertools.pairwise(edges):
        mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        buckets.append(
            {
                "confidence_range": f"{lower:.2f}-{min(upper, 1.0):.2f}",
                "n": count,
                "share": round(count / max(len(working), 1), 4),
                "accuracy": round(float(correct[mask].mean()), 5) if count else None,
            }
        )
    return {"available": True, "buckets": buckets}


def misclassified_examples(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_label: str,
    negative_label: str = "OTHER",
    per_type: int = 10,
) -> dict[str, Any]:
    """Return the most confident false positives and false negatives.

    Only the (truncated) headline, the label, the score and length metadata are
    recorded. Full article bodies are not copied into reports, both to respect
    dataset licensing and to keep user text out of diagnostic artefacts.
    """
    working = _slice_frame(frame, y_score, y_pred)
    truth = (working["label"].astype(str) == str(positive_label)).astype(int)
    working["_truth"] = truth
    working["_correct"] = (working["_pred"] == working["_truth"]).astype(int)

    def _rows(subset: pd.DataFrame, ascending: bool) -> list[dict[str, Any]]:
        ordered = subset.sort_values("_score", ascending=ascending).head(per_type)
        records: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            title = str(row.get("title", ""))
            records.append(
                {
                    "row_id": str(row.get("row_id", "")),
                    "label": str(row.get("label", "")),
                    "predicted": str(positive_label if row["_pred"] == 1 else negative_label),
                    "score_positive": round(float(row["_score"]), 5),
                    "title": (title[:140] + "…") if len(title) > 140 else title,
                    "subject": str(row.get("subject", "")),
                    "row_length": int(row.get("row_length", 0)) if "row_length" in row else None,
                }
            )
        return records

    false_positives = working[(working["_pred"] == 1) & (working["_truth"] == 0)]
    false_negatives = working[(working["_pred"] == 0) & (working["_truth"] == 1)]

    return {
        "positive_label": str(positive_label),
        "false_positives": _rows(false_positives, ascending=False),
        "false_negatives": _rows(false_negatives, ascending=True),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "note": (
            "Examples are shown with truncated headlines only. A misclassification is evidence "
            "about the model's decision boundary, not evidence about the factual accuracy of the "
            "article."
        ),
    }
