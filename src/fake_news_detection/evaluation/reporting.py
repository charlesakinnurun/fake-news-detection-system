"""Report generation: plots (optional) and Markdown/JSON artefacts.

Plotting depends on :mod:`matplotlib`, which is only installed through the
``train`` extra. Every plot function degrades to a no-op returning ``None`` when
matplotlib is unavailable, so the pipeline never fails because a chart could not
be drawn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from fake_news_detection.utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)

__all__ = [
    "metrics_markdown_table",
    "plot_confusion_matrices",
    "plot_model_comparison",
    "plot_pr_curve",
    "plot_reliability",
    "plot_roc_curve",
    "plot_threshold_sweep",
    "render_confusion_matrix_markdown",
    "write_markdown",
    "write_report_json",
]


def _plt() -> Any | None:
    """Import matplotlib lazily and configure a headless backend."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "figure.dpi": 130,
                "savefig.dpi": 130,
                "font.size": 9,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.autolayout": True,
            }
        )
        return plt
    except Exception as error:  # pragma: no cover - matplotlib is optional
        LOGGER.warning("matplotlib unavailable, skipping plot: %s", error)
        return None


def _save(figure: Any, path: str | Path) -> Path | None:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        figure.savefig(path, bbox_inches="tight")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return path


def plot_confusion_matrices(
    raw: list[list[int]],
    normalized: list[list[float]],
    *,
    labels: list[str],
    path: str | Path,
) -> Path | None:
    """Plot raw and row-normalised confusion matrices side by side."""
    plt = _plt()
    if plt is None:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for axis, matrix, title, fmt in (
        (axes[0], np.asarray(raw), "Confusion matrix (counts)", "d"),
        (axes[1], np.asarray(normalized), "Confusion matrix (row normalised)", ".2f"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=np.max(matrix) or 1)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.grid(False)
        threshold = np.max(matrix) / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    format(matrix[row, column], fmt),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save(figure, path)


def plot_reliability(
    curve: dict[str, Any], *, path: str | Path, title: str = "Reliability diagram"
) -> Path | None:
    """Plot a reliability diagram with the identity line for reference."""
    plt = _plt()
    if plt is None:
        return None
    points = [
        (bin_["mean_predicted"], bin_["observed_frequency"])
        for bin_ in curve["bins"]
        if bin_["count"] and bin_["mean_predicted"] is not None
    ]
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    if points:
        predicted, observed = zip(*points, strict=True)
        axis.plot(predicted, observed, marker="o", label="Model")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title(title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="best")
    return _save(figure, path)


def plot_roc_curve(y_binary: np.ndarray, y_score: np.ndarray, *, path: str | Path) -> Path | None:
    """Plot the ROC curve."""
    plt = _plt()
    if plt is None:
        return None
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_binary, y_score)
    auc = roc_auc_score(y_binary, y_score)
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curve")
    axis.legend(loc="lower right")
    return _save(figure, path)


def plot_pr_curve(y_binary: np.ndarray, y_score: np.ndarray, *, path: str | Path) -> Path | None:
    """Plot the precision-recall curve."""
    plt = _plt()
    if plt is None:
        return None
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_binary, y_score)
    ap = average_precision_score(y_binary, y_score)
    baseline = float(np.mean(y_binary))
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.plot(recall, precision, label=f"AP = {ap:.4f}")
    axis.axhline(baseline, linestyle="--", color="grey", label=f"Prevalence = {baseline:.3f}")
    axis.set_xlabel("Recall (positive class)")
    axis.set_ylabel("Precision (positive class)")
    axis.set_title("Precision-recall curve")
    axis.legend(loc="best")
    return _save(figure, path)


def plot_threshold_sweep(
    sweep: dict[str, Any], *, path: str | Path, threshold: float | None = None
) -> Path | None:
    """Plot precision/recall/F1 as a function of the decision threshold."""
    plt = _plt()
    if plt is None:
        return None
    entries = sweep.get("thresholds", [])
    if not entries:
        return None
    thresholds = [entry["threshold"] for entry in entries]
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    for key, style in (("precision", "-"), ("recall", "--"), ("f1_positive", "-.")):
        axis.plot(thresholds, [entry[key] for entry in entries], style, label=key)
    if threshold is not None:
        axis.axvline(threshold, color="crimson", linestyle=":", label=f"chosen = {threshold:.2f}")
    axis.set_xlabel("Decision threshold on P(positive)")
    axis.set_ylabel("Score")
    axis.set_title("Threshold trade-off (validation split)")
    axis.set_ylim(0, 1.02)
    axis.legend(loc="best")
    return _save(figure, path)


def plot_model_comparison(candidates: list[dict[str, Any]], *, path: str | Path) -> Path | None:
    """Bar chart of validation macro-F1 and p95 latency per candidate."""
    plt = _plt()
    if plt is None or not candidates:
        return None
    names = [entry["name"] for entry in candidates]
    macro_f1 = [entry.get("macro_f1") or 0.0 for entry in candidates]
    latency = [entry.get("latency_ms_p95") or 0.0 for entry in candidates]

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].barh(names, macro_f1, color="#4c72b0")
    axes[0].set_xlabel("Validation macro-F1")
    axes[0].set_xlim(0, 1.0)
    axes[0].invert_yaxis()
    axes[1].barh(names, latency, color="#dd8452")
    axes[1].set_xlabel("p95 latency (ms/request)")
    axes[1].invert_yaxis()
    figure.suptitle("Candidate comparison (validation split)")
    return _save(figure, path)


def render_confusion_matrix_markdown(matrix: list[list[int]], labels: list[str]) -> str:
    """Render a raw confusion matrix as a Markdown table."""
    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    divider = "|" + "---|" * (len(labels) + 1)
    rows = [
        "| " + labels[row] + " | " + " | ".join(str(value) for value in matrix[row]) + " |"
        for row in range(len(matrix))
    ]
    return "\n".join([header, divider, *rows])


def metrics_markdown_table(rows: dict[str, Any], *, decimals: int = 4) -> str:
    """Render a flat metric mapping as a two-column Markdown table."""
    lines = ["| metric | value |", "|---|---|"]
    for key, value in rows.items():
        if isinstance(value, float):
            rendered = f"{value:.{decimals}f}"
        elif value is None:
            rendered = "n/a"
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {rendered} |")
    return "\n".join(lines)


def write_markdown(path: str | Path, content: str) -> Path:
    """Write a UTF-8 Markdown document, creating parents as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content, "utf-8")
    return path


def write_report_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON report."""
    return write_json(path, payload)
