#!/usr/bin/env python
"""Evaluate the saved model on the held-out test split - exactly once.

Produces the headline metrics, confusion matrices, calibration diagnostics,
threshold behaviour, slice-based error analysis and misclassified examples, and
merges the test metrics back into the model metadata so the served model can
report what it actually scored.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --error-examples 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from fake_news_detection.config import load_data_config, repo_root
from fake_news_detection.evaluation.calibration import calibration_summary
from fake_news_detection.evaluation.error_analysis import (
    confidence_profile,
    length_bucket_analysis,
    misclassified_examples,
    slice_metrics,
)
from fake_news_detection.evaluation.metrics import (
    artifact_size_mb,
    evaluate_predictions,
    latency_benchmark,
    threshold_sweep,
)
from fake_news_detection.evaluation.reporting import (
    metrics_markdown_table,
    plot_confusion_matrices,
    plot_pr_curve,
    plot_reliability,
    plot_roc_curve,
    plot_threshold_sweep,
    render_confusion_matrix_markdown,
)
from fake_news_detection.models.predict import Predictor
from fake_news_detection.models.registry import ModelMetadata, load_bundle
from fake_news_detection.monitoring.logging import configure_logging
from fake_news_detection.utils import (
    display_path,
    environment_summary,
    read_json,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger("evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--error-examples", type=int, default=12)
    parser.add_argument("--latency-samples", type=int, default=200)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _report_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Test-set evaluation", ""]
    lines.append(f"- model: **{payload['model_name']}** v{payload['model_version']}")
    lines.append(f"- evaluated at: {payload['created_at']}")
    lines.append(f"- split: `{payload['split']}` ({payload['n_samples']:,} rows)")
    lines.append(
        f"- positive class for all binary metrics: **{payload['positive_label']}** "
        f"(negative class: {payload['negative_label']})"
    )
    lines.append(f"- decision threshold: {payload['threshold']:.2f}")
    lines.append(f"- artifact size: {payload['artifact_size_mb']:.2f} MiB")
    lines.append("")
    if payload.get("quick_mode"):
        lines.append(
            "> **Warning:** the artifact was produced by a quick/smoke training run. These "
            "numbers must not be reported as results."
        )
        lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    lines.append(metrics_markdown_table(payload["metrics"]))
    lines.append("")

    lines.append("## Confusion matrix (rows = actual, columns = predicted)")
    lines.append("")
    lines.append(
        render_confusion_matrix_markdown(payload["metrics"]["confusion_matrix"], payload["labels"])
    )
    lines.append("")
    lines.append("Row-normalised:")
    lines.append("")
    lines.append(
        render_confusion_matrix_markdown(
            [
                [round(value, 4) for value in row]
                for row in payload["metrics"]["confusion_matrix_normalized"]
            ],
            payload["labels"],
        )
    )
    lines.append("")

    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("| class | precision | recall | F1 | support |")
    lines.append("|---|---|---|---|---|")
    for label, stats in payload["metrics"]["per_class"].items():
        lines.append(
            f"| {label} | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | "
            f"{stats['support']:,} |"
        )
    lines.append("")

    lines.append("## Calibration")
    lines.append("")
    lines.append(
        metrics_markdown_table(
            {
                "brier_score": payload["calibration"]["brier_score"],
                "log_loss": payload["calibration"]["log_loss"],
                "expected_calibration_error": payload["calibration"]["expected_calibration_error"],
                "max_calibration_error": payload["calibration"]["max_calibration_error"],
            }
        )
    )
    lines.append("")
    lines.append("| predicted bin | n | mean predicted | observed frequency |")
    lines.append("|---|---|---|---|")
    for entry in payload["calibration"]["reliability_curve"]["bins"]:
        if not entry["count"]:
            continue
        lines.append(
            f"| {entry['lower']:.2f}-{entry['upper']:.2f} | {entry['count']:,} | "
            f"{entry['mean_predicted']:.4f} | {entry['observed_frequency']:.4f} |"
        )
    lines.append("")

    lines.append("## Inference latency")
    lines.append("")
    lines.append(metrics_markdown_table(dict(payload["latency"])))
    lines.append("")

    lines.append("## Error analysis")
    lines.append("")
    for key, title in (
        ("by_subject", "By subject / topic bucket"),
        ("by_length", "By document length"),
        ("by_confidence", "By model confidence"),
    ):
        section = payload["slices"].get(key, {})
        if not section or not section.get("available", True):
            lines.append(f"### {title}")
            lines.append("")
            lines.append("Not available for this dataset.")
            lines.append("")
            continue
        rows = section.get("slices") or section.get("buckets") or []
        if not rows:
            continue
        lines.append(f"### {title}")
        lines.append("")
        header = list(rows[0].keys())
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in header) + " |")
        lines.append("")

    lines.append("### Most confident false positives")
    lines.append("")
    lines.append("| label | predicted | score | subject | headline |")
    lines.append("|---|---|---|---|---|")
    for row in payload["examples"]["false_positives"]:
        lines.append(
            f"| {row['label']} | {row['predicted']} | {row['score_positive']:.4f} | "
            f"{row['subject']} | {row['title']} |"
        )
    lines.append("")
    lines.append("### Most confident false negatives")
    lines.append("")
    lines.append("| label | predicted | score | subject | headline |")
    lines.append("|---|---|---|---|---|")
    for row in payload["examples"]["false_negatives"]:
        lines.append(
            f"| {row['label']} | {row['predicted']} | {row['score_positive']:.4f} | "
            f"{row['subject']} | {row['title']} |"
        )
    lines.append("")

    lines.append("## Threshold sensitivity (test split, for information only)")
    lines.append("")
    lines.append(
        "The threshold was selected on the validation split. This table shows what would have "
        "happened at other thresholds; it is **not** used to change the deployed threshold."
    )
    lines.append("")
    lines.append("| threshold | precision | recall | F1 | predicted positive |")
    lines.append("|---|---|---|---|---|")
    for entry in payload["threshold_sweep"]["thresholds"]:
        lines.append(
            f"| {entry['threshold']:.2f} | {entry['precision']:.4f} | {entry['recall']:.4f} | "
            f"{entry['f1_positive']:.4f} | {entry['predicted_positive']:,} |"
        )
    lines.append("")

    lines.append("## Interpretation limits")
    lines.append("")
    lines.extend(f"- {note}" for note in payload["notes"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, json_output=False, service="evaluate")
    root = repo_root()

    data_config = load_data_config()
    labels = data_config.dataset.labels
    positive_label = data_config.dataset.positive_label
    negative_label = next((label for label in labels if label != positive_label), "")

    split_path = root / args.data_dir / f"{args.split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found; run `python scripts/prepare_data.py` first"
        )
    frame = pd.read_parquet(split_path)
    LOGGER.info("evaluating on %s (%d rows)", args.split, len(frame))

    artifact_path = root / args.artifact_dir / "model.joblib"
    metadata_path = root / args.artifact_dir / "model_metadata.json"
    bundle = load_bundle(artifact_path, metadata_path)
    predictor = Predictor.from_bundle(bundle, explain=False)
    threshold = predictor.threshold

    scores = bundle.pipeline.predict_proba(frame)[
        :, list(bundle.pipeline.classes_).index(positive_label)
    ]
    y_true = frame["label"].to_numpy()
    y_pred = (scores >= threshold).astype(int)

    metrics = evaluate_predictions(
        y_true, scores, positive_label=positive_label, labels=labels, threshold=threshold
    )
    calibration = calibration_summary(y_true, scores, positive_label=positive_label)
    sweep = threshold_sweep(
        y_true,
        scores,
        positive_label=positive_label,
        grid=[round(0.05 * index, 2) for index in range(1, 20)],
    )

    samples = [
        {"title": str(row.title), "text": str(row.text)}
        for row in frame.head(min(args.latency_samples, len(frame))).itertuples(index=False)
    ]

    def _predict(record: dict[str, str]) -> Any:
        return bundle.pipeline.predict_proba(
            pd.DataFrame({"title": [record["title"]], "text": [record["text"]]})
        )

    latency = latency_benchmark(_predict, samples)

    slices: dict[str, Any] = {
        "by_subject": slice_metrics(
            frame,
            y_pred,
            scores,
            column="subject",
            positive_label=positive_label,
            labels=labels,
        ),
        "by_length": length_bucket_analysis(frame, y_pred, scores, positive_label=positive_label),
        "by_confidence": confidence_profile(frame, y_pred, scores, positive_label=positive_label),
    }

    examples = misclassified_examples(
        frame,
        y_pred,
        scores,
        positive_label=positive_label,
        negative_label=negative_label,
        per_type=args.error_examples,
    )

    figures = root / args.reports_dir / "figures"
    plot_confusion_matrices(
        metrics.confusion_matrix,
        metrics.confusion_matrix_normalized,
        labels=labels,
        path=figures / f"{args.split}_confusion_matrix.png",
    )
    plot_reliability(
        calibration["reliability_curve"],
        path=figures / f"{args.split}_reliability_diagram.png",
    )
    y_binary = (y_true == positive_label).astype(int)
    plot_roc_curve(y_binary, scores, path=figures / f"{args.split}_roc_curve.png")
    plot_pr_curve(y_binary, scores, path=figures / f"{args.split}_pr_curve.png")
    plot_threshold_sweep(
        sweep, path=figures / f"{args.split}_threshold_sweep.png", threshold=threshold
    )

    payload = {
        "created_at": utc_now_iso(),
        "split": args.split,
        "model_name": bundle.metadata.model_name,
        "model_version": bundle.metadata.model_version,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "labels": labels,
        "threshold": threshold,
        "n_samples": len(frame),
        "quick_mode": bool(bundle.metadata.training.get("quick_mode")),
        "dataset_fingerprint": bundle.metadata.dataset.get("fingerprint", ""),
        "metrics": metrics.to_dict(),
        "calibration": calibration,
        "threshold_sweep": sweep,
        "latency": latency,
        "artifact_size_mb": artifact_size_mb(artifact_path),
        "slices": slices,
        "examples": examples,
        "environment": environment_summary(),
        "notes": [
            "F1/recall/precision refer to the positive class stated above.",
            "The test split is read once, here. Model selection, calibration and threshold tuning "
            "used only the training and validation splits.",
            "The corpus contains strong publisher and topic artifacts (see "
            "reports/data_quality_report.md); in-distribution accuracy is an upper bound on what "
            "to expect on genuinely new publishers.",
            "A high probability is not evidence that a claim is true or false.",
        ],
    }

    reports_dir = root / args.reports_dir
    write_json(reports_dir / f"{args.split}_evaluation.json", payload)
    (reports_dir / f"{args.split}_evaluation.md").write_text(_report_markdown(payload), "utf-8")

    # Merge the test metrics into the served metadata so /metadata can report them.
    if args.split == "test":
        raw_metadata = read_json(metadata_path)
        raw_metadata["test_metrics"] = metrics.to_dict()
        raw_metadata["test_calibration"] = {
            key: value for key, value in calibration.items() if key != "reliability_curve"
        }
        raw_metadata["evaluated_at"] = payload["created_at"]
        write_json(metadata_path, raw_metadata)
        ModelMetadata.model_validate(raw_metadata)
        LOGGER.info("updated %s with test metrics", display_path(metadata_path, root))

    LOGGER.info(
        "%s: accuracy=%.4f macro_f1=%.4f f1(%s)=%.4f roc_auc=%s",
        args.split,
        metrics.accuracy,
        metrics.macro_f1,
        positive_label,
        metrics.f1,
        "n/a" if metrics.roc_auc is None else f"{metrics.roc_auc:.4f}",
    )
    LOGGER.info(
        "report: %s",
        display_path(reports_dir / f"{args.split}_evaluation.md", root),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
