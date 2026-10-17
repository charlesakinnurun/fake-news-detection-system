#!/usr/bin/env python
"""Train, compare and select the fake-news classifier; save the inference artifact.

The script is intentionally strict about evaluation hygiene:

1. candidates are fitted on the **training** split only;
2. cross-validation runs on grouped folds over duplicate clusters;
3. the winner is chosen on the **validation** split (macro-F1, then latency/size);
4. post-hoc calibration and threshold tuning also use the validation split;
5. the **test** split is not read here - ``scripts/evaluate.py`` touches it once.

Usage
-----
    python scripts/train.py
    python scripts/train.py --candidates logistic_regression complement_nb
    python scripts/train.py --quick            # fast smoke run on a subsample
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from fake_news_detection.config import (
    load_data_config,
    load_training_config,
    repo_root,
    validate_config_files,
)
from fake_news_detection.evaluation.calibration import calibration_summary, fit_calibrator
from fake_news_detection.evaluation.metrics import (
    MetricsResult,
    evaluate_predictions,
    tune_threshold,
)
from fake_news_detection.evaluation.reporting import (
    metrics_markdown_table,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_threshold_sweep,
)
from fake_news_detection.features.tfidf import top_features_by_class
from fake_news_detection.models.baseline import configured_candidates
from fake_news_detection.models.registry import ModelMetadata, save_bundle
from fake_news_detection.models.train import (
    rank_candidates,
    select_best,
    stratified_subsample,
    train_candidates,
)
from fake_news_detection.monitoring.logging import configure_logging
from fake_news_detection.utils import (
    display_path,
    environment_summary,
    package_versions,
    read_json,
    set_global_seeds,
    sha256_file,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--training-config", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument(
        "--quick",
        type=int,
        default=0,
        metavar="N",
        help="smoke mode: subsample N training rows (metrics are not meaningful)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = {}
    for name in ("train", "validation", "test"):
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/prepare_data.py` (or `make data`) first."
            )
        frames[name] = pd.read_parquet(path)
    return frames["train"], frames["validation"], frames["test"]


def _md_report(
    payload: dict[str, Any], selection: dict[str, Any], calibration: dict[str, Any]
) -> str:
    lines = ["# Training report", ""]
    lines.append(f"- model version: `{payload['model_version']}`")
    lines.append(f"- selected model: **{payload['model_name']}**")
    lines.append(f"- trained at: {payload['created_at']}")
    lines.append(f"- seed: `{payload['seed']}`")
    lines.append(f"- dataset fingerprint: `{payload['dataset']['fingerprint']}`")
    lines.append("")
    lines.append("## Split sizes")
    lines.append("")
    lines.append("| split | rows | positives (FAKE) | negatives (REAL) |")
    lines.append("|---|---|---|---|")
    for name, info in payload["splits"].items():
        lines.append(f"| {name} | {info['rows']:,} | {info['positive']:,} | {info['negative']:,} |")
    lines.append("")
    lines.append("## Candidate comparison (validation split)")
    lines.append("")
    lines.append(
        "| candidate | macro-F1 | F1 (FAKE) | precision | recall | ROC-AUC | p95 latency (ms) | size (MiB) | fit (s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for candidate in payload["candidates"]:
        lines.append(
            "| {name} | {macro_f1:.4f} | {f1_positive:.4f} | {precision:.4f} | {recall:.4f} | "
            "{roc} | {latency:.2f} | {size:.2f} | {fit:.1f} |".format(
                name=candidate["name"],
                macro_f1=candidate["macro_f1"],
                f1_positive=candidate["f1_positive"],
                precision=candidate["precision"],
                recall=candidate["recall"],
                roc="n/a" if candidate["roc_auc"] is None else f"{candidate['roc_auc']:.4f}",
                latency=candidate["latency_ms_p95"] or 0.0,
                size=candidate["artifact_size_mb"],
                fit=candidate["fit_seconds"],
            )
        )
    lines.append("")
    lines.append("## Selection rule")
    lines.append("")
    lines.append(f"- primary metric: `{selection['primary_metric']}` (validation split)")
    lines.append(f"- tie-breakers: {', '.join(f'`{item}`' for item in selection['tie_breakers'])}")
    lines.append(f"- winner: **{selection['winner']}**")
    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    lines.append(f"```json\n{calibration}\n```")
    lines.append("")
    lines.append("## Final validation metrics (calibrated model, tuned threshold)")
    lines.append("")
    lines.append(metrics_markdown_table(payload["validation_metrics"]))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.extend(f"- {note}" for note in payload["notes"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, json_output=False, service="train")
    root = repo_root()

    data_config = load_data_config(args.data_config)
    training_config = load_training_config(args.training_config)
    validate_config_files(data_config, training_config)
    set_global_seeds(training_config.seed)

    labels = data_config.dataset.labels
    positive_label = data_config.dataset.positive_label

    train_frame, validation_frame, _test_frame = _load_splits(root / args.data_dir)
    notes: list[str] = []
    if args.quick:
        train_frame, _ = stratified_subsample(
            train_frame, train_frame["label"].to_numpy(), args.quick, training_config.seed
        )
        validation_frame, _ = stratified_subsample(
            validation_frame,
            validation_frame["label"].to_numpy(),
            max(args.quick // 4, 40),
            training_config.seed,
        )
        notes.append(
            f"QUICK MODE: trained on {len(train_frame)} rows only; metrics are a smoke test "
            "and must not be reported as results."
        )
        LOGGER.warning(notes[-1])

    candidates = args.candidates or configured_candidates(training_config)
    LOGGER.info("candidates: %s", ", ".join(candidates))

    outcomes = train_candidates(
        train_frame,
        validation_frame,
        training_config,
        data_config.preprocessing,
        candidates=candidates,
        labels=labels,
        positive_label=positive_label,
        run_cross_validation=not args.no_cv and not args.quick,
    )
    best = select_best(outcomes, training_config.selection)

    # --- calibration on the validation split --------------------------------
    calibration_info: dict[str, Any] = {"applied": False, "reason": "disabled by flag"}
    model: Any = best.pipeline
    if training_config.calibration.enabled and not args.no_calibration:
        model, calibration_info = fit_calibrator(
            best.pipeline,
            validation_frame,
            validation_frame["label"].to_numpy(),
            method=training_config.calibration.method,
        )
    LOGGER.info("calibration: %s", calibration_info)

    # --- threshold tuning on the validation split ---------------------------
    classes = list(model.classes_)
    validation_scores = model.predict_proba(validation_frame)[:, classes.index(positive_label)]
    y_validation = validation_frame["label"].to_numpy()

    threshold = training_config.threshold.default
    threshold_info: dict[str, Any] = {"tuned": False, "threshold": threshold}
    if training_config.threshold.tune:
        tuned = tune_threshold(
            y_validation,
            validation_scores,
            positive_label=positive_label,
            grid=training_config.threshold.grid(),
            metric=training_config.threshold.metric,
        )
        threshold = float(tuned["threshold"])
        threshold_info = {
            "tuned": True,
            "metric": tuned["metric"],
            "threshold": threshold,
            "best_value": tuned["best_value"],
            "curve": tuned["curve"]["thresholds"],
            "tuned_on": "validation split",
        }
        LOGGER.info(
            "threshold tuned to %.2f (%s=%.4f)", threshold, tuned["metric"], tuned["best_value"]
        )

    validation_metrics: MetricsResult = evaluate_predictions(
        y_validation,
        validation_scores,
        positive_label=positive_label,
        labels=labels,
        threshold=threshold,
    )
    calibration_diagnostics = calibration_summary(
        y_validation, validation_scores, positive_label=positive_label
    )

    # --- dataset provenance -------------------------------------------------
    dataset_manifest_path = root / args.data_dir / "dataset_manifest.json"
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.exists() else {}
    if not dataset_manifest and (root / "data" / "raw" / "download_manifest.json").exists():
        dataset_manifest = read_json(root / "data" / "raw" / "download_manifest.json")
    dataset_fingerprint = dataset_manifest.get("dataset", {}).get("fingerprint", "") or (
        dataset_manifest.get("fingerprint", "")
    )

    # --- persist ------------------------------------------------------------
    payload = {
        "model_name": best.name,
        "model_version": training_config.model_version,
        "created_at": utc_now_iso(),
        "seed": training_config.seed,
        "quick_mode": bool(args.quick),
        "dataset": {
            "name": data_config.dataset.name,
            "version": data_config.dataset.version,
            "fingerprint": dataset_fingerprint,
            "citation": data_config.dataset.citation,
            "license_note": data_config.dataset.license_note,
            "homepage": data_config.dataset.homepage,
        },
        "splits": {
            "train": {
                "rows": len(train_frame),
                "positive": int((train_frame["label"] == positive_label).sum()),
                "negative": int((train_frame["label"] != positive_label).sum()),
            },
            "validation": {
                "rows": len(validation_frame),
                "positive": int((validation_frame["label"] == positive_label).sum()),
                "negative": int((validation_frame["label"] != positive_label).sum()),
            },
        },
        "candidates": [outcome.to_dict() for outcome in outcomes],
        "validation_metrics": validation_metrics.to_dict(),
        "validation_calibration": calibration_diagnostics,
        "threshold": threshold_info,
        "calibration": calibration_info,
        "top_features": top_features_by_class(model, positive_label, top_n=25),
        "notes": notes,
        "environment": environment_summary(),
        "package_versions": package_versions(),
    }

    reports_dir = root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "training_report.json", payload)

    selection_payload = {
        "primary_metric": training_config.selection.primary_metric,
        "tie_breakers": training_config.selection.tie_breakers,
        "winner": best.name,
        "ranking": [
            outcome.name for outcome in rank_candidates(outcomes, training_config.selection)
        ],
        "criterion": (
            "Highest validation macro-F1; ties broken by positive-class F1, then p95 inference "
            "latency, then artifact size. The test split is not involved."
        ),
    }
    write_json(reports_dir / "model_selection.json", selection_payload)

    plot_model_comparison(
        payload["candidates"], path=reports_dir / "figures" / "model_comparison.png"
    )
    plot_confusion_matrices(
        validation_metrics.confusion_matrix,
        validation_metrics.confusion_matrix_normalized,
        labels=labels,
        path=reports_dir / "figures" / "validation_confusion_matrix.png",
    )
    plot_threshold_sweep(
        {"thresholds": threshold_info.get("curve", [])},
        path=reports_dir / "figures" / "validation_threshold_sweep.png",
        threshold=threshold,
    )

    (reports_dir / "training_report.md").write_text(
        _md_report(payload, selection_payload, calibration_info), "utf-8"
    )

    # --- save the inference artifact ---------------------------------------
    metadata = ModelMetadata(
        model_version=training_config.model_version,
        model_name=best.name,
        labels=labels,
        positive_label=positive_label,
        decision_threshold=threshold,
        text_field=data_config.preprocessing.text_field,
        calibration=calibration_info,
        feature_config=training_config.features.model_dump(),
        training={
            "seed": training_config.seed,
            "candidates": candidates,
            "selection": selection_payload,
            "fit_seconds": round(best.fit_seconds, 3),
            "cross_validation": best.cross_validation,
            "package_versions": package_versions(),
            "quick_mode": bool(args.quick),
        },
        dataset={
            "name": data_config.dataset.name,
            "version": data_config.dataset.version,
            "fingerprint": dataset_fingerprint,
            "citation": data_config.dataset.citation,
            "license_note": data_config.dataset.license_note,
            "homepage": data_config.dataset.homepage,
        },
        selection=selection_payload,
        candidates=[outcome.to_dict() for outcome in outcomes],
        validation_metrics=validation_metrics.to_dict(),
        top_features=payload["top_features"],
        notes=(
            "Classification only. Probabilities describe similarity to the labelled corpus, not "
            "factual accuracy. See docs/MODEL_CARD.md."
        ),
    )

    artifact_dir = root / args.artifact_dir
    save_bundle(
        model,
        metadata,
        artifact_dir / "model.joblib",
        artifact_dir / "model_metadata.json",
    )
    LOGGER.info("artifact sha256: %s", sha256_file(artifact_dir / "model.joblib")[:16])
    LOGGER.info("reports written to %s", display_path(reports_dir, root))

    LOGGER.info(
        "validation macro-F1=%.4f  accuracy=%.4f  threshold=%.2f",
        validation_metrics.macro_f1,
        validation_metrics.accuracy,
        threshold,
    )
    LOGGER.info("done. Next: python scripts/evaluate.py (touches the test split exactly once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
