"""End-to-end pipeline test on synthetic data (no dataset download required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fake_news_detection.data.splitting import make_splits
from fake_news_detection.evaluation.calibration import calibration_summary, fit_calibrator
from fake_news_detection.evaluation.metrics import evaluate_predictions, tune_threshold
from fake_news_detection.models.predict import Predictor
from fake_news_detection.models.registry import ModelMetadata, load_bundle, save_bundle
from fake_news_detection.models.train import select_best, train_candidates

pytestmark = pytest.mark.integration


def test_pipeline_runs_from_raw_rows_to_a_served_prediction(
    synthetic_frame,
    data_config,
    training_config,
    tmp_path: Path,
) -> None:
    from fake_news_detection.data.validation import validate_and_clean

    # --- prepare ------------------------------------------------------------
    clean, report = validate_and_clean(synthetic_frame, data_config)
    assert report.rows_out > 0
    assert report.rows_out == len(clean)

    splits = make_splits(clean, data_config)
    assert splits.leakage_audit["passed"] is True
    train_frame, validation_frame, test_frame = (
        splits.train,
        splits.validation,
        splits.test,
    )
    assert not train_frame.empty and not validation_frame.empty and not test_frame.empty

    # --- train --------------------------------------------------------------
    outcomes = train_candidates(
        train_frame,
        validation_frame,
        training_config,
        data_config.preprocessing,
        candidates=["dummy", "logistic_regression"],
        labels=["FAKE", "REAL"],
        positive_label="FAKE",
        run_cross_validation=False,
    )
    best = select_best(outcomes, training_config.selection)
    assert best.name == "logistic_regression", "the learned model must beat the prior baseline"

    # --- calibrate + threshold on validation --------------------------------
    model, info = fit_calibrator(
        best.pipeline, validation_frame, validation_frame["label"].to_numpy(), method="sigmoid"
    )
    assert info["applied"] is True

    classes = list(model.classes_)
    validation_scores = model.predict_proba(validation_frame)[:, classes.index("FAKE")]
    tuned = tune_threshold(
        validation_frame["label"].to_numpy(),
        validation_scores,
        positive_label="FAKE",
        grid=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    threshold = float(tuned["threshold"])

    # --- persist ------------------------------------------------------------
    metadata = ModelMetadata(
        model_version="0.0.1-e2e",
        model_name=best.name,
        labels=["FAKE", "REAL"],
        positive_label="FAKE",
        decision_threshold=threshold,
        validation_metrics=evaluate_predictions(
            validation_frame["label"].to_numpy(),
            validation_scores,
            positive_label="FAKE",
            threshold=threshold,
        ).to_dict(),
    )
    artifact = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    save_bundle(model, metadata, artifact, metadata_path)

    # --- load and serve -----------------------------------------------------
    bundle = load_bundle(artifact, metadata_path)
    predictor = Predictor.from_bundle(bundle)

    assert predictor.threshold == pytest.approx(threshold)
    assert predictor.attribution_supported() is True

    fake_like = synthetic_frame[synthetic_frame["label"] == "FAKE"].iloc[0]
    real_like = synthetic_frame[synthetic_frame["label"] == "REAL"].iloc[0]
    fake_prediction = predictor.predict(fake_like["title"], fake_like["text"])
    real_prediction = predictor.predict(real_like["title"], real_like["text"])
    assert fake_prediction.label == "FAKE"
    assert real_prediction.label == "REAL"
    assert fake_prediction.score > real_prediction.score

    # --- test-set evaluation happens exactly once ---------------------------
    test_scores = model.predict_proba(test_frame)[:, classes.index("FAKE")]
    test_metrics = evaluate_predictions(
        test_frame["label"].to_numpy(), test_scores, positive_label="FAKE", threshold=threshold
    )
    assert test_metrics.n_samples == len(test_frame)
    assert test_metrics.macro_f1 > 0.75  # synthetic classes are linearly separable

    diagnostics = calibration_summary(
        test_frame["label"].to_numpy(), test_scores, positive_label="FAKE"
    )
    assert 0.0 <= diagnostics["expected_calibration_error"] <= 1.0

    # --- metadata can be updated with the test metrics ----------------------
    bundle.metadata.test_metrics = test_metrics.to_dict()
    save_bundle(model, bundle.metadata, artifact, metadata_path)
    reloaded = load_bundle(artifact, metadata_path)
    assert reloaded.metadata.test_metrics is not None
    assert reloaded.metadata.test_metrics["n_samples"] == len(test_frame)


def test_splits_are_reproducible_end_to_end(synthetic_frame, data_config) -> None:
    from fake_news_detection.data.validation import validate_and_clean

    clean, _ = validate_and_clean(synthetic_frame, data_config)
    first = make_splits(clean, data_config)
    second = make_splits(clean, data_config)
    assert first.train["row_id"].tolist() == second.train["row_id"].tolist()
    assert first.test["row_id"].tolist() == second.test["row_id"].tolist()


def test_leakage_ablation_shows_the_grouped_split_is_stricter(synthetic_frame, data_config) -> None:
    from fake_news_detection.data.validation import validate_and_clean

    clean, _ = validate_and_clean(synthetic_frame, data_config)
    grouped = make_splits(clean, data_config, strategy="grouped_stratified")
    plain = make_splits(clean, data_config, strategy="stratified")
    assert grouped.leakage_audit["checks"]["duplicate_cluster_overlap_total"] == 0
    assert plain.leakage_audit["checks"]["duplicate_cluster_overlap_total"] > 0
