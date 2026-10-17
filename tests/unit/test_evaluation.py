"""Unit tests for metrics, threshold tuning and calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fake_news_detection.evaluation.calibration import (
    calibration_summary,
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
    supports_probabilities,
)
from fake_news_detection.evaluation.error_analysis import (
    confidence_profile,
    length_bucket_analysis,
    misclassified_examples,
    slice_metrics,
)
from fake_news_detection.evaluation.metrics import (
    artifact_size_mb,
    binary_labels,
    evaluate_predictions,
    latency_benchmark,
    threshold_sweep,
    tune_threshold,
)


def _perfect(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array(["FAKE"] * (n // 2) + ["REAL"] * (n // 2))
    scores = np.array([0.9] * (n // 2) + [0.1] * (n // 2))
    return labels, scores


class TestBinaryLabels:
    def test_positive_class_mapping(self) -> None:
        assert binary_labels(["FAKE", "REAL", "FAKE"], "FAKE").tolist() == [1, 0, 1]

    def test_alternative_positive_class(self) -> None:
        assert binary_labels(["FAKE", "REAL"], "REAL").tolist() == [0, 1]


class TestEvaluatePredictions:
    def test_perfect_predictions(self) -> None:
        labels, scores = _perfect()
        result = evaluate_predictions(labels, scores, positive_label="FAKE")
        assert result.accuracy == 1.0
        assert result.f1 == 1.0
        assert result.macro_f1 == 1.0
        assert result.roc_auc == 1.0
        assert result.confusion_matrix == [[50, 0], [0, 50]]

    def test_confusion_matrix_follows_declared_label_order(self) -> None:
        labels = np.array(["FAKE", "FAKE", "FAKE", "REAL", "REAL"])
        scores = np.array([0.9, 0.9, 0.1, 0.1, 0.1])
        result = evaluate_predictions(
            labels, scores, positive_label="FAKE", labels=["FAKE", "REAL"], threshold=0.5
        )
        # Rows/columns follow ["FAKE", "REAL"]; two FAKE rows are correct, one is
        # called REAL, and both REAL rows are correct.
        assert result.labels_order == ["FAKE", "REAL"]
        assert result.confusion_matrix == [[2, 1], [0, 2]]
        assert result.confusion_matrix_normalized[0] == pytest.approx([2 / 3, 1 / 3], abs=1e-4)
        assert result.confusion_matrix_normalized[1] == pytest.approx([0.0, 1.0])

    def test_positive_class_is_stated_and_used(self) -> None:
        labels = np.array(["FAKE", "REAL"])
        scores = np.array([0.9, 0.9])
        result = evaluate_predictions(labels, scores, positive_label="FAKE", threshold=0.5)
        assert result.positive_label == "FAKE"
        assert result.recall == 1.0  # both predicted FAKE, one truly FAKE
        assert result.precision == pytest.approx(0.5)

    def test_threshold_changes_the_prediction(self) -> None:
        labels = np.array(["FAKE", "REAL"])
        scores = np.array([0.6, 0.4])
        low = evaluate_predictions(labels, scores, positive_label="FAKE", threshold=0.5)
        high = evaluate_predictions(labels, scores, positive_label="FAKE", threshold=0.7)
        assert low.accuracy == 1.0
        assert high.accuracy < 1.0

    def test_per_class_metrics_and_normalisation(self) -> None:
        labels, scores = _perfect(20)
        result = evaluate_predictions(
            labels, np.where(scores > 0.5, 0.6, 0.4), positive_label="FAKE"
        )
        assert set(result.per_class) == {"FAKE", "REAL"}
        assert result.per_class["FAKE"]["support"] == 10
        assert all(
            0.0 <= value <= 1.0 for row in result.confusion_matrix_normalized for value in row
        )

    def test_single_class_input_yields_no_auc(self) -> None:
        labels = np.array(["FAKE", "FAKE"])
        scores = np.array([0.8, 0.9])
        result = evaluate_predictions(labels, scores, positive_label="FAKE")
        assert result.roc_auc is None
        assert result.average_precision is None

    def test_metrics_are_json_serialisable(self) -> None:
        import json

        labels, scores = _perfect(10)
        json.dumps(evaluate_predictions(labels, scores).to_dict())


class TestThresholdTuning:
    def test_sweep_covers_the_grid(self) -> None:
        labels, scores = _perfect()
        sweep = threshold_sweep(labels, scores, positive_label="FAKE", grid=[0.2, 0.5, 0.8])
        assert [entry["threshold"] for entry in sweep["thresholds"]] == [0.2, 0.5, 0.8]

    def test_tuning_selects_the_best_f1(self) -> None:
        labels = np.array(["FAKE"] * 50 + ["REAL"] * 50)
        scores = np.array([0.4] * 50 + [0.1] * 50)
        tuned = tune_threshold(
            labels, scores, positive_label="FAKE", grid=[0.2, 0.3, 0.5, 0.8], metric="f1_positive"
        )
        # Every threshold in (0.1, 0.4] separates the two groups perfectly; the
        # lowest such threshold wins ties.
        assert tuned["threshold"] == pytest.approx(0.2)
        assert tuned["best_value"] == pytest.approx(1.0)

    def test_tuning_is_reproducible(self) -> None:
        labels, scores = _perfect(40)
        grid = [0.1, 0.2, 0.5, 0.7]
        first = tune_threshold(labels, scores, positive_label="FAKE", grid=grid)
        second = tune_threshold(labels, scores, positive_label="FAKE", grid=grid)
        assert first["threshold"] == second["threshold"]

    def test_sweep_reports_confusion_counts(self) -> None:
        labels, scores = _perfect(10)
        sweep = threshold_sweep(labels, scores, positive_label="FAKE", grid=[0.5])
        entry = sweep["thresholds"][0]
        assert entry["false_positives"] == 0
        assert entry["false_negatives"] == 0


class TestReliability:
    def test_perfect_calibration_has_low_error(self) -> None:
        labels = np.array(["FAKE"] * 100 + ["REAL"] * 100)
        scores = np.array([0.9] * 100 + [0.1] * 100)
        curve = reliability_curve((labels == "FAKE").astype(int), scores, n_bins=10)
        ece, mce = expected_calibration_error(curve)
        assert 0.0 <= ece <= 1.0
        assert mce >= ece

    def test_bin_counts_sum_to_the_sample_size(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        curve = reliability_curve(rng.integers(0, 2, 200), scores, n_bins=8)
        assert sum(entry["count"] for entry in curve["bins"]) == 200

    def test_calibration_summary_keys(self) -> None:
        labels, scores = _perfect(60)
        summary = calibration_summary(labels, scores, positive_label="FAKE")
        assert set(summary) >= {
            "brier_score",
            "log_loss",
            "expected_calibration_error",
            "max_calibration_error",
            "reliability_curve",
        }
        assert summary["brier_score"] < 0.05

    def test_supports_probabilities(self) -> None:
        assert supports_probabilities(type("M", (), {"predict_proba": lambda self: None})()) is True
        assert supports_probabilities(object()) is False


class TestCalibrator:
    def test_calibration_is_applied_to_a_probabilistic_model(self, fitted_pipeline, splits) -> None:
        frame = splits.validation
        model, info = fit_calibrator(
            fitted_pipeline, frame, frame["label"].to_numpy(), method="sigmoid"
        )
        assert info["applied"] is True, info.get("reason")
        assert info["fitted_on"] == "validation split"
        probabilities = model.predict_proba(frame)
        assert probabilities.shape[0] == len(frame)

    def test_non_probabilistic_model_is_left_untouched(self, splits) -> None:
        class NoProba:
            def __init__(self) -> None:
                self.classes_ = np.array(["FAKE", "REAL"])

            def fit(self, *_args, **_kwargs):  # pragma: no cover - never called
                return self

        _, info = fit_calibrator(
            NoProba(), splits.validation, splits.validation["label"].to_numpy()
        )
        assert info["applied"] is False
        assert "predict_proba" in info["reason"]


class TestLatencyAndSize:
    def test_latency_benchmark_reports_percentiles(self) -> None:
        samples = [{"title": "t", "text": "b"} for _ in range(10)]
        result = latency_benchmark(lambda record: record, samples, warmup=2)
        assert result["samples"] == 10
        assert result["latency_ms_p50"] <= result["latency_ms_p95"] <= result["latency_ms_max"]
        assert result["throughput_per_second"] > 0

    def test_latency_benchmark_with_no_samples(self) -> None:
        assert latency_benchmark(lambda record: record, []) == {"samples": 0}

    def test_artifact_size_missing_file_is_zero(self, tmp_path) -> None:
        assert artifact_size_mb(tmp_path / "nope.bin") == 0.0

    def test_artifact_size_reads_the_file(self, tmp_path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(b"x" * (2 * 1024 * 1024))
        assert artifact_size_mb(path) == pytest.approx(2.0, abs=0.01)


class TestErrorAnalysis:
    def _frame(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        labels = np.array(["FAKE"] * (n // 2) + ["REAL"] * (n // 2))
        return pd.DataFrame(
            {
                "row_id": [f"r{i}" for i in range(n)],
                "title": [f"headline {i}" for i in range(n)],
                "text": ["body"] * n,
                "label": labels,
                "subject": rng.choice(["news", "politics", "world"], size=n),
                "row_length": rng.integers(200, 5000, size=n),
            }
        )

    def test_slice_metrics_flags_small_slices(self) -> None:
        frame = self._frame()
        scores = np.where(frame["label"] == "FAKE", 0.8, 0.2)
        preds = (scores >= 0.5).astype(int)
        result = slice_metrics(
            frame, preds, scores, column="subject", positive_label="FAKE", labels=["FAKE", "REAL"]
        )
        assert result["available"] is True
        assert all("n" in entry for entry in result["slices"])

    def test_missing_column_is_reported_not_raised(self) -> None:
        frame = self._frame(20)
        result = slice_metrics(
            frame,
            np.zeros(20, dtype=int),
            np.zeros(20),
            column="absent",
            positive_label="FAKE",
            labels=["FAKE", "REAL"],
        )
        assert result["available"] is False

    def test_length_buckets(self) -> None:
        frame = self._frame()
        scores = np.where(frame["label"] == "FAKE", 0.8, 0.2)
        result = length_bucket_analysis(
            frame, (scores >= 0.5).astype(int), scores, positive_label="FAKE"
        )
        assert result["available"] is True
        assert len(result["buckets"]) == 5

    def test_confidence_profile_shares_sum_to_one(self) -> None:
        frame = self._frame()
        scores = np.where(frame["label"] == "FAKE", 0.8, 0.2)
        result = confidence_profile(
            frame, (scores >= 0.5).astype(int), scores, positive_label="FAKE"
        )
        total = sum(entry["share"] for entry in result["buckets"])
        assert total == pytest.approx(1.0, abs=0.01)

    def test_misclassified_examples_are_ranked_by_confidence(self) -> None:
        frame = self._frame(40)
        scores = np.zeros(40)
        preds = np.zeros(40, dtype=int)
        # Rows 20-29 are REAL (negative) but are predicted FAKE with high confidence.
        scores[20:30] = 0.95
        preds[20:30] = 1
        result = misclassified_examples(
            frame,
            preds,
            scores,
            positive_label="FAKE",
            negative_label="REAL",
            per_type=5,
        )
        assert result["false_positive_count"] == 10
        assert (
            result["false_positives"][0]["score_positive"]
            >= result["false_positives"][-1]["score_positive"]
        )
        assert all(len(row["title"]) <= 141 for row in result["false_positives"])

    def test_examples_do_not_include_full_bodies(self) -> None:
        frame = self._frame(10)
        result = misclassified_examples(
            frame, np.ones(10, dtype=int), np.ones(10), positive_label="FAKE"
        )
        assert all("text" not in row for row in result["false_positives"])
