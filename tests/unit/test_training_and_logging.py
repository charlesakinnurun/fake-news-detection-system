"""Unit tests for training orchestration, model ranking and logging utilities."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fake_news_detection.config import SelectionConfig
from fake_news_detection.evaluation.metrics import MetricsResult
from fake_news_detection.models.train import (
    LOWER_IS_BETTER,
    CandidateOutcome,
    rank_candidates,
    select_best,
    stratified_subsample,
)
from fake_news_detection.monitoring.logging import (
    JsonFormatter,
    LatencyTracker,
    configure_logging,
    new_request_id,
    set_request_id,
)


def _metrics(macro_f1: float) -> MetricsResult:
    return MetricsResult(
        n_samples=100,
        n_positive=50,
        n_negative=50,
        positive_label="FAKE",
        threshold=0.5,
        accuracy=macro_f1,
        balanced_accuracy=macro_f1,
        precision=macro_f1,
        recall=macro_f1,
        f1=macro_f1,
        macro_f1=macro_f1,
        weighted_f1=macro_f1,
        roc_auc=macro_f1,
        average_precision=macro_f1,
        brier_score=0.1,
        log_loss=0.2,
        cohen_kappa=macro_f1,
        matthews_corrcoef=macro_f1,
        confusion_matrix=[[50, 0], [0, 50]],
        confusion_matrix_normalized=[[1.0, 0.0], [0.0, 1.0]],
        per_class={},
        labels_order=["FAKE", "REAL"],
    )


def _outcome(name: str, macro_f1: float, latency_p95: float, size: float) -> CandidateOutcome:
    return CandidateOutcome(
        name=name,
        pipeline=None,
        validation_metrics=_metrics(macro_f1),
        fit_seconds=1.0,
        latency={"latency_ms_p95": latency_p95, "latency_ms_p50": latency_p95 / 2},
        artifact_size_mb=size,
    )


class TestRanking:
    def test_higher_macro_f1_wins(self) -> None:
        selection = SelectionConfig()
        outcomes = [_outcome("a", 0.8, 10.0, 1.0), _outcome("b", 0.9, 10.0, 1.0)]
        assert select_best(outcomes, selection).name == "b"

    def test_latency_breaks_ties(self) -> None:
        selection = SelectionConfig()
        outcomes = [_outcome("slow", 0.9, 50.0, 1.0), _outcome("fast", 0.9, 5.0, 1.0)]
        assert select_best(outcomes, selection).name == "fast"

    def test_size_breaks_ties_when_latency_is_equal(self) -> None:
        selection = SelectionConfig()
        outcomes = [_outcome("big", 0.9, 5.0, 20.0), _outcome("small", 0.9, 5.0, 2.0)]
        assert select_best(outcomes, selection).name == "small"

    def test_ranking_is_a_total_order(self) -> None:
        rank = rank_candidates(
            [_outcome("a", 0.5, 1.0, 1.0), _outcome("b", 0.9, 1.0, 1.0)], SelectionConfig()
        )
        assert [outcome.name for outcome in rank] == ["b", "a"]

    def test_lower_is_better_set_is_consistent(self) -> None:
        assert {"latency_ms_p95", "artifact_size_mb", "fit_seconds"} <= LOWER_IS_BETTER

    def test_empty_outcomes_raise(self) -> None:
        with pytest.raises(ValueError):
            select_best([], SelectionConfig())

    def test_missing_latency_does_not_crash(self) -> None:
        outcome = _outcome("a", 0.9, 1.0, 1.0)
        outcome.latency = {}
        assert select_best([outcome], SelectionConfig()).name == "a"


class TestStratifiedSubsample:
    def test_preserves_the_label_mix(self, synthetic_frame) -> None:
        y = synthetic_frame["label"].to_numpy()
        subset, y_subset = stratified_subsample(synthetic_frame, y, 40, 1)
        assert len(subset) <= 40
        shares = np.unique(y_subset, return_counts=True)[1] / len(y_subset)
        assert np.allclose(shares, 0.5, atol=0.1)

    def test_is_deterministic(self, synthetic_frame) -> None:
        y = synthetic_frame["label"].to_numpy()
        first, _ = stratified_subsample(synthetic_frame, y, 40, 3)
        second, _ = stratified_subsample(synthetic_frame, y, 40, 3)
        assert first["row_id"].tolist() == second["row_id"].tolist()

    def test_no_subsample_when_below_limit(self, synthetic_frame) -> None:
        y = synthetic_frame["label"].to_numpy()
        subset, _ = stratified_subsample(synthetic_frame, y, len(synthetic_frame) + 10, 1)
        assert len(subset) == len(synthetic_frame)
        assert set(subset["label"]) == {"FAKE", "REAL"}


class TestJsonLogging:
    def test_formatter_emits_json_with_extras(self) -> None:
        import json

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.http_path = "/predict"
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["http_path"] == "/predict"

    def test_request_id_is_included(self) -> None:
        import json

        set_request_id("req_abc")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hi",
            args=(),
            exc_info=None,
        )
        assert json.loads(JsonFormatter().format(record))["request_id"] == "req_abc"
        set_request_id(None)

    def test_exception_information_is_included(self) -> None:
        import json

        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        assert "ValueError" in json.loads(JsonFormatter().format(record))["exception"]

    def test_configure_logging_is_idempotent(self) -> None:
        configure_logging("DEBUG", json_output=True)
        configure_logging("INFO", json_output=False)
        assert len(logging.getLogger().handlers) == 1

    def test_request_ids_are_unique(self) -> None:
        assert new_request_id() != new_request_id()


class TestLatencyTracker:
    def test_snapshot_reports_percentiles(self) -> None:
        tracker = LatencyTracker()
        for value in (1.0, 2.0, 3.0, 4.0):
            tracker.observe("/predict", value)
        snapshot = tracker.snapshot()
        entry = snapshot["endpoints"]["/predict"]
        assert entry["requests"] == 4
        assert entry["p50_ms"] == pytest.approx(2.5)
        assert entry["max_ms"] == pytest.approx(4.0)

    def test_errors_are_counted(self) -> None:
        tracker = LatencyTracker()
        tracker.observe("/predict", 5.0, error=True)
        assert tracker.snapshot()["endpoints"]["/predict"]["errors"] == 1

    def test_window_bounds_memory(self) -> None:
        tracker = LatencyTracker(window=10)
        for value in range(100):
            tracker.observe("/x", float(value))
        assert tracker.snapshot()["endpoints"]["/x"]["window"] == 10

    def test_empty_snapshot(self) -> None:
        assert LatencyTracker().snapshot()["endpoints"] == {}
