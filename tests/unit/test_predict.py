"""Unit tests for the inference layer (predictions, schema, thresholding)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fake_news_detection.models.predict import (
    EmptyInputError,
    InferenceError,
    Predictor,
)
from fake_news_detection.models.registry import ModelMetadata


@pytest.fixture()
def predictor(fitted_pipeline, model_metadata: ModelMetadata) -> Predictor:
    return Predictor(fitted_pipeline, model_metadata, explain=True)


class TestPredictions:
    def test_returns_a_labelled_probability_pair(self, predictor: Predictor) -> None:
        result = predictor.predict(
            "Council approves the annual budget", "Officials said the bridge will reopen."
        )
        assert result.label in {"FAKE", "REAL"}
        assert set(result.probabilities) == {"FAKE", "REAL"}
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_probabilities_match_the_positive_score(self, predictor: Predictor) -> None:
        result = predictor.predict("Some headline", "Some article body text.")
        assert result.score == pytest.approx(result.probabilities["FAKE"])

    def test_title_only_and_body_only_are_both_accepted(self, predictor: Predictor) -> None:
        assert predictor.predict(title="A headline only").label in {"FAKE", "REAL"}
        assert predictor.predict(text="A body only, with enough text.").label in {"FAKE", "REAL"}

    def test_label_follows_the_threshold(self, predictor: Predictor) -> None:
        predictor.threshold = 0.99
        result = predictor.predict("Any headline", "Any body at all.")
        assert result.label == "REAL"
        predictor.threshold = 0.01
        assert predictor.predict("Any headline", "Any body at all.").label == "FAKE"
        predictor.threshold = 0.5

    def test_empty_input_raises(self, predictor: Predictor) -> None:
        with pytest.raises(EmptyInputError):
            predictor.predict("", "")
        with pytest.raises(EmptyInputError):
            predictor.predict("   ", "\n\t ")

    def test_whitespace_only_title_with_body_is_fine(self, predictor: Predictor) -> None:
        assert predictor.predict("   ", "A real body of text.").label in {"FAKE", "REAL"}

    def test_batch_preserves_order_and_size(self, predictor: Predictor) -> None:
        records = [
            {"title": f"headline {index}", "text": f"body text number {index}"}
            for index in range(5)
        ]
        results = predictor.predict_many(records)
        assert len(results) == 5
        assert all(result.label in {"FAKE", "REAL"} for result in results)

    def test_empty_batch_returns_empty_list(self, predictor: Predictor) -> None:
        assert predictor.predict_many([]) == []

    def test_missing_keys_in_batch_records_are_tolerated(self, predictor: Predictor) -> None:
        results = predictor.predict_many([{"title": "only a title"}])
        assert len(results) == 1

    def test_to_dict_is_json_shaped(self, predictor: Predictor) -> None:
        payload = predictor.predict("Headline", "Body text here.").to_dict()
        assert set(payload) >= {
            "prediction",
            "probabilities",
            "score",
            "decision_threshold",
            "model_version",
        }
        assert isinstance(payload["probabilities"]["FAKE"], float)


class TestAttribution:
    def test_attribution_is_returned_for_linear_models(self, predictor: Predictor) -> None:
        result = predictor.predict("Headline", "Body text here.", explain=True)
        assert result.attribution is not None
        assert result.attribution["supported"] is True

    def test_attribution_can_be_switched_off(self, predictor: Predictor) -> None:
        assert predictor.predict("Headline", "Body text.", explain=False).attribution is None

    def test_attribution_supported_flag(self, predictor: Predictor) -> None:
        assert predictor.attribution_supported() is True


class TestConstructionGuards:
    def test_invalid_threshold_rejected(
        self, fitted_pipeline, model_metadata: ModelMetadata
    ) -> None:
        with pytest.raises(InferenceError, match="threshold"):
            Predictor(fitted_pipeline, model_metadata, decision_threshold=1.4)

    def test_mismatched_labels_are_rejected(self, fitted_pipeline) -> None:
        metadata = ModelMetadata(
            model_version="x", model_name="m", labels=["HAM", "SPAM"], positive_label="SPAM"
        )
        with pytest.raises(InferenceError, match="inconsistent"):
            Predictor(fitted_pipeline, metadata)

    def test_from_artifacts_loads_and_warms_up(self, artifact_paths: tuple[Path, Path]) -> None:
        artifact, metadata = artifact_paths
        predictor = Predictor.from_artifacts(artifact, metadata)
        assert predictor.metadata.model_version == "0.0.1-test"
        assert predictor.predict("t", "b").label in {"FAKE", "REAL"}

    def test_metadata_is_exposed(self, predictor: Predictor) -> None:
        assert predictor.metadata.model_name == "logistic_regression"
        assert predictor.labels == ["FAKE", "REAL"]
