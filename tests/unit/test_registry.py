"""Unit tests for the model registry (saving, loading, checksum verification)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fake_news_detection.models.registry import (
    ArtifactChecksumError,
    ArtifactNotFoundError,
    ModelMetadata,
    load_bundle,
    save_bundle,
)


class TestMetadataValidation:
    def test_round_trip(self) -> None:
        metadata = ModelMetadata(
            model_version="1.2.3",
            model_name="logistic_regression",
            labels=["FAKE", "REAL"],
            positive_label="FAKE",
        )
        assert ModelMetadata.model_validate(metadata.model_dump()) == metadata

    def test_defaults_are_sane(self) -> None:
        metadata = ModelMetadata(
            model_version="1.0.0",
            model_name="dummy",
            labels=["FAKE", "REAL"],
            positive_label="FAKE",
        )
        assert metadata.decision_threshold == 0.5
        assert metadata.created_at
        assert metadata.schema_version == 1


class TestSaveAndLoad:
    def test_save_records_checksum_and_size(
        self, fitted_pipeline, model_metadata: ModelMetadata, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "model.joblib"
        metadata_path = tmp_path / "model_metadata.json"
        saved = save_bundle(fitted_pipeline, model_metadata, artifact, metadata_path)
        assert artifact.exists() and metadata_path.exists()
        assert saved.artifact["sha256"]
        assert saved.artifact["bytes"] == artifact.stat().st_size
        assert "B" in saved.artifact["human_size"] or "KiB" in saved.artifact["human_size"]

    def test_metadata_file_is_valid_json(
        self, fitted_pipeline, model_metadata: ModelMetadata, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "model.joblib"
        metadata_path = tmp_path / "model_metadata.json"
        save_bundle(fitted_pipeline, model_metadata, artifact, metadata_path)
        payload = json.loads(metadata_path.read_text("utf-8"))
        assert payload["model_name"] == "logistic_regression"
        assert payload["artifact"]["sha256"]

    def test_load_returns_a_usable_pipeline(
        self, fitted_pipeline, model_metadata: ModelMetadata, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "model.joblib"
        metadata_path = tmp_path / "model_metadata.json"
        save_bundle(fitted_pipeline, model_metadata, artifact, metadata_path)
        bundle = load_bundle(artifact, metadata_path)
        assert bundle.metadata.model_version == "0.0.1-test"
        prediction = bundle.pipeline.predict([{"title": "hello", "text": "world"}])
        assert prediction[0] in {"FAKE", "REAL"}

    def test_missing_artifact_raises_actionable_error(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactNotFoundError, match="train"):
            load_bundle(tmp_path / "nope.joblib", tmp_path / "nope.json")

    def test_missing_metadata_raises(self, fitted_pipeline, tmp_path: Path) -> None:
        artifact = tmp_path / "model.joblib"
        save_bundle(
            fitted_pipeline,
            ModelMetadata(
                model_version="1", model_name="m", labels=["FAKE", "REAL"], positive_label="FAKE"
            ),
            artifact,
            tmp_path / "meta.json",
        )
        (tmp_path / "meta.json").unlink()
        with pytest.raises(ArtifactNotFoundError, match="metadata"):
            load_bundle(artifact, tmp_path / "meta.json")

    def test_checksum_mismatch_is_detected(
        self, fitted_pipeline, model_metadata: ModelMetadata, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "model.joblib"
        metadata_path = tmp_path / "model_metadata.json"
        save_bundle(fitted_pipeline, model_metadata, artifact, metadata_path)
        # Simulate corruption/tampering.
        payload = json.loads(metadata_path.read_text("utf-8"))
        payload["artifact"]["sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(payload), "utf-8")
        with pytest.raises(ArtifactChecksumError, match="checksum mismatch"):
            load_bundle(artifact, metadata_path)

    def test_checksum_verification_can_be_disabled(
        self, fitted_pipeline, model_metadata: ModelMetadata, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "model.joblib"
        metadata_path = tmp_path / "model_metadata.json"
        save_bundle(fitted_pipeline, model_metadata, artifact, metadata_path)
        payload = json.loads(metadata_path.read_text("utf-8"))
        payload["artifact"]["sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(payload), "utf-8")
        bundle = load_bundle(artifact, metadata_path, verify_checksum=False)
        assert bundle.pipeline is not None

    def test_tampered_artifact_is_detected(
        self, fitted_pipeline, model_metadata: ModelMetadata, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "model.joblib"
        metadata_path = tmp_path / "model_metadata.json"
        save_bundle(fitted_pipeline, model_metadata, artifact, metadata_path)
        with artifact.open("ab") as handle:
            handle.write(b"tampered")
        with pytest.raises(ArtifactChecksumError):
            load_bundle(artifact, metadata_path)
