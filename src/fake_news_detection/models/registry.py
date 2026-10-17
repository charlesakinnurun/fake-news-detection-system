"""Model artifact registry: versioned, checksummed saving and loading.

Security note
-------------
Artifacts are serialised with :mod:`joblib`, which uses pickle under the hood and
can therefore execute arbitrary code on load. Treat the artifact directory as
trusted, deployment-controlled content:

* never load an artifact supplied by a user or fetched from an untrusted URL;
* the service loads the model exactly once at start-up and never downloads one
  during a request;
* :func:`load_bundle` verifies the SHA-256 recorded in the metadata file, which
  detects accidental corruption or a swapped artifact (it is *not* a signature -
  a tamperer who can replace the model can usually replace the metadata too;
  use signed storage if that threat is in scope).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, ConfigDict, Field

from fake_news_detection.utils import (
    ensure_dir,
    human_bytes,
    read_json,
    sha256_file,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ArtifactChecksumError",
    "ArtifactError",
    "ArtifactNotFoundError",
    "ModelBundle",
    "ModelMetadata",
    "load_bundle",
    "save_bundle",
]


class ArtifactError(RuntimeError):
    """Base class for model-artifact problems."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when the model artifact or metadata file is missing."""


class ArtifactChecksumError(ArtifactError):
    """Raised when the artifact does not match the checksum in its metadata."""


class ModelMetadata(BaseModel):
    """Everything needed to interpret and audit a saved model."""

    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    model_version: str
    created_at: str = Field(default_factory=utc_now_iso)
    model_name: str
    labels: list[str]
    positive_label: str
    decision_threshold: float = 0.5
    text_field: str = "title_body"
    calibration: dict[str, Any] | None = None
    feature_config: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    selection: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] | None = None
    top_features: dict[str, Any] | None = None
    artifact: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


@dataclass(slots=True)
class ModelBundle:
    """A loaded pipeline together with its metadata."""

    pipeline: Any
    metadata: ModelMetadata
    artifact_path: Path


def save_bundle(
    pipeline: Any,
    metadata: ModelMetadata,
    artifact_path: str | Path,
    metadata_path: str | Path,
) -> ModelMetadata:
    """Persist ``pipeline`` and its metadata, recording size and checksum."""
    artifact_path = Path(artifact_path)
    metadata_path = Path(metadata_path)
    ensure_dir(artifact_path.parent)

    joblib.dump(pipeline, artifact_path, compress=3)
    checksum = sha256_file(artifact_path)
    size = artifact_path.stat().st_size
    metadata.artifact = {
        "path": artifact_path.name,
        "sha256": checksum,
        "bytes": int(size),
        "human_size": human_bytes(size),
    }
    if not metadata.created_at:
        metadata.created_at = utc_now_iso()
    write_json(metadata_path, metadata.model_dump())
    LOGGER.info(
        "saved model %s v%s (%s, sha256=%s)",
        metadata.model_name,
        metadata.model_version,
        human_bytes(size),
        checksum[:12],
    )
    return metadata


def load_bundle(
    artifact_path: str | Path,
    metadata_path: str | Path,
    *,
    verify_checksum: bool = True,
) -> ModelBundle:
    """Load a pipeline and its metadata, optionally verifying the checksum."""
    artifact_path = Path(artifact_path)
    metadata_path = Path(metadata_path)

    if not artifact_path.exists():
        raise ArtifactNotFoundError(
            f"model artifact not found: {artifact_path}\n"
            "Train a model first (`python scripts/train.py` or `make train`)."
        )
    if not metadata_path.exists():
        raise ArtifactNotFoundError(
            f"model metadata not found: {metadata_path}\n"
            "The metadata file is written by scripts/train.py alongside the artifact."
        )

    metadata = ModelMetadata.model_validate(read_json(metadata_path))
    expected = (metadata.artifact or {}).get("sha256")
    if verify_checksum and expected:
        actual = sha256_file(artifact_path)
        if actual != expected:
            raise ArtifactChecksumError(
                f"artifact checksum mismatch for {artifact_path.name}: metadata records "
                f"{expected[:12]}… but the file hashes to {actual[:12]}…"
            )

    pipeline = joblib.load(artifact_path)
    LOGGER.info(
        "loaded model %s v%s (threshold=%.2f)",
        metadata.model_name,
        metadata.model_version,
        metadata.decision_threshold,
    )
    return ModelBundle(pipeline=pipeline, metadata=metadata, artifact_path=artifact_path)
