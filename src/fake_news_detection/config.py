"""Typed, centralised configuration for every stage of the pipeline.

All tunable values live here instead of being scattered as magic constants
through the codebase. YAML files under ``configs/`` provide the defaults; the
API additionally reads a small number of environment variables so that the
same container image can be deployed in different environments.

Precedence for the service settings is: constructor argument > environment
variable > ``.env`` file > default.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_DIR_ENV = "FND_CONFIG_DIR"


def repo_root() -> Path:
    """Return the repository root (two levels above this source file)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Resolve ``path`` against ``base`` (default: repository root)."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base or repo_root()) / candidate


@lru_cache(maxsize=1)
def get_service_settings() -> ServiceSettings:
    """Return the process-wide service settings (cached).

    Tests can call ``get_service_settings.cache_clear()`` after changing
    environment variables.
    """
    return ServiceSettings()


class StrictModel(BaseModel):
    """Base model that rejects unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


class DataSourceConfig(StrictModel):
    """One raw CSV file plus the label it contributes."""

    path: str = Field(description="Path to the CSV, relative to the repository root.")
    label: str = Field(description="Canonical label assigned to every row of the file.")
    description: str = ""


class ColumnMapping(StrictModel):
    """Maps dataset-specific column names onto the internal representation."""

    title: str = "title"
    text: str = "text"
    subject: str | None = "subject"
    date: str | None = "date"
    label: str | None = Field(
        default=None,
        description="Column holding a label; when unset the source-level label is used.",
    )


class DatasetConfig(StrictModel):
    """Identity and provenance of the corpus."""

    name: str
    version: str = "1.0.0"
    citation: str = ""
    license_note: str = ""
    homepage: str = ""
    sources: list[DataSourceConfig] = Field(min_length=1)
    columns: ColumnMapping = ColumnMapping()
    encoding: str = "utf-8"
    labels: list[str] = Field(default_factory=lambda: ["FAKE", "REAL"], min_length=2)
    positive_label: str = "FAKE"
    label_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional normalisation map for label values found in the data, e.g. "
            "{'TRUE': 'REAL', 'FALSE': 'FAKE'}. Keys are matched case-insensitively."
        ),
    )
    date_formats: list[str] = Field(
        default_factory=lambda: ["%B %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%B-%y"]
    )

    @model_validator(mode="after")
    def _check_positive_label(self) -> DatasetConfig:
        if self.positive_label not in self.labels:
            raise ValueError(
                f"positive_label={self.positive_label!r} is not one of labels={self.labels}"
            )
        return self


class ValidationConfig(StrictModel):
    """Thresholds used by :mod:`fake_news_detection.data.validation`."""

    min_title_chars: int = 3
    min_text_chars: int = 40
    min_total_chars: int = 60
    max_title_chars: int = 1000
    max_text_chars: int = 60000
    drop_empty_bodies: bool = True
    drop_exact_duplicates: bool = True
    conflicting_label_policy: Literal["drop", "majority"] = "drop"
    majority_min_agreement: float = 0.75
    max_reported_examples: int = 20


class PreprocessingConfig(StrictModel):
    """Deterministic text normalisation applied before feature extraction."""

    text_field: Literal["title_body", "title_only", "body_only"] = "title_body"
    separator: str = " \n "
    lowercase: bool = False
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    remove_control_chars: bool = True
    strip_urls: bool = False
    strip_html: bool = False
    replace_numbers: bool = False
    min_token_length: int = 0


class NearDuplicateConfig(StrictModel):
    """MinHash + LSH parameters for near-duplicate clustering."""

    enabled: bool = True
    shingle_size: int = 4
    num_hashes: int = 64
    num_bands: int = 16
    jaccard_threshold: float = 0.8
    max_shingles_per_doc: int = 300

    @model_validator(mode="after")
    def _check_bands(self) -> NearDuplicateConfig:
        if self.num_hashes % self.num_bands != 0:
            raise ValueError("num_hashes must be divisible by num_bands")
        if not 0.0 < self.jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be in (0, 1]")
        return self


class TemporalSplitConfig(StrictModel):
    """Date-based split used for the temporal generalisation experiment."""

    enabled: bool = False
    date_column: str = "date"
    val_size: float = 0.15
    test_size: float = 0.15


class SplittingConfig(StrictModel):
    """Leakage-aware split strategy."""

    strategy: Literal["grouped_stratified", "stratified", "temporal"] = "grouped_stratified"
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    near_duplicate: NearDuplicateConfig = Field(default_factory=NearDuplicateConfig)
    temporal: TemporalSplitConfig = Field(default_factory=TemporalSplitConfig)

    @model_validator(mode="after")
    def _check_sizes(self) -> SplittingConfig:
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split sizes must sum to 1.0, got {total}")
        if min(self.train_size, self.val_size, self.test_size) <= 0:
            raise ValueError("all split sizes must be positive")
        return self


class DataConfig(StrictModel):
    """Top-level configuration consumed by ``scripts/prepare_data.py``."""

    seed: int = 42
    dataset: DatasetConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


class WordTfidfConfig(StrictModel):
    """Word-level TF-IDF settings."""

    enabled: bool = True
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.9
    max_features: int | None = 300_000
    sublinear_tf: bool = True
    lowercase: bool = True
    strip_accents: str | None = "unicode"
    token_pattern: str = r"(?u)\b\w[\w'’\-]+\b"
    use_idf: bool = True
    binary: bool = False
    norm: Literal["l1", "l2"] | None = "l2"


class CharTfidfConfig(StrictModel):
    """Character n-gram TF-IDF settings (disabled by default; see README)."""

    enabled: bool = False
    analyzer: Literal["char", "char_wb"] = "char_wb"
    ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 3
    max_df: float = 1.0
    max_features: int | None = 100_000
    sublinear_tf: bool = True
    lowercase: bool = True
    strip_accents: str | None = "unicode"
    norm: Literal["l1", "l2"] | None = "l2"


class FeaturesConfig(StrictModel):
    """Feature extraction configuration."""

    word: WordTfidfConfig = Field(default_factory=WordTfidfConfig)
    char: CharTfidfConfig = Field(default_factory=CharTfidfConfig)


class SelectorConfig(StrictModel):
    """Optional univariate feature selection applied inside the pipeline."""

    enabled: bool = False
    method: Literal["chi2", "mutual_info"] = "chi2"
    k: int = 200_000


class DummyParams(StrictModel):
    enabled: bool = True
    strategy: Literal["prior", "most_frequent", "stratified", "uniform"] = "prior"


class LogisticRegressionParams(StrictModel):
    enabled: bool = True
    C: float = 1.0
    l1_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Regularisation mix (sklearn >= 1.8 API): 0.0 is pure L2, 1.0 is pure L1. "
            "Values in between require solver='saga'."
        ),
    )
    solver: Literal["liblinear", "lbfgs", "saga"] = "liblinear"
    max_iter: int = 1000
    class_weight: str | None = None
    tol: float = 1e-4


class ComplementNBParams(StrictModel):
    enabled: bool = True
    alpha: float = 1.0
    fit_prior: bool = True
    norm: bool = False


class RandomForestParams(StrictModel):
    enabled: bool = True
    n_estimators: int = 150
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    class_weight: str | None = "balanced_subsample"
    n_jobs: int = -1
    random_state: int | None = None


class CalibratedLinearSVCParams(StrictModel):
    enabled: bool = True
    C: float = 1.0
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge"
    calibration_cv: int = 3
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"


class ModelsConfig(StrictModel):
    """Candidate models compared during training."""

    dummy: DummyParams = Field(default_factory=DummyParams)
    logistic_regression: LogisticRegressionParams = Field(default_factory=LogisticRegressionParams)
    complement_nb: ComplementNBParams = Field(default_factory=ComplementNBParams)
    random_forest: RandomForestParams = Field(default_factory=RandomForestParams)
    linear_svc_calibrated: CalibratedLinearSVCParams = Field(
        default_factory=CalibratedLinearSVCParams
    )


class CrossValidationConfig(StrictModel):
    """Cross-validation settings for candidate comparison.

    Cross-validation runs on a stratified subsample so the comparison stays
    affordable on CPU-only hardware; the *reported* metrics always come from the
    full validation split, and the final model is refit on the entire training
    split.
    """

    enabled: bool = True
    folds: int = 3
    max_samples: int | None = 12_000
    n_jobs: int = 1


class SelectionConfig(StrictModel):
    """How the final model is chosen. Never uses the test set."""

    primary_metric: str = "macro_f1"
    tie_breakers: list[str] = Field(
        default_factory=lambda: ["f1_positive", "latency_ms_p95", "artifact_size_mb"]
    )
    latency_sample_size: int = 200


class CalibrationConfig(StrictModel):
    enabled: bool = True
    method: Literal["isotonic", "sigmoid"] = "isotonic"
    cv: int = 3


class ThresholdConfig(StrictModel):
    tune: bool = True
    metric: Literal["f1_positive", "balanced_accuracy", "youden"] = "f1_positive"
    grid_start: float = 0.05
    grid_stop: float = 0.95
    grid_step: float = 0.05
    default: float = 0.5

    def grid(self) -> list[float]:
        """Return the threshold grid as an explicit list of floats."""
        count = round((self.grid_stop - self.grid_start) / self.grid_step) + 1
        return [round(self.grid_start + i * self.grid_step, 6) for i in range(count)]


class TrainingConfig(StrictModel):
    """Top-level configuration consumed by ``scripts/train.py``."""

    seed: int = 42
    model_version: str = "1.0.0"
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    cv: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)

    @field_validator("seed")
    @classmethod
    def _positive_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value


# ---------------------------------------------------------------------------
# Service (API) configuration
# ---------------------------------------------------------------------------


class ServiceSettings(BaseSettings):
    """Runtime settings for the FastAPI service, read from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="FND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "fake-news-detection-api"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1
    log_level: str = "INFO"

    artifact_dir: str = "artifacts"
    model_artifact: str = "model.joblib"
    model_metadata: str = "model_metadata.json"

    decision_threshold: float | None = Field(
        default=None,
        description="Override the threshold stored in the model metadata.",
    )
    max_title_chars: int = 512
    max_text_chars: int = 20000
    max_request_bytes: int = 256 * 1024
    max_batch_size: int = 64
    min_total_chars: int = 4

    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    log_request_bodies: bool = False
    fail_fast_on_missing_model: bool = False
    disclaimer: str = (
        "This is a model classification based on statistical patterns learned from a "
        "labelled corpus, not independent verification of factual accuracy."
    )

    @field_validator("log_level")
    @classmethod
    def _valid_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @field_validator("decision_threshold")
    @classmethod
    def _valid_threshold(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError("decision_threshold must be in (0, 1)")
        return value

    @property
    def artifact_path(self) -> Path:
        """Absolute path of the model artifact."""
        return resolve_path(self.artifact_dir) / self.model_artifact

    @property
    def metadata_path(self) -> Path:
        """Absolute path of the model metadata JSON document."""
        return resolve_path(self.artifact_dir) / self.model_metadata


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    """Return the directory searched for YAML configuration files."""
    override = os.environ.get(DEFAULT_CONFIG_DIR_ENV)
    if override:
        return resolve_path(override)
    return repo_root() / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text("utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping at the top level of {path}")
    return payload


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Load and validate ``configs/data.yaml`` (or a custom path)."""
    resolved = Path(path) if path else config_dir() / "data.yaml"
    return DataConfig.model_validate(load_yaml(resolved))


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    """Load and validate ``configs/training.yaml`` (or a custom path)."""
    resolved = Path(path) if path else config_dir() / "training.yaml"
    return TrainingConfig.model_validate(load_yaml(resolved))


def validate_config_files(data_cfg: DataConfig, training_cfg: TrainingConfig) -> None:
    """Cross-file checks that a single model cannot express."""
    if data_cfg.seed != training_cfg.seed:
        raise ValueError(
            "data.yaml seed and training.yaml seed differ; use the same value for "
            "reproducibility (or update both deliberately)"
        )
    # Cross-validation inside the calibrated candidate must not exceed the
    # number of samples we can afford to hold out per fold.
    if training_cfg.cv.enabled and training_cfg.cv.folds < 2:
        raise ValueError("cv.folds must be >= 2 when cross-validation is enabled")
    if training_cfg.calibration.enabled and training_cfg.calibration.cv < 2:
        raise ValueError("calibration.cv must be >= 2 when calibration is enabled")


def load_service_settings(**overrides: Any) -> ServiceSettings:
    """Load service settings with optional explicit overrides (tests, CLI)."""
    return ServiceSettings(**overrides) if overrides else ServiceSettings()


def describe_paths(data_cfg: DataConfig, base: Path | None = None) -> dict[str, Path]:
    """Return the concrete input paths declared by ``data_cfg``."""
    return {source.path: resolve_path(source.path, base) for source in data_cfg.dataset.sources}
