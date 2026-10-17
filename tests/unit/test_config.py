"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fake_news_detection.config import (
    DataConfig,
    DatasetConfig,
    ServiceSettings,
    SplittingConfig,
    ThresholdConfig,
    TrainingConfig,
    load_data_config,
    load_training_config,
    repo_root,
    resolve_path,
    validate_config_files,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class TestRepositoryConfigs:
    def test_data_config_loads_and_validates(self) -> None:
        config = load_data_config(REPO_CONFIGS / "data.yaml")
        assert isinstance(config, DataConfig)
        assert config.dataset.positive_label == "FAKE"
        assert len(config.dataset.sources) == 2
        assert (
            abs(
                config.splitting.train_size
                + config.splitting.val_size
                + config.splitting.test_size
                - 1.0
            )
            < 1e-9
        )

    def test_training_config_loads_and_validates(self) -> None:
        config = load_training_config(REPO_CONFIGS / "training.yaml")
        assert isinstance(config, TrainingConfig)
        assert config.features.word.enabled is True
        assert config.selection.primary_metric == "macro_f1"

    def test_seeds_agree_across_files(self) -> None:
        data = load_data_config(REPO_CONFIGS / "data.yaml")
        training = load_training_config(REPO_CONFIGS / "training.yaml")
        validate_config_files(data, training)

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetConfig(name="x", unexpected_key=True)


class TestValidationRules:
    def test_split_sizes_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError):
            SplittingConfig(train_size=0.5, val_size=0.1, test_size=0.1)

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(seed=-1)

    def test_positive_label_must_be_declared(self) -> None:
        with pytest.raises(ValidationError):
            DatasetConfig(name="x", labels=["A", "B"], positive_label="C")

    def test_threshold_grid_is_bounded_and_increasing(self) -> None:
        grid = ThresholdConfig(grid_start=0.1, grid_stop=0.5, grid_step=0.1).grid()
        assert grid == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert grid == sorted(grid)

    def test_near_duplicate_bands_must_divide_hashes(self) -> None:
        with pytest.raises(ValidationError):
            SplittingConfig(near_duplicate={"num_hashes": 64, "num_bands": 7})


class TestServiceSettings:
    def test_defaults_are_safe(self) -> None:
        settings = ServiceSettings(_env_file=None)
        assert settings.rate_limit_enabled is True
        assert settings.log_request_bodies is False
        assert settings.max_batch_size > 0

    def test_environment_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FND_API_PORT", "9999")
        monkeypatch.setenv("FND_MAX_TEXT_CHARS", "123")
        settings = ServiceSettings(_env_file=None)
        assert settings.api_port == 9999
        assert settings.max_text_chars == 123

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ServiceSettings(_env_file=None, decision_threshold=1.5)

    def test_unknown_log_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ServiceSettings(_env_file=None, log_level="SHOUTING")

    def test_artifact_paths_resolve_under_repo_root(self) -> None:
        settings = ServiceSettings(_env_file=None, artifact_dir="artifacts")
        assert settings.artifact_path.name == "model.joblib"
        assert settings.metadata_path.name == "model_metadata.json"

    def test_absolute_artifact_dir_is_honoured(self, tmp_path: Path) -> None:
        settings = ServiceSettings(_env_file=None, artifact_dir=str(tmp_path))
        assert settings.artifact_path.parent == tmp_path


class TestPathHelpers:
    def test_relative_paths_resolve_against_repo_root(self) -> None:
        assert resolve_path("data/raw/True.csv") == repo_root() / "data" / "raw" / "True.csv"

    def test_absolute_paths_are_untouched(self, tmp_path: Path) -> None:
        assert resolve_path(tmp_path) == tmp_path

    def test_missing_yaml_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_data_config(REPO_CONFIGS / "does_not_exist.yaml")
