"""Shared pytest fixtures.

Every fixture is synthetic and generated in-process: the test suite never
downloads the real corpus and never depends on `data/processed/*.parquet`.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fake_news_detection.api.main import create_app
from fake_news_detection.config import (
    CalibratedLinearSVCParams,
    CalibrationConfig,
    CharTfidfConfig,
    ComplementNBParams,
    CrossValidationConfig,
    DataConfig,
    DatasetConfig,
    DataSourceConfig,
    DummyParams,
    FeaturesConfig,
    LogisticRegressionParams,
    ModelsConfig,
    PreprocessingConfig,
    RandomForestParams,
    SelectorConfig,
    ServiceSettings,
    SplittingConfig,
    ThresholdConfig,
    TrainingConfig,
    WordTfidfConfig,
    get_service_settings,
)
from fake_news_detection.data.splitting import SplitResult, make_splits
from fake_news_detection.data.validation import validate_and_clean
from fake_news_detection.models.baseline import build_pipeline
from fake_news_detection.models.registry import ModelMetadata, save_bundle

TEST_DATA_DIR = Path(__file__).parent / "test_data"

REAL_SENTENCES = [
    "The council approved the annual budget after a two hour debate on Tuesday.",
    "Officials said the bridge will reopen to traffic once inspections are complete.",
    "The agency reported that unemployment fell slightly in the last quarter.",
    "The university announced a new research grant for materials science.",
    "The rail operator added two evening services on the northern line.",
]
FAKE_SENTENCES = [
    "SHOCKING secret the government does not want you to know about tap water!",
    "Insiders reveal that the moon landing was filmed in a secret studio.",
    "Doctors hate this miracle cure that big pharma is hiding from you.",
    "You will not believe what this celebrity did next, mainstream media is silent.",
    "Wake up sheeple, the truth about the global elite is being suppressed!",
]


def build_synthetic_frame(rows_per_class: int = 80, seed: int = 0) -> pd.DataFrame:
    """Build a small, clearly synthetic two-class news corpus.

    The lexical split between the classes is deliberately obvious, but each
    document also carries a unique token sequence so that near-duplicate
    clustering does not collapse unrelated documents together. These tests
    verify plumbing, determinism and leakage control, not model quality.
    """
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for index in range(rows_per_class):
        for label, sentences, subjects in (
            ("REAL", REAL_SENTENCES, "politicsNews"),
            ("FAKE", FAKE_SENTENCES, "News"),
        ):
            body_sentences = [rng.choice(sentences) for _ in range(rng.randint(3, 6))]
            # A per-document tail with tokens that appear in no other document.
            unique_tail = " ".join(f"{label.lower()}token{index}{k}" for k in range(12))
            body = " ".join(body_sentences) + " " + unique_tail
            title = rng.choice(sentences).split(".")[0] + f" ({label.lower()}{index})"
            records.append(
                {
                    "row_id": f"synthetic:{label}:{index}",
                    "title": title,
                    "text": body,
                    "subject": subjects,
                    "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.random() * 300)),
                    "date_raw": "January 1, 2018",
                    "label": label,
                    "source": f"{label.lower()}.csv",
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["label"] = frame["label"].astype("string")
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_near_duplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Append a near copy of the first row (identical except for punctuation)."""
    clone = frame.iloc[[0]].copy()
    clone["row_id"] = "synthetic:clone:0"
    clone["text"] = (
        clone["text"].str.replace(".", ",", regex=False) + " In addition,   extra wording."
    )
    return pd.concat([frame, clone], ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return add_near_duplicate(build_synthetic_frame())


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    return DataConfig(
        seed=7,
        dataset=DatasetConfig(
            name="synthetic-fixture",
            version="0.1.0",
            labels=["FAKE", "REAL"],
            positive_label="FAKE",
            sources=[
                DataSourceConfig(path="tests/test_data/synthetic_news.csv", label="REAL"),
            ],
            columns={"title": "headline", "text": "body", "subject": "region", "date": "published"},
        ),
        validation={"min_text_chars": 10, "min_total_chars": 20, "max_reported_examples": 3},
        preprocessing=PreprocessingConfig(text_field="title_body"),
        splitting=SplittingConfig(
            strategy="grouped_stratified",
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            near_duplicate={
                "enabled": True,
                "shingle_size": 4,
                "num_hashes": 32,
                "num_bands": 8,
                "jaccard_threshold": 0.8,
            },
        ),
    )


@pytest.fixture(scope="session")
def training_config() -> TrainingConfig:
    return TrainingConfig(
        seed=7,
        model_version="0.0.1-test",
        features=FeaturesConfig(
            word=WordTfidfConfig(min_df=1, max_features=2000, ngram_range=(1, 2)),
            char=CharTfidfConfig(enabled=False),
        ),
        selector=SelectorConfig(enabled=False),
        models=ModelsConfig(
            dummy=DummyParams(enabled=True, strategy="prior"),
            logistic_regression=LogisticRegressionParams(enabled=True, C=1.0, max_iter=200),
            complement_nb=ComplementNBParams(enabled=False),
            random_forest=RandomForestParams(enabled=False),
            linear_svc_calibrated=CalibratedLinearSVCParams(enabled=False),
        ),
        cv=CrossValidationConfig(enabled=False, folds=2),
        calibration=CalibrationConfig(enabled=True, method="sigmoid", cv=2),
        threshold=ThresholdConfig(tune=True, grid_start=0.1, grid_stop=0.9, grid_step=0.1),
    )


@pytest.fixture(scope="session")
def splits(synthetic_frame: pd.DataFrame, data_config: DataConfig) -> SplitResult:
    clean, _ = validate_and_clean(synthetic_frame, data_config)
    return make_splits(clean, data_config)


@pytest.fixture(scope="session")
def fitted_pipeline(splits: SplitResult, training_config: TrainingConfig):
    pipeline = build_pipeline(
        "logistic_regression", training_config, PreprocessingConfig(text_field="title_body")
    )
    pipeline.fit(splits.train, splits.train["label"].to_numpy())
    return pipeline


@pytest.fixture(scope="session")
def model_metadata(fitted_pipeline: Any) -> ModelMetadata:
    return ModelMetadata(
        model_version="0.0.1-test",
        model_name="logistic_regression",
        labels=["FAKE", "REAL"],
        positive_label="FAKE",
        decision_threshold=0.5,
        text_field="title_body",
        dataset={"name": "synthetic-fixture", "version": "0.1.0", "fingerprint": "deadbeef"},
        training={"seed": 7},
        validation_metrics={"macro_f1": 0.9},
    )


@pytest.fixture(scope="session")
def artifact_paths(
    tmp_path_factory: pytest.TempPathFactory,
    fitted_pipeline: Any,
    model_metadata: ModelMetadata,
) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("artifacts")
    artifact = directory / "model.joblib"
    metadata = directory / "model_metadata.json"
    save_bundle(fitted_pipeline, model_metadata, artifact, metadata)
    return artifact, metadata


@pytest.fixture()
def service_settings(
    artifact_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> ServiceSettings:
    artifact, metadata = artifact_paths
    monkeypatch.setenv("FND_ARTIFACT_DIR", str(artifact.parent))
    monkeypatch.setenv("FND_MODEL_ARTIFACT", artifact.name)
    monkeypatch.setenv("FND_MODEL_METADATA", metadata.name)
    monkeypatch.setenv("FND_RATE_LIMIT_ENABLED", "false")
    get_service_settings.cache_clear()
    settings = get_service_settings()
    yield settings  # type: ignore[misc]
    get_service_settings.cache_clear()


@pytest.fixture()
def api_client(service_settings: ServiceSettings) -> Iterator[TestClient]:
    app = create_app(service_settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def empty_dir(tmp_path: Path) -> Path:
    """An artifact directory that deliberately contains no model."""
    directory = tmp_path / "no-model"
    directory.mkdir()
    return directory


@pytest.fixture()
def missing_model_settings(
    empty_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ServiceSettings]:
    monkeypatch.setenv("FND_ARTIFACT_DIR", str(empty_dir))
    monkeypatch.setenv("FND_RATE_LIMIT_ENABLED", "false")
    get_service_settings.cache_clear()
    yield get_service_settings()  # type: ignore[misc]
    get_service_settings.cache_clear()
