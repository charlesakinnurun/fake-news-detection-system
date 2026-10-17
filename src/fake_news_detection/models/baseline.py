"""Construction of the candidate estimators and the end-to-end pipelines.

A "pipeline" here is always the complete inference unit:

``TextCombiner`` -> ``TfidfVectorizer`` (or a word+char ``FeatureUnion``)
-> optional ``SelectKBest`` -> classifier

Because the text normalisation is the first step, saving the pipeline saves the
preprocessing too, and there is no second implementation that could drift.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from fake_news_detection.config import PreprocessingConfig, TrainingConfig
from fake_news_detection.data.preprocessing import TextCombiner
from fake_news_detection.features.tfidf import build_feature_extraction, build_selector

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CANDIDATE_NAMES",
    "build_estimator",
    "build_pipeline",
    "configured_candidates",
]

CANDIDATE_NAMES = (
    "dummy",
    "logistic_regression",
    "complement_nb",
    "random_forest",
    "linear_svc_calibrated",
)


def is_linear_model(name: str) -> bool:
    """Return ``True`` for candidates whose decision function is linear."""
    return name in {"logistic_regression", "linear_svc_calibrated", "complement_nb"}


def build_estimator(name: str, config: TrainingConfig) -> Any:
    """Instantiate the estimator (or calibrated wrapper) for a candidate name."""
    models = config.models

    if name == "dummy":
        return DummyClassifier(strategy=models.dummy.strategy)

    if name == "logistic_regression":
        logistic_params = models.logistic_regression
        return LogisticRegression(
            C=logistic_params.C,
            l1_ratio=logistic_params.l1_ratio,
            solver=logistic_params.solver,
            max_iter=logistic_params.max_iter,
            class_weight=logistic_params.class_weight,
            tol=logistic_params.tol,
            random_state=config.seed,
        )

    if name == "complement_nb":
        nb_params = models.complement_nb
        return ComplementNB(
            alpha=nb_params.alpha, fit_prior=nb_params.fit_prior, norm=nb_params.norm
        )

    if name == "random_forest":
        forest_params = models.random_forest
        return RandomForestClassifier(
            n_estimators=forest_params.n_estimators,
            max_depth=forest_params.max_depth,
            min_samples_leaf=forest_params.min_samples_leaf,
            max_features=forest_params.max_features,
            class_weight=forest_params.class_weight,
            n_jobs=forest_params.n_jobs,
            random_state=(
                forest_params.random_state
                if forest_params.random_state is not None
                else config.seed
            ),
        )

    if name == "linear_svc_calibrated":
        svc_params = models.linear_svc_calibrated
        base = LinearSVC(
            C=svc_params.C,
            loss=svc_params.loss,
            random_state=config.seed,
            max_iter=5000,
            dual="auto",
        )
        # Calibration happens with an internal cross-validation on the training
        # split only, so this wrapper never sees validation or test data.
        return CalibratedClassifierCV(
            estimator=base,
            method=svc_params.calibration_method,
            cv=svc_params.calibration_cv,
        )

    raise ValueError(f"unknown candidate model: {name!r}; expected one of {CANDIDATE_NAMES}")


def configured_candidates(config: TrainingConfig) -> list[str]:
    """Return the candidate names enabled in ``training.yaml``, in a stable order."""
    enabled = {
        "dummy": config.models.dummy.enabled,
        "logistic_regression": config.models.logistic_regression.enabled,
        "complement_nb": config.models.complement_nb.enabled,
        "random_forest": config.models.random_forest.enabled,
        "linear_svc_calibrated": config.models.linear_svc_calibrated.enabled,
    }
    names = [name for name in CANDIDATE_NAMES if enabled.get(name)]
    if not names:
        raise ValueError("no candidate models are enabled in configs/training.yaml")
    if "dummy" not in names:
        LOGGER.warning("the dummy baseline is disabled; model selection has no reference point")
    return names


def build_pipeline(
    name: str,
    config: TrainingConfig,
    preprocessing: PreprocessingConfig,
    *,
    title_column: str = "title",
    text_column: str = "text",
) -> Pipeline:
    """Build the complete, serialisable inference pipeline for a candidate."""
    steps: list[tuple[str, Any]] = [
        (
            "text",
            TextCombiner.from_config(
                preprocessing, title_column=title_column, text_column=text_column
            ),
        ),
        ("features", build_feature_extraction(config.features)),
    ]
    selector = build_selector(config.selector)
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("clf", build_estimator(name, config)))
    return Pipeline(steps=steps)
