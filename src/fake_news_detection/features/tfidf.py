"""TF-IDF feature extraction and linear-model attribution.

Everything here is fitted *inside* a scikit-learn pipeline, so the vocabulary,
IDF weights and feature selection statistics are learned from the training split
only and are serialised with the model. There is no way to accidentally fit a
vectorizer on the test split.

Attribution is only produced when the final estimator exposes a genuine linear
parameterisation (Logistic Regression, LinearSVC, Complement NB). For other
estimators the explanation is reported as unavailable rather than approximated
with something that would look authoritative but would not be the model's own
decision function.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.pipeline import FeatureUnion, Pipeline

from fake_news_detection.config import (
    CharTfidfConfig,
    FeaturesConfig,
    SelectorConfig,
    WordTfidfConfig,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "build_char_vectorizer",
    "build_feature_extraction",
    "build_selector",
    "build_word_vectorizer",
    "explain_document",
    "supports_attribution",
    "top_features_by_class",
    "unwrap_pipeline",
    "word_vectorizer",
]

EXPLAINABILITY_NOTE = (
    "Attribution lists the features with the largest signed contribution to the model's own "
    "decision score for this input. It explains the classifier, not the article: a strongly "
    "positive feature means the model associated that phrasing with the labelled class in the "
    "training corpus, not that the statement is true or false."
)


# ---------------------------------------------------------------------------
# Vectorizers
# ---------------------------------------------------------------------------


def build_word_vectorizer(config: WordTfidfConfig) -> TfidfVectorizer:
    """Construct the word-level TF-IDF vectorizer from configuration."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(config.ngram_range),
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
        lowercase=config.lowercase,
        strip_accents=config.strip_accents,
        token_pattern=config.token_pattern,
        use_idf=config.use_idf,
        binary=config.binary,
        norm=config.norm,
        dtype=np.float64,
    )


def build_char_vectorizer(config: CharTfidfConfig) -> TfidfVectorizer:
    """Construct the character n-gram TF-IDF vectorizer from configuration."""
    return TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=tuple(config.ngram_range),
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
        lowercase=config.lowercase,
        strip_accents=config.strip_accents,
        norm=config.norm,
        dtype=np.float64,
    )


def build_feature_extraction(config: FeaturesConfig) -> BaseEstimator:
    """Return either a single vectorizer or a union of word and character features."""
    parts: list[tuple[str, Any]] = []
    if config.word.enabled:
        parts.append(("word", build_word_vectorizer(config.word)))
    if config.char.enabled:
        parts.append(("char", build_char_vectorizer(config.char)))
    if not parts:
        raise ValueError(
            "at least one of features.word.enabled / features.char.enabled must be true"
        )
    if len(parts) == 1:
        return parts[0][1]
    return FeatureUnion(transformer_list=parts)


def build_selector(config: SelectorConfig) -> BaseEstimator | None:
    """Return an optional univariate feature selector."""
    if not config.enabled:
        return None
    score_func = chi2 if config.method == "chi2" else mutual_info_classif
    return SelectKBest(score_func=score_func, k=config.k)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def word_vectorizer(pipeline: Pipeline) -> TfidfVectorizer | None:
    """Return the word-level TF-IDF vectorizer inside ``pipeline``, if present.

    A character n-gram vectorizer is *not* a word vectorizer, even though it is
    the same class; the ``analyzer`` attribute decides.
    """
    features = pipeline.named_steps.get("features")
    if isinstance(features, TfidfVectorizer):
        return features if features.analyzer == "word" else None
    if isinstance(features, FeatureUnion):
        for name, transformer in features.transformer_list:
            if name == "word" and isinstance(transformer, TfidfVectorizer):
                return transformer
    return None


def _feature_names(pipeline: Pipeline) -> np.ndarray:
    """Return output feature names for every step before the estimator."""
    prefix = Pipeline(pipeline.steps[:-1])
    return np.asarray(prefix.get_feature_names_out(), dtype=object)


def _transform_prefix(pipeline: Pipeline, texts: list[str]) -> tuple[Any, np.ndarray]:
    """Transform raw texts through every step before the final estimator."""
    prefix = Pipeline(pipeline.steps[:-1])
    matrix = prefix.transform(texts)
    return matrix, _feature_names(pipeline)


def _unwrap_to_inner(estimator: Any) -> Any:
    """Peel calibration wrappers until a concrete estimator is reached."""
    try:
        from sklearn.frozen import FrozenEstimator

        frozen_types: tuple[type, ...] = (FrozenEstimator,)
    except ImportError:  # pragma: no cover - sklearn < 1.6
        frozen_types = ()

    seen = 0
    while seen < 5:
        seen += 1
        if isinstance(estimator, frozen_types):
            estimator = estimator.estimator  # type: ignore[attr-defined]
            continue
        calibrated = getattr(estimator, "calibrated_classifiers_", None)
        if calibrated:
            estimator = calibrated[0].estimator
            continue
        break
    return estimator


def unwrap_pipeline(model: Any) -> Pipeline | None:
    """Return the underlying sklearn :class:`Pipeline` of a possibly calibrated model."""
    candidate = _unwrap_to_inner(model)
    if isinstance(candidate, Pipeline):
        return candidate
    if isinstance(model, Pipeline):
        return model
    return None


def _positive_class_index(estimator: Any, positive_label: str) -> int:
    classes = list(getattr(estimator, "classes_", []))
    if positive_label in classes:
        return classes.index(positive_label)
    return 0


def _signed_coefficients(estimator: Any, positive_label: str) -> tuple[np.ndarray, float] | None:
    """Return ``(coef, intercept)`` such that a higher score means ``positive_label``."""
    index = (
        _positive_class_index(estimator, positive_label) if hasattr(estimator, "classes_") else 0
    )

    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef, dtype=np.float64)
        raw_intercept = float(np.asarray(getattr(estimator, "intercept_", 0.0)).ravel()[0])
        if coef.ndim == 1:
            return (coef if index == 1 else -coef), (
                raw_intercept if index == 1 else -raw_intercept
            )
        if coef.shape[0] == 1:
            # Binary problem: the single row scores ``classes_[1]``.
            return (coef[0] if index == 1 else -coef[0]), (
                raw_intercept if index == 1 else -raw_intercept
            )
        intercepts = np.asarray(getattr(estimator, "intercept_", np.zeros(coef.shape[0])))
        return coef[index], float(intercepts.ravel()[index])

    log_prob = getattr(estimator, "feature_log_prob_", None)
    if log_prob is not None:
        log_prob = np.asarray(log_prob, dtype=np.float64)
        if log_prob.shape[0] < 2:
            return None
        negative_index = 1 - index if log_prob.shape[0] == 2 else index
        class_log_prior = np.asarray(
            getattr(estimator, "class_log_prior_", np.zeros(log_prob.shape[0]))
        )
        vector = log_prob[index] - log_prob[negative_index]
        intercept = float(class_log_prior[index] - class_log_prior[negative_index])
        return vector, intercept

    return None


def supports_attribution(estimator: Any, positive_label: str) -> bool:
    """Return ``True`` when signed linear attribution is available for this model."""
    return _signed_coefficients(_unwrap_to_inner(estimator), positive_label) is not None


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def top_features_by_class(
    model: Any, positive_label: str, top_n: int = 25
) -> dict[str, list[dict[str, float | str]]]:
    """Return the highest-weighted n-grams for each class, for the model card."""
    pipeline = unwrap_pipeline(model)
    if pipeline is None:
        return {}
    estimator = _unwrap_to_inner(pipeline.steps[-1][1])
    signed = _signed_coefficients(estimator, positive_label)
    if signed is None:
        return {}
    vector, _ = signed
    names = _feature_names(pipeline)
    if names.shape[0] != vector.shape[0]:
        LOGGER.warning("feature/coefficient dimension mismatch; skipping top-feature export")
        return {}
    order = np.argsort(vector)
    positive = order[::-1][:top_n]
    negative = order[:top_n]
    return {
        positive_label: [
            {"feature": str(names[i]), "coefficient": round(float(vector[i]), 5)} for i in positive
        ],
        "OTHER": [
            {"feature": str(names[i]), "coefficient": round(float(vector[i]), 5)} for i in negative
        ],
    }


def explain_document(
    model: Any,
    texts: list[str],
    *,
    positive_label: str,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Return per-document signed feature contributions for linear models."""
    pipeline = unwrap_pipeline(model)
    unsupported: list[dict[str, Any]] = [
        {
            "supported": False,
            "reason": (
                "The selected model does not expose a linear decision function, so no "
                "feature attribution is produced."
            ),
            "features": [],
            "method": None,
            "note": EXPLAINABILITY_NOTE,
        }
        for _ in texts
    ]
    if pipeline is None:
        return unsupported
    estimator = _unwrap_to_inner(pipeline.steps[-1][1])
    signed = _signed_coefficients(estimator, positive_label)
    if signed is None:
        return unsupported

    vector, _ = signed
    matrix, names = _transform_prefix(pipeline, texts)
    if matrix.shape[1] != vector.shape[0]:
        return [
            {
                "supported": False,
                "reason": "Feature dimension mismatch between the vectorizer and the estimator.",
                "features": [],
                "method": None,
                "note": EXPLAINABILITY_NOTE,
            }
            for _ in texts
        ]

    outputs: list[dict[str, Any]] = []
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        indices = matrix.indices[start:stop]
        values = matrix.data[start:stop]
        contributions = values * vector[indices]
        if contributions.size == 0:
            outputs.append(
                {
                    "supported": True,
                    "method": "tfidf_linear_attribution",
                    "features": [],
                    "note": EXPLAINABILITY_NOTE,
                }
            )
            continue
        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        outputs.append(
            {
                "supported": True,
                "method": "tfidf_linear_attribution",
                "features": [
                    {
                        "feature": str(names[indices[i]]),
                        "contribution": round(float(contributions[i]), 6),
                        "direction": positive_label if contributions[i] > 0 else "OTHER",
                    }
                    for i in order
                ],
                "note": EXPLAINABILITY_NOTE,
            }
        )
    return outputs
