"""Unit tests for TF-IDF construction and linear-model attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fake_news_detection.config import (
    CharTfidfConfig,
    FeaturesConfig,
    PreprocessingConfig,
    SelectorConfig,
    WordTfidfConfig,
)
from fake_news_detection.data.preprocessing import TextCombiner
from fake_news_detection.features.tfidf import (
    build_char_vectorizer,
    build_feature_extraction,
    build_selector,
    build_word_vectorizer,
    explain_document,
    supports_attribution,
    top_features_by_class,
    unwrap_pipeline,
    word_vectorizer,
)


def _tiny_pipeline(seed: int = 0) -> Pipeline:
    from sklearn.linear_model import LogisticRegression

    texts = [
        "the council approved the budget after a long debate",
        "officials reported that unemployment fell last quarter",
        "shocking secret the government hides about water",
        "miracle cure big pharma does not want you to know",
    ]
    labels = np.array(["REAL", "REAL", "FAKE", "FAKE"])
    pipeline = Pipeline(
        [
            ("text", TextCombiner.from_config(PreprocessingConfig())),
            ("features", build_word_vectorizer(WordTfidfConfig(min_df=1, max_features=500))),
            ("clf", LogisticRegression(max_iter=200, random_state=seed)),
        ]
    )
    pipeline.fit(pd.DataFrame({"title": ["", "", "", ""], "text": texts}), labels)
    return pipeline


class TestVectorizerConstruction:
    def test_word_vectorizer_honours_configuration(self) -> None:
        vectorizer = build_word_vectorizer(
            WordTfidfConfig(ngram_range=(1, 3), min_df=2, max_features=123, sublinear_tf=True)
        )
        assert vectorizer.ngram_range == (1, 3)
        assert vectorizer.min_df == 2
        assert vectorizer.max_features == 123
        assert vectorizer.sublinear_tf is True

    def test_char_vectorizer_honours_configuration(self) -> None:
        vectorizer = build_char_vectorizer(CharTfidfConfig(analyzer="char", ngram_range=(2, 4)))
        assert vectorizer.analyzer == "char"
        assert vectorizer.ngram_range == (2, 4)

    def test_single_enabled_feature_returns_a_vectorizer_not_a_union(self) -> None:
        extraction = build_feature_extraction(
            FeaturesConfig(word=WordTfidfConfig(enabled=True), char=CharTfidfConfig(enabled=False))
        )
        assert hasattr(extraction, "vocabulary") or hasattr(extraction, "fit")
        assert extraction.__class__.__name__ == "TfidfVectorizer"

    def test_both_enabled_returns_a_feature_union(self) -> None:
        extraction = build_feature_extraction(
            FeaturesConfig(word=WordTfidfConfig(enabled=True), char=CharTfidfConfig(enabled=True))
        )
        assert extraction.__class__.__name__ == "FeatureUnion"
        assert {name for name, _ in extraction.transformer_list} == {"word", "char"}

    def test_no_feature_enabled_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_feature_extraction(
                FeaturesConfig(
                    word=WordTfidfConfig(enabled=False), char=CharTfidfConfig(enabled=False)
                )
            )

    def test_selector_is_optional(self) -> None:
        assert build_selector(SelectorConfig(enabled=False)) is None
        assert build_selector(SelectorConfig(enabled=True, k=10)) is not None


class TestVectorizerFitting:
    def test_vocabulary_is_learned_from_training_data_only(self) -> None:
        vectorizer = build_word_vectorizer(WordTfidfConfig(min_df=1))
        vectorizer.fit(["alpha beta gamma", "delta epsilon"])
        assert "zeta" not in vectorizer.vocabulary_
        matrix = vectorizer.transform(["zeta"])
        assert matrix.nnz == 0

    def test_word_vectorizer_is_found_inside_a_pipeline(self) -> None:
        pipeline = _tiny_pipeline()
        assert word_vectorizer(pipeline) is not None

    def test_word_vectorizer_returns_none_for_char_only(self) -> None:
        pipeline = Pipeline(
            [
                ("text", TextCombiner.from_config(PreprocessingConfig())),
                ("features", build_char_vectorizer(CharTfidfConfig())),
            ]
        )
        assert word_vectorizer(pipeline) is None


class TestAttribution:
    def test_supports_attribution_for_logistic_regression(self) -> None:
        pipeline = _tiny_pipeline()
        assert supports_attribution(pipeline.steps[-1][1], "FAKE") is True

    def test_explanations_are_signed_and_ranked(self) -> None:
        pipeline = _tiny_pipeline()
        [explanation] = explain_document(
            pipeline, ["shocking miracle cure the government hides"], positive_label="FAKE", top_n=5
        )
        assert explanation["supported"] is True
        assert explanation["method"] == "tfidf_linear_attribution"
        contributions = [abs(feature["contribution"]) for feature in explanation["features"]]
        assert contributions == sorted(contributions, reverse=True)
        assert all(feature["direction"] in {"FAKE", "OTHER"} for feature in explanation["features"])
        assert explanation["note"]

    def test_explanation_direction_flips_with_the_positive_label(self) -> None:
        pipeline = _tiny_pipeline()
        text = ["shocking miracle cure the government hides"]
        as_fake = explain_document(pipeline, text, positive_label="FAKE", top_n=3)[0]
        as_real = explain_document(pipeline, text, positive_label="REAL", top_n=3)[0]
        fake_top = {feature["feature"]: feature["contribution"] for feature in as_fake["features"]}
        real_top = {feature["feature"]: feature["contribution"] for feature in as_real["features"]}
        shared = set(fake_top) & set(real_top)
        assert shared, "expected overlapping features between the two directions"
        for feature in shared:
            assert fake_top[feature] == pytest.approx(-real_top[feature])

    def test_empty_document_yields_no_features(self) -> None:
        pipeline = _tiny_pipeline()
        explanation = explain_document(pipeline, [""], positive_label="FAKE")[0]
        assert explanation["supported"] is True
        assert explanation["features"] == []

    def test_models_without_linear_scores_report_unsupported(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        pipeline = Pipeline(
            [
                ("text", TextCombiner.from_config(PreprocessingConfig())),
                ("features", build_word_vectorizer(WordTfidfConfig(min_df=1))),
                ("clf", RandomForestClassifier(n_estimators=3, random_state=0)),
            ]
        )
        pipeline.fit(
            pd.DataFrame({"title": ["", ""], "text": ["alpha beta", "gamma delta"]}),
            np.array(["FAKE", "REAL"]),
        )
        explanation = explain_document(pipeline, ["alpha"], positive_label="FAKE")[0]
        assert explanation["supported"] is False
        assert explanation["features"] == []
        assert "does not expose a linear decision function" in explanation["reason"]

    def test_top_features_by_class(self) -> None:
        pipeline = _tiny_pipeline()
        features = top_features_by_class(pipeline, "FAKE", top_n=3)
        assert set(features) == {"FAKE", "OTHER"}
        assert len(features["FAKE"]) == 3
        assert all("feature" in item and "coefficient" in item for item in features["FAKE"])


class TestUnwrapPipeline:
    def test_plain_pipeline_is_returned(self) -> None:
        pipeline = _tiny_pipeline()
        assert unwrap_pipeline(pipeline) is pipeline

    def test_calibrated_pipeline_is_unwrapped(self) -> None:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator

        base = _tiny_pipeline()
        texts = [
            "the council approved the budget",
            "officials reported unemployment fell",
            "shocking secret the government hides",
            "miracle cure big pharma hides",
            "another council budget approved",
            "more shocking miracle claims",
        ]
        labels = np.array(["REAL", "REAL", "FAKE", "FAKE", "REAL", "FAKE"])
        calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", cv=2)
        calibrator.fit(pd.DataFrame({"title": [""] * len(texts), "text": texts}), labels)
        assert unwrap_pipeline(calibrator) is base

    def test_unknown_object_returns_none(self) -> None:
        assert unwrap_pipeline(object()) is None
