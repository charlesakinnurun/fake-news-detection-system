"""Unit tests for text normalisation and headline/body combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fake_news_detection.config import PreprocessingConfig
from fake_news_detection.data.preprocessing import (
    TextCombiner,
    collapse_whitespace,
    combine_fields,
    normalize_text,
    normalize_unicode,
    remove_control_chars,
    strip_html,
    strip_urls,
)


class TestNormalisation:
    def test_unicode_is_nfkc_normalised(self) -> None:
        assert normalize_unicode("ＦＡＫＥ   ＮＥＷＳ") == "FAKE   NEWS"
        assert normalize_unicode("ﬁnance") == "finance"

    def test_control_characters_removed(self) -> None:
        assert remove_control_chars("hello\x00world\x0btest") == "hello world test"

    def test_whitespace_collapsed(self) -> None:
        assert collapse_whitespace("  a \t\t b \n\n  c  ") == "a b\nc"

    def test_urls_stripped_when_enabled(self) -> None:
        assert strip_urls("see https://example.com/x for more") == "see   for more"

    def test_html_stripped_when_enabled(self) -> None:
        assert " ".join(strip_html("<p>hello <b>world</b></p>").split()) == "hello world"

    @pytest.mark.parametrize("value", [None, float("nan")])
    def test_missing_values_become_empty_strings(self, value: object) -> None:
        assert normalize_text(value) == ""

    def test_punctuation_is_preserved_by_default(self) -> None:
        config = PreprocessingConfig()
        text = normalize_text("It's not true, is it?", config)
        assert "?" in text and "'" in text and "," in text

    def test_number_replacement_is_opt_in(self) -> None:
        assert normalize_text("in 2018", PreprocessingConfig(replace_numbers=True)) == "in <num>"
        assert "2018" in normalize_text("in 2018", PreprocessingConfig(replace_numbers=False))

    def test_deterministic(self) -> None:
        config = PreprocessingConfig()
        sample = "  Breaking—news   about ＮＡＳＡ  "
        assert normalize_text(sample, config) == normalize_text(sample, config)


class TestCombineFields:
    def test_title_body_concatenation(self) -> None:
        config = PreprocessingConfig(text_field="title_body", separator=" | ")
        assert combine_fields("Title", "Body", config) == "Title | Body"

    def test_title_only(self) -> None:
        config = PreprocessingConfig(text_field="title_only")
        assert combine_fields("Title", "Body", config) == "Title"

    def test_body_only(self) -> None:
        config = PreprocessingConfig(text_field="body_only")
        assert combine_fields("Title", "Body", config) == "Body"

    def test_missing_side_falls_back_to_the_other(self) -> None:
        config = PreprocessingConfig(text_field="title_body")
        assert combine_fields("", "Body", config) == "Body"
        assert combine_fields("Title", "", config) == "Title"
        assert combine_fields("", "", config) == ""


class TestTextCombiner:
    def test_accepts_dataframe_and_returns_array(self) -> None:
        combiner = TextCombiner.from_config(PreprocessingConfig())
        frame = pd.DataFrame({"title": ["Head"], "text": ["Body text."]})
        result = combiner.fit_transform(frame)
        assert isinstance(result, np.ndarray)
        assert "Head" in result[0] and "Body text." in result[0]

    def test_string_input_is_treated_as_body(self) -> None:
        combiner = TextCombiner.from_config(PreprocessingConfig())
        result = combiner.fit_transform(["only a body"])
        assert result[0] == "only a body"

    def test_series_and_dict_inputs(self) -> None:
        combiner = TextCombiner.from_config(PreprocessingConfig())
        assert combiner.transform(pd.Series(["a", "b"])).tolist() == ["a", "b"]
        assert combiner.transform({"title": ["t"], "text": ["b"]}).tolist() == ["t \n b"]

    def test_missing_columns_do_not_raise(self) -> None:
        combiner = TextCombiner.from_config(PreprocessingConfig())
        frame = pd.DataFrame({"title": ["Head"]})
        assert combiner.transform(frame).tolist() == ["Head"]

    def test_none_and_nan_are_handled(self) -> None:
        combiner = TextCombiner.from_config(PreprocessingConfig())
        frame = pd.DataFrame({"title": [None, "T"], "text": [np.nan, ""]})
        combined = combiner.transform(frame)
        assert combined.tolist() == ["", "T"]

    def test_round_trip_config(self) -> None:
        original = PreprocessingConfig(text_field="body_only", lowercase=True, strip_urls=True)
        restored = TextCombiner.from_config(original).to_config()
        assert restored == original

    def test_unsupported_input_raises_type_error(self) -> None:
        combiner = TextCombiner.from_config(PreprocessingConfig())
        with pytest.raises(TypeError):
            combiner.transform(12345)

    def test_get_feature_names_out(self) -> None:
        combiner = TextCombiner.from_config(PreprocessingConfig())
        assert combiner.get_feature_names_out().tolist() == ["combined_text"]

    def test_sklearn_clone_is_supported(self) -> None:
        from sklearn.base import clone

        combiner = TextCombiner.from_config(PreprocessingConfig(text_field="title_only"))
        cloned = clone(combiner)
        assert cloned.get_params() == combiner.get_params()
