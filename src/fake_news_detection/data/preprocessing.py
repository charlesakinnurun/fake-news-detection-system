"""Deterministic text normalisation and headline/body combination.

The :class:`TextCombiner` transformer is the first step of every saved
scikit-learn pipeline, so the *exact* preprocessing used during training is
serialised with the model. There is no second implementation that could drift
out of sync at inference time.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fake_news_detection.config import PreprocessingConfig

__all__ = [
    "TextCombiner",
    "build_preprocessor",
    "collapse_whitespace",
    "combine_fields",
    "normalize_text",
    "normalize_unicode",
    "remove_control_chars",
    "strip_html",
    "strip_urls",
]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_NEWLINE_RE = re.compile(r"\s*\n\s*")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_NUMBER_RE = re.compile(r"\b\d[\d.,]*\b")
_TOKEN_SPLIT_RE = re.compile(r"\S+")


def normalize_unicode(text: str, *, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC") -> str:
    """Apply Unicode normalisation.

    NFKC folds compatibility characters (full-width letters, ligatures, the
    various dash and quote glyphs) onto their canonical form. This makes the
    vocabulary smaller and the pipeline reproducible across platforms without
    changing the semantic content of the text.
    """
    return unicodedata.normalize(form, text)


def remove_control_chars(text: str) -> str:
    """Remove control characters that break tokenisation and JSON encoding."""
    return _CONTROL_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Collapse horizontal runs of whitespace and strip newline padding."""
    text = _NEWLINE_RE.sub("\n", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def strip_urls(text: str, replacement: str = " ") -> str:
    """Replace URLs with ``replacement`` (default: a space)."""
    return _URL_RE.sub(replacement, text)


def strip_html(text: str) -> str:
    """Remove simple HTML tags. Tags are dropped, their text is kept."""
    return _HTML_TAG_RE.sub(" ", text)


def normalize_text(text: Any, config: PreprocessingConfig | None = None) -> str:
    """Apply the configured normalisation chain to a single string.

    ``None`` and ``NaN`` collapse to the empty string so downstream code never
    has to special-case missing values.
    """
    config = config or PreprocessingConfig()
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    value = str(text)
    if config.normalize_unicode:
        value = normalize_unicode(value)
    if config.remove_control_chars:
        value = remove_control_chars(value)
    if config.strip_html:
        value = strip_html(value)
    if config.strip_urls:
        value = strip_urls(value)
    if config.replace_numbers:
        value = _NUMBER_RE.sub(" <num> ", value)
    value = collapse_whitespace(value) if config.collapse_whitespace else value.strip()
    if config.lowercase:
        value = value.casefold()
    if config.min_token_length > 1:
        value = " ".join(
            token
            for token in _TOKEN_SPLIT_RE.findall(value)
            if len(token) >= config.min_token_length
        )
    return value


def combine_fields(title: str, body: str, config: PreprocessingConfig) -> str:
    """Combine a headline and an article body according to ``config.text_field``.

    Strategy:
        * ``title_body`` – ``"<title><separator><body>"`` (separator defaults to
          a newline, which keeps the two zones distinguishable for n-grams).
        * ``title_only`` – headline only (useful for headline-only models).
        * ``body_only``  – article body only.
    """
    if config.text_field == "title_only":
        return title
    if config.text_field == "body_only":
        return body
    if title and body:
        return f"{title}{config.separator}{body}"
    return title or body


class TextCombiner(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that turns ``(title, text)`` columns into one string.

    Parameters mirror :class:`~fake_news_detection.config.PreprocessingConfig`.
    The transformer is stateless: ``fit`` only records the feature names.
    """

    def __init__(
        self,
        text_field: str = "title_body",
        separator: str = " \n ",
        lowercase: bool = False,
        normalize_unicode_flag: bool = True,
        collapse_whitespace_flag: bool = True,
        remove_control_chars_flag: bool = True,
        strip_urls_flag: bool = False,
        strip_html_flag: bool = False,
        replace_numbers: bool = False,
        min_token_length: int = 0,
        title_column: str = "title",
        text_column: str = "text",
    ) -> None:
        self.text_field = text_field
        self.separator = separator
        self.lowercase = lowercase
        self.normalize_unicode_flag = normalize_unicode_flag
        self.collapse_whitespace_flag = collapse_whitespace_flag
        self.remove_control_chars_flag = remove_control_chars_flag
        self.strip_urls_flag = strip_urls_flag
        self.strip_html_flag = strip_html_flag
        self.replace_numbers = replace_numbers
        self.min_token_length = min_token_length
        self.title_column = title_column
        self.text_column = text_column

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_config(
        cls, config: PreprocessingConfig, *, title_column: str = "title", text_column: str = "text"
    ) -> TextCombiner:
        """Build a combiner from a :class:`PreprocessingConfig`."""
        return cls(
            text_field=config.text_field,
            separator=config.separator,
            lowercase=config.lowercase,
            normalize_unicode_flag=config.normalize_unicode,
            collapse_whitespace_flag=config.collapse_whitespace,
            remove_control_chars_flag=config.remove_control_chars,
            strip_urls_flag=config.strip_urls,
            strip_html_flag=config.strip_html,
            replace_numbers=config.replace_numbers,
            min_token_length=config.min_token_length,
            title_column=title_column,
            text_column=text_column,
        )

    def to_config(self) -> PreprocessingConfig:
        """Reconstruct the equivalent :class:`PreprocessingConfig`."""
        return PreprocessingConfig(
            text_field=self.text_field,  # type: ignore[arg-type]
            separator=self.separator,
            lowercase=self.lowercase,
            normalize_unicode=self.normalize_unicode_flag,
            collapse_whitespace=self.collapse_whitespace_flag,
            remove_control_chars=self.remove_control_chars_flag,
            strip_urls=self.strip_urls_flag,
            strip_html=self.strip_html_flag,
            replace_numbers=self.replace_numbers,
            min_token_length=self.min_token_length,
        )

    # -- sklearn API ---------------------------------------------------------
    def fit(self, X: Any, y: Any = None) -> TextCombiner:
        self.n_features_in_ = self._n_columns(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        config = self.to_config()
        titles, bodies = self._extract_fields(X)
        combined = [
            combine_fields(normalize_text(t, config), normalize_text(b, config), config)
            for t, b in zip(titles, bodies, strict=True)
        ]
        return np.asarray(combined, dtype=object)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.asarray(["combined_text"], dtype=object)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _n_columns(X: Any) -> int:
        if isinstance(X, pd.DataFrame):
            return int(X.shape[1])
        return 1

    def _extract_fields(self, X: Any) -> tuple[list[str], list[str]]:
        """Return ``(titles, bodies)`` for every supported input container."""
        if isinstance(X, pd.DataFrame):
            titles = self._column_or_empty(X, self.title_column)
            bodies = self._column_or_empty(X, self.text_column)
            return titles, bodies
        if isinstance(X, pd.Series):
            return ["" for _ in range(len(X))], [str(v) for v in X.tolist()]
        if isinstance(X, Mapping):
            return self._extract_fields(pd.DataFrame(X))
        if isinstance(X, str):
            return [""], [X]
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return self._extract_fields(pd.DataFrame(X))
        if isinstance(X, Iterable):
            values = list(X)
            if values and isinstance(values[0], Mapping):
                return self._extract_fields(pd.DataFrame(values))
            if values and isinstance(values[0], str):
                return ["" for _ in values], [str(v) for v in values]
        raise TypeError(
            "TextCombiner expects a DataFrame with title/text columns, a Series, a list of "
            f"strings, a list of dicts or a dict of columns; got {type(X).__name__}"
        )

    @staticmethod
    def _column_or_empty(frame: pd.DataFrame, column: str) -> list[str]:
        """Return a string column, mapping any missing marker to an empty string.

        ``str(pd.NA)`` would produce the literal text ``"nan"`` or ``"<NA>"`` and
        silently inject that token into the document, so missing values are
        detected explicitly rather than coerced.
        """
        if column not in frame.columns:
            return ["" for _ in range(len(frame))]
        return ["" if pd.isna(value) else str(value) for value in frame[column].tolist()]


def build_preprocessor(config: PreprocessingConfig, **kwargs: Any) -> TextCombiner:
    """Public factory used by the training pipeline and the tests."""
    return TextCombiner.from_config(config, **kwargs)
