"""Fake News Detection System.

A leakage-aware text classification pipeline for the ISOT "Fake and Real News"
dataset, together with a reproducible training/evaluation workflow, a FastAPI
inference service and a Streamlit frontend.

The package is intentionally explicit about what it does *not* do: it
classifies the writing style of a document relative to a labelled corpus. It
does not verify facts. See ``docs/MODEL_CARD.md``.
"""

from __future__ import annotations

__all__ = ["__version__", "get_version"]

__version__ = "0.1.0"


def get_version() -> str:
    """Return the installed package version (falls back to the source version)."""
    try:  # pragma: no cover - depends on installation mode
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("fake-news-detection")
        except PackageNotFoundError:
            return __version__
    except Exception:  # pragma: no cover - extremely defensive
        return __version__
