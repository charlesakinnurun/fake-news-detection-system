"""Tests for path- and format-related helpers."""

from __future__ import annotations

from pathlib import Path

from fake_news_detection.utils import display_path, human_bytes


def test_display_path_returns_relative_path_inside_base(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "result.json"
    assert display_path(target, tmp_path) == str(Path("reports") / "result.json")


def test_display_path_falls_back_for_paths_outside_base(tmp_path: Path) -> None:
    outside = tmp_path.parent / "elsewhere" / "result.json"
    assert display_path(outside, tmp_path) == str(outside)


def test_display_path_without_base_returns_full_path(tmp_path: Path) -> None:
    target = tmp_path / "artifacts" / "model.joblib"
    assert display_path(target) == str(target)


def test_human_bytes_uses_binary_units() -> None:
    assert human_bytes(0) == "0 B"
    assert human_bytes(1024) == "1.00 KiB"
    assert human_bytes(5 * 1024 * 1024) == "5.00 MiB"
