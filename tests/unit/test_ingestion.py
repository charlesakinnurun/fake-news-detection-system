"""Unit tests for the ingestion layer (schema mapping, labels, dates)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fake_news_detection.config import (
    ColumnMapping,
    DataConfig,
    DatasetConfig,
    DataSourceConfig,
)
from fake_news_detection.data.ingestion import (
    CANONICAL_COLUMNS,
    load_dataset,
    load_source,
    parse_dates,
)

FIXTURE = Path(__file__).resolve().parents[1] / "test_data" / "synthetic_news.csv"


def _config(
    path: Path, *, label_column: str | None = None, label_map: dict | None = None
) -> DataConfig:
    return DataConfig(
        seed=1,
        dataset=DatasetConfig(
            name="fixture",
            version="0.1.0",
            labels=["FAKE", "REAL"],
            positive_label="FAKE",
            label_map=label_map or {},
            sources=[DataSourceConfig(path=str(path), label="REAL", description="fixture")],
            columns=ColumnMapping(
                title="headline",
                text="body",
                subject="region",
                date="published",
                label=label_column,
            ),
            date_formats=["%Y-%m-%d", "%B %d, %Y"],
        ),
    )


class TestColumnMapping:
    def test_columns_are_renamed_to_the_canonical_schema(self) -> None:
        frame, report = load_source(_config(FIXTURE).dataset.sources[0], _config(FIXTURE))
        assert list(frame.columns) == list(CANONICAL_COLUMNS)
        assert report.rows == 24
        assert report.columns == ["headline", "body", "is_fake", "region", "published"]
        assert frame["title"].iloc[0] != ""
        assert frame["subject"].iloc[0] in {
            "europe",
            "asia",
            "africa",
            "north-america",
            "south-america",
        }

    def test_source_label_is_applied_when_no_label_column_is_configured(self) -> None:
        frame, _ = load_source(_config(FIXTURE).dataset.sources[0], _config(FIXTURE))
        assert set(frame["label"]) == {"REAL"}

    def test_label_column_is_normalised_through_label_map(self) -> None:
        config = _config(
            FIXTURE, label_column="is_fake", label_map={"TRUE": "FAKE", "FALSE": "REAL"}
        )
        frame, _ = load_source(config.dataset.sources[0], config)
        assert set(frame["label"]) == {"FAKE", "REAL"}
        assert (frame["label"] == "FAKE").sum() == 12

    def test_dates_are_parsed(self) -> None:
        frame, _ = load_source(_config(FIXTURE).dataset.sources[0], _config(FIXTURE))
        assert frame["date"].notna().all()
        assert str(frame["date"].min().date()) == "2019-03-04"

    def test_row_ids_are_unique(self) -> None:
        frame, _ = load_source(_config(FIXTURE).dataset.sources[0], _config(FIXTURE))
        assert frame["row_id"].is_unique

    def test_source_file_is_recorded(self) -> None:
        frame, _ = load_source(_config(FIXTURE).dataset.sources[0], _config(FIXTURE))
        assert set(frame["source"]) == {"synthetic_news.csv"}


class TestFailureModes:
    def test_missing_file_gives_actionable_error(self, tmp_path: Path) -> None:
        config = _config(tmp_path / "nope.csv")
        with pytest.raises(FileNotFoundError, match=r"download_data\.py"):
            load_source(config.dataset.sources[0], config)

    def test_missing_required_column_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")
        config = _config(path)
        with pytest.raises(ValueError, match="required column"):
            load_source(config.dataset.sources[0], config)

    def test_undeclared_label_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_label.csv"
        path.write_text("headline,body\nh,b\n", encoding="utf-8")
        config = _config(path)
        config.dataset.sources[0].label = "MAYBE"
        with pytest.raises(ValueError, match="not declared"):
            load_dataset(config)

    def test_non_utf8_bytes_trigger_a_warning_not_a_crash(self, tmp_path: Path) -> None:
        path = tmp_path / "latin.csv"
        path.write_bytes("headline,body\nCafé,Some body text\n".encode("latin-1"))
        config = _config(path)
        frame, report = load_dataset(config)
        assert report.sources[0].encoding_fallback is True
        assert len(frame) == 1
        assert any("latin-1" in warning for warning in report.warnings)


class TestDateParsing:
    def test_mixed_formats_are_handled(self) -> None:
        series = pd.Series(["December 31, 2017", "2018-01-05", "not a date"])
        parsed = parse_dates(series, ["%B %d, %Y", "%Y-%m-%d"])
        assert parsed.notna().sum() == 2
        assert str(parsed.iloc[0].date()) == "2017-12-31"
        assert pd.isna(parsed.iloc[2])

    def test_trailing_whitespace_is_tolerated(self) -> None:
        parsed = parse_dates(pd.Series(["December 31, 2017 "]), ["%B %d, %Y"])
        assert parsed.notna().all()

    def test_empty_series(self) -> None:
        assert parse_dates(pd.Series([], dtype="object"), ["%Y"]).empty


class TestDatasetAssembly:
    def test_fingerprint_changes_with_content(self, tmp_path: Path) -> None:
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text("headline,body\nh,body text here\n", encoding="utf-8")
        path_b.write_text("headline,body\nh,body text here!\n", encoding="utf-8")
        _, report_a = load_dataset(_config(path_a))
        _, report_b = load_dataset(_config(path_b))
        assert report_a.fingerprint != report_b.fingerprint

    def test_fingerprint_is_stable(self) -> None:
        _, first = load_dataset(_config(FIXTURE))
        _, second = load_dataset(_config(FIXTURE))
        assert first.fingerprint == second.fingerprint

    def test_report_serialises_to_json(self) -> None:
        import json

        _, report = load_dataset(_config(FIXTURE))
        json.dumps(report.to_dict())
