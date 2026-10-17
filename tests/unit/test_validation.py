"""Unit tests for validation, exclusion accounting and artifact detection."""

from __future__ import annotations

import pandas as pd
import pytest

from fake_news_detection.config import (
    DataConfig,
    DatasetConfig,
    DataSourceConfig,
    ValidationConfig,
)
from fake_news_detection.data.validation import (
    exclusion_summary,
    normalise_for_matching,
    validate_and_clean,
)


def _config(**overrides: object) -> DataConfig:
    validation: dict[str, object] = {
        "min_text_chars": 10,
        "min_total_chars": 20,
        "max_reported_examples": 3,
    }
    validation.update(overrides)
    return DataConfig(
        seed=1,
        dataset=DatasetConfig(
            name="unit",
            labels=["FAKE", "REAL"],
            positive_label="FAKE",
            sources=[DataSourceConfig(path="unused.csv", label="FAKE")],
        ),
        validation=ValidationConfig(**validation),  # type: ignore[arg-type]
    )


def _frame(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    for column in ("title", "text", "subject", "date_raw", "label", "source"):
        if column not in frame:
            frame[column] = ""
    frame["label"] = frame["label"].astype("string")
    frame["date"] = pd.NaT
    return frame


class TestExclusionAccounting:
    def test_every_excluded_row_has_exactly_one_reason(self) -> None:
        frame = _frame(
            [
                {
                    "row_id": "1",
                    "title": "Good",
                    "text": "A long enough body for validation.",
                    "label": "REAL",
                },
                {"row_id": "2", "title": "", "text": "", "label": "REAL"},
                {"row_id": "3", "title": "Short", "text": "tiny", "label": "FAKE"},
                {
                    "row_id": "4",
                    "title": "Weird",
                    "text": "A long enough body for validation.",
                    "label": "MAYBE",
                },
            ]
        )
        clean, report = validate_and_clean(frame, _config())
        assert len(clean) == 1
        assert report.rows_in == 4
        assert report.rows_excluded == 3
        assert sum(report.exclusion_counts.values()) == 3
        assert set(report.exclusion_counts) <= {"empty_content", "short_body", "invalid_label"}

    def test_counts_sum_to_rows_removed(self) -> None:
        frame = _frame(
            [
                {"row_id": str(i), "title": f"T{i}", "text": "x" * 50, "label": "REAL"}
                for i in range(5)
            ]
        )
        _, report = validate_and_clean(frame, _config())
        assert report.rows_in - report.rows_out == sum(report.exclusion_counts.values())

    def test_examples_are_attached_but_truncated(self) -> None:
        frame = _frame(
            [
                {
                    "row_id": "1",
                    "title": "A" * 200,
                    "text": "short",
                    "label": "FAKE",
                }
            ]
        )
        _, report = validate_and_clean(frame, _config(max_reported_examples=5))
        rule = next(r for r in report.rules if r.rule == "short_body")
        assert rule.affected == 1
        assert len(rule.examples[0]["title"]) <= 111


class TestDuplicatesAndConflicts:
    def test_exact_duplicates_are_dropped_keeping_the_first(self) -> None:
        frame = _frame(
            [
                {
                    "row_id": "1",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "REAL",
                },
                {
                    "row_id": "2",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "REAL",
                },
            ]
        )
        clean, report = validate_and_clean(frame, _config())
        assert len(clean) == 1
        assert clean["row_id"].iloc[0] == "1"
        assert report.duplicate_rows_exact == 1

    def test_duplicates_differing_only_in_punctuation_still_collapse(self) -> None:
        frame = _frame(
            [
                {
                    "row_id": "1",
                    "title": "Hello world",
                    "text": "Body, text! here.",
                    "label": "REAL",
                },
                {"row_id": "2", "title": "hello  world", "text": "body text here", "label": "REAL"},
            ]
        )
        clean, _ = validate_and_clean(frame, _config())
        assert len(clean) == 1

    def test_conflicting_labels_are_dropped_under_the_default_policy(self) -> None:
        frame = _frame(
            [
                {
                    "row_id": "1",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "REAL",
                },
                {
                    "row_id": "2",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "FAKE",
                },
            ]
        )
        clean, report = validate_and_clean(frame, _config(conflicting_label_policy="drop"))
        assert clean.empty
        assert report.conflicting_label_groups == 1
        assert report.conflicting_label_rows == 2

    def test_majority_policy_keeps_the_agreed_label(self) -> None:
        frame = _frame(
            [
                {
                    "row_id": "1",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "REAL",
                },
                {
                    "row_id": "2",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "REAL",
                },
                {
                    "row_id": "3",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "FAKE",
                },
            ]
        )
        clean, report = validate_and_clean(
            frame, _config(conflicting_label_policy="majority", majority_min_agreement=0.6)
        )
        assert set(clean["label"]) == {"REAL"}
        assert report.conflicting_label_groups == 1

    def test_deduplication_can_be_disabled(self) -> None:
        frame = _frame(
            [
                {
                    "row_id": "1",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "REAL",
                },
                {
                    "row_id": "2",
                    "title": "Same",
                    "text": "Body text repeated here.",
                    "label": "REAL",
                },
            ]
        )
        clean, _ = validate_and_clean(frame, _config(drop_exact_duplicates=False))
        assert len(clean) == 2


class TestLengthRules:
    def test_long_documents_are_excluded(self) -> None:
        frame = _frame([{"row_id": "1", "title": "T", "text": "x" * 5000, "label": "REAL"}])
        clean, report = validate_and_clean(frame, _config(max_text_chars=1000))
        assert clean.empty
        assert "text_too_long" in report.exclusion_counts

    def test_length_statistics_are_reported_per_class(self) -> None:
        records = []
        for index in range(10):
            records.append(
                {"row_id": f"r{index}", "title": "T", "text": "a" * 100, "label": "REAL"}
            )
            records.append(
                {"row_id": f"f{index}", "title": "T", "text": "b" * 300, "label": "FAKE"}
            )
        _, report = validate_and_clean(_frame(records), _config())
        assert report.text_length_by_class["REAL"]["max"] == 100
        assert report.text_length_by_class["FAKE"]["max"] == 300


class TestArtifactFindings:
    def test_wire_marker_concentration_is_flagged(self) -> None:
        records = []
        for index in range(10):
            records.append(
                {
                    "row_id": f"r{index}",
                    "title": "T",
                    "text": "WASHINGTON (Reuters) - Something happened today in the capital.",
                    "label": "REAL",
                }
            )
            records.append(
                {
                    "row_id": f"f{index}",
                    "title": "T",
                    "text": "You will not believe what happened next, share before it is deleted.",
                    "label": "FAKE",
                }
            )
        _, report = validate_and_clean(_frame(records), _config())
        kinds = {finding.get("finding") for finding in report.artifact_findings}
        assert "wire_marker_imbalance" in kinds

    def test_disjoint_subjects_are_flagged(self) -> None:
        records = [
            {
                "row_id": "1",
                "title": "T",
                "text": "a" * 100,
                "label": "REAL",
                "subject": "worldnews",
            },
            {
                "row_id": "2",
                "title": "T",
                "text": "b" * 100,
                "label": "FAKE",
                "subject": "politics",
            },
        ]
        _, report = validate_and_clean(_frame(records), _config())
        overlap = next(
            finding
            for finding in report.artifact_findings
            if finding.get("finding") == "subject_vocabulary_overlap"
        )
        assert overlap["intersection"] == []


class TestMatching:
    def test_normalise_for_matching_is_case_and_punctuation_insensitive(self) -> None:
        series = pd.Series(["Hello, World!", "hello world", "  HELLO   world  "])
        assert normalise_for_matching(series).nunique() == 1

    def test_exclusion_summary_counts_reasons(self) -> None:
        reasons = pd.Series(["a", None, "b", "a", None], dtype=object)
        assert exclusion_summary(reasons) == {"a": 2, "b": 1}

    def test_report_is_json_serialisable(self) -> None:
        import json

        frame = _frame([{"row_id": "1", "title": "T", "text": "x" * 100, "label": "REAL"}])
        _, report = validate_and_clean(frame, _config())
        json.dumps(report.to_dict())


@pytest.mark.parametrize("policy", ["drop", "majority"])
def test_policy_variants_never_crash_on_empty_input(policy: str) -> None:
    empty = _frame([])
    clean, report = validate_and_clean(empty, _config(conflicting_label_policy=policy))
    assert clean.empty
    assert report.rows_in == 0
