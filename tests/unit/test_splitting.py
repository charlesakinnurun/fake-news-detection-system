"""Unit tests for leakage-safe splitting and near-duplicate clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fake_news_detection.config import (
    DataConfig,
    DatasetConfig,
    DataSourceConfig,
    NearDuplicateConfig,
    SplittingConfig,
)
from fake_news_detection.data.splitting import (
    NearDuplicateIndex,
    _union_group_ids,
    assert_no_group_leakage,
    assign_duplicate_clusters,
    grouped_stratified_split,
    leakage_audit,
    make_splits,
)


def _frame(labels: list[str], bodies: list[str], groups: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(len(labels))],
            "title": [f"title {i}" for i in range(len(labels))],
            "text": bodies,
            "label": pd.Series(labels, dtype="string"),
            "subject": "news",
            "date": pd.NaT,
            "source": "synthetic.csv",
            "match_key": pd.Series(bodies, dtype="string").str.lower(),
            "row_length": [len(body) for body in bodies],
        }
    )
    if groups is not None:
        frame["dup_cluster"] = groups
    return frame


class TestMinHashClustering:
    def test_identical_documents_cluster_together(self) -> None:
        texts = [
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta epsilon",
            "zulu yankee xray",
        ]
        index = NearDuplicateIndex(NearDuplicateConfig(shingle_size=3, num_hashes=32, num_bands=8))
        labels, stats = index.fit(texts).clusters()
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]
        assert stats["multi_document_clusters"] == 1

    def test_near_duplicates_are_detected_above_the_threshold(self) -> None:
        base = "the quick brown fox jumps over the lazy dog and runs away quickly"
        variant = base + " indeed"
        index = NearDuplicateIndex(
            NearDuplicateConfig(shingle_size=3, num_hashes=32, num_bands=8, jaccard_threshold=0.5)
        )
        labels, _ = index.fit([base, variant]).clusters()
        assert labels[0] == labels[1]

    def test_unrelated_documents_stay_separate(self) -> None:
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "completely different content about municipal budgets and roads",
            "sports results from the weekend league matches",
        ]
        index = NearDuplicateIndex(NearDuplicateConfig(shingle_size=3, num_hashes=32, num_bands=8))
        labels, _ = index.fit(texts).clusters()
        assert len(set(labels.tolist())) == 3

    def test_jaccard_is_exact(self) -> None:
        left = np.array([1, 2, 3, 4], dtype=np.uint64)
        right = np.array([3, 4, 5, 6], dtype=np.uint64)
        assert NearDuplicateIndex.jaccard(left, right) == pytest.approx(2 / 6)

    def test_empty_documents_are_isolated(self) -> None:
        index = NearDuplicateIndex(NearDuplicateConfig(shingle_size=3, num_hashes=16, num_bands=4))
        labels, _ = index.fit(["", "", "some real content here for shingling"]).clusters()
        assert len(set(labels.tolist())) == 3

    def test_candidate_pairs_requires_fit(self) -> None:
        with pytest.raises(RuntimeError):
            NearDuplicateIndex(NearDuplicateConfig()).candidate_pairs()


class TestUnionFind:
    def test_transitive_union_of_partitions(self) -> None:
        exact = np.array([0, 0, 1, 1], dtype=np.int64)
        near = np.array([0, 1, 1, 2], dtype=np.int64)
        merged = _union_group_ids(exact, near)
        # row0 ~ row1 (exact), row1 ~ row2 (near), row2 ~ row3 (exact) => all together
        assert len(set(merged.tolist())) == 1

    def test_disjoint_partitions_are_preserved(self) -> None:
        merged = _union_group_ids(np.array([0, 1, 1]), np.array([0, 1, 2]))
        assert merged[0] != merged[1] == merged[2]

    def test_no_arrays(self) -> None:
        assert _union_group_ids().size == 0


class TestAssignDuplicateClusters:
    def test_near_duplicates_land_in_one_cluster(self) -> None:
        bodies = [
            "breaking news the government hides the truth about water supplies today",
            "breaking news the government hides the truth about water supplies today!",
            "completely unrelated reporting on municipal transport budgets for next year",
        ]
        frame = _frame(["FAKE", "FAKE", "REAL"], bodies)
        assigned, stats = assign_duplicate_clusters(
            frame, NearDuplicateConfig(shingle_size=3, num_hashes=32, num_bands=8)
        )
        assert assigned["dup_cluster"].nunique() == 2
        assert stats["combined_multi_document_clusters"] == 1

    def test_disabled_configuration_falls_back_to_exact_groups(self) -> None:
        frame = _frame(["FAKE", "FAKE"], ["same body text here", "same body text here"])
        assigned, stats = assign_duplicate_clusters(frame, NearDuplicateConfig(enabled=False))
        assert assigned["dup_cluster"].nunique() == 1
        assert stats["enabled"] is False


class TestGroupedSplit:
    @pytest.fixture()
    def clustered(self) -> pd.DataFrame:
        labels: list[str] = []
        bodies: list[str] = []
        groups: list[int] = []
        for group in range(60):
            label = "FAKE" if group % 2 == 0 else "REAL"
            for member in range(2):
                labels.append(label)
                bodies.append(f"group {group} member {member} " + "filler " * (5 + group % 3))
                groups.append(group)
        return _frame(labels, bodies, groups)

    def test_no_group_crosses_a_split_boundary(self, clustered: pd.DataFrame) -> None:
        splits = grouped_stratified_split(
            clustered,
            label_column="label",
            group_column="dup_cluster",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=1,
        )
        assert_no_group_leakage(splits, "dup_cluster")
        audit = leakage_audit(splits)
        assert audit["passed"] is True

    def test_label_balance_is_preserved(self, clustered: pd.DataFrame) -> None:
        splits = grouped_stratified_split(
            clustered,
            label_column="label",
            group_column="dup_cluster",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=1,
        )
        for name, frame in splits.items():
            share = (frame["label"] == "FAKE").mean()
            assert 0.4 <= share <= 0.6, f"{name} is imbalanced: {share}"

    def test_split_sizes_are_respected(self, clustered: pd.DataFrame) -> None:
        splits = grouped_stratified_split(
            clustered,
            label_column="label",
            group_column="dup_cluster",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=1,
        )
        assert len(splits["train"]) > len(splits["validation"])
        assert abs(len(splits["validation"]) - len(splits["test"])) <= 5

    def test_deterministic_for_a_fixed_seed(self, clustered: pd.DataFrame) -> None:
        first = grouped_stratified_split(
            clustered,
            label_column="label",
            group_column="dup_cluster",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=3,
        )
        second = grouped_stratified_split(
            clustered,
            label_column="label",
            group_column="dup_cluster",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=3,
        )
        assert first["train"]["row_id"].tolist() == second["train"]["row_id"].tolist()

    def test_different_seeds_change_the_assignment(self, clustered: pd.DataFrame) -> None:
        first = grouped_stratified_split(
            clustered,
            label_column="label",
            group_column="dup_cluster",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=1,
        )
        second = grouped_stratified_split(
            clustered,
            label_column="label",
            group_column="dup_cluster",
            ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=99,
        )
        assert first["train"]["row_id"].tolist() != second["train"]["row_id"].tolist()


class TestMakeSplits:
    def _config(self, strategy: str) -> DataConfig:
        return DataConfig(
            seed=5,
            dataset=DatasetConfig(
                name="unit",
                labels=["FAKE", "REAL"],
                positive_label="FAKE",
                sources=[DataSourceConfig(path="unused.csv", label="FAKE")],
            ),
            splitting=SplittingConfig(
                strategy=strategy,  # type: ignore[arg-type]
                near_duplicate=NearDuplicateConfig(shingle_size=3, num_hashes=16, num_bands=4),
            ),
        )

    def _corpus(self) -> pd.DataFrame:
        """100 distinct documents plus 20 duplicated pairs (20 leakage groups)."""
        labels, bodies = [], []
        for index in range(100):
            label = "FAKE" if index % 2 == 0 else "REAL"
            labels.append(label)
            # Every token is unique to its document, so unrelated documents have
            # ~zero shingle overlap and only the deliberate copies cluster.
            bodies.append(f"unique doc {index} " + " ".join(f"t{index}_{k}" for k in range(20)))
        # 20 near-duplicate pairs spread across the corpus.
        for index in range(0, 100, 5):
            labels.append(labels[index])
            bodies.append(bodies[index].replace(".", ","))
        return _frame(labels, bodies)

    def test_grouped_strategy_passes_the_leakage_audit(self) -> None:
        result = make_splits(self._corpus(), self._config("grouped_stratified"))
        assert result.leakage_audit["passed"] is True
        assert result.summary["train"]["rows"] > 0

    def test_stratified_strategy_reports_leakage_without_raising(self) -> None:
        result = make_splits(self._corpus(), self._config("stratified"))
        # The point of this strategy is to *demonstrate* leakage, so it must not raise.
        assert result.leakage_audit["checks"]["duplicate_cluster_overlap_total"] >= 1

    def test_temporal_strategy_orders_splits_chronologically(self) -> None:
        frame = self._corpus()
        frame["date"] = pd.date_range("2018-01-01", periods=len(frame), freq="D")
        frame["match_key"] = pd.Series(frame["text"]).str.lower()
        result = make_splits(frame, self._config("temporal"), strategy="temporal")
        assert result.train["date"].max() <= result.validation["date"].min()
        assert result.validation["date"].max() <= result.test["date"].min()

    def test_temporal_requires_parsed_dates(self) -> None:
        with pytest.raises(ValueError, match="parsed date"):
            make_splits(self._corpus(), self._config("temporal"), strategy="temporal")

    def test_unknown_group_leakage_is_detected(self) -> None:
        splits = {
            "train": _frame(["FAKE"], ["a"], [1]),
            "test": _frame(["FAKE"], ["b"], [1]),
        }
        with pytest.raises(AssertionError):
            assert_no_group_leakage(splits, "dup_cluster")

    def test_summary_is_json_serialisable(self) -> None:
        import json

        result = make_splits(self._corpus(), self._config("grouped_stratified"))
        json.dumps(result.as_dict())
