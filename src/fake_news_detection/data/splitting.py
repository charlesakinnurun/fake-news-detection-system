"""Leakage-safe dataset splitting.

Two-stage strategy:

1. **Cluster first.** Articles are grouped by exact normalised content and by
   approximate content (MinHash + LSH over token 4-grams, verified with an
   exact Jaccard test). Every article belongs to exactly one cluster.
2. **Split by cluster.** Clusters are assigned to train/validation/test as
   indivisible units, stratified by label, so no article (or a near copy of it)
   can appear in more than one split.

The splitter never looks at the test split to make a decision; it only decides
which clusters go where, using the label and the cluster size.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from fake_news_detection.config import DataConfig, NearDuplicateConfig
from fake_news_detection.utils import timer

LOGGER = logging.getLogger(__name__)

__all__ = [
    "NearDuplicateIndex",
    "SplitResult",
    "assert_no_group_leakage",
    "grouped_stratified_split",
    "make_splits",
    "split_summary",
]

# 2**31 - 1 is prime; keeping hashes and multipliers below 2**31 guarantees that
# ``a * h + b`` cannot overflow a uint64 accumulator.
_MOD = np.uint64(2**31 - 1)


class NearDuplicateIndex:
    """MinHash + LSH index for near-duplicate detection over short documents.

    Parameters are drawn from :class:`~fake_news_detection.config.NearDuplicateConfig`.
    32-bit ``crc32`` shingle hashes are used deliberately: the expected number of
    collisions over a corpus this size is negligible (a few thousand across
    millions of shingles) and it keeps the index build well under a minute on
    CPU-only hardware.
    """

    def __init__(self, config: NearDuplicateConfig) -> None:
        self.config = config
        self._signatures: np.ndarray | None = None
        self._shingle_hashes: list[np.ndarray] = []
        self._multipliers: np.ndarray | None = None
        self._offsets: np.ndarray | None = None

    # -- shingles ------------------------------------------------------------
    def _shingle_hashes_for(self, text: str) -> np.ndarray:
        tokens = text.split()
        size = self.config.shingle_size
        if len(tokens) < size:
            payload = [" ".join(tokens)] if tokens else []
        else:
            payload = [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]
        if len(payload) > self.config.max_shingles_per_doc:
            # Deterministic subsample: keep an evenly spaced prefix.
            step = max(1, len(payload) // self.config.max_shingles_per_doc)
            payload = payload[::step][: self.config.max_shingles_per_doc]
        if not payload:
            return np.zeros(0, dtype=np.uint64)
        # Sort so that identical token sets always yield the same array layout.
        encoded = sorted(zlib.crc32(item.encode("utf-8")) & 0xFFFFFFFF for item in payload)
        return np.asarray(encoded, dtype=np.uint64)

    # -- fit -----------------------------------------------------------------
    def fit(self, texts: list[str]) -> NearDuplicateIndex:
        """Build signatures and the candidate pair list."""
        rng = np.random.default_rng(self.config.num_hashes)
        self._multipliers = rng.integers(
            1, 2**31 - 1, size=self.config.num_hashes, dtype=np.int64
        ).astype(np.uint64)
        self._offsets = rng.integers(
            0, 2**31 - 1, size=self.config.num_hashes, dtype=np.int64
        ).astype(np.uint64)

        signatures = np.zeros((len(texts), self.config.num_hashes), dtype=np.uint64)
        self._shingle_hashes = []
        for index, text in enumerate(texts):
            hashes = self._shingle_hashes_for(text)
            self._shingle_hashes.append(hashes)
            if hashes.size == 0:
                signatures[index, :] = _MOD - 1
                continue
            permuted = (
                self._multipliers[None, :] * hashes[:, None] + self._offsets[None, :]
            ) % _MOD
            signatures[index, :] = permuted.min(axis=0)
        self._signatures = signatures
        return self

    # -- candidate generation ------------------------------------------------
    def candidate_pairs(self, *, max_bucket_size: int = 500) -> list[tuple[int, int]]:
        """Return candidate index pairs sharing at least one LSH band."""
        if self._signatures is None:
            raise RuntimeError("NearDuplicateIndex.fit must be called before candidate_pairs")
        rows = self.config.num_hashes // self.config.num_bands
        buckets: dict[tuple[int, bytes], list[int]] = {}
        for band in range(self.config.num_bands):
            block = self._signatures[:, band * rows : (band + 1) * rows]
            keys = block.tobytes()
            width = rows * 8
            for index in range(block.shape[0]):
                key = (band, keys[index * width : (index + 1) * width])
                buckets.setdefault(key, []).append(index)

        pairs: set[tuple[int, int]] = set()
        skipped = 0
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > max_bucket_size:
                skipped += 1
                continue
            for left in range(len(members)):
                for right in range(left + 1, len(members)):
                    a, b = members[left], members[right]
                    pairs.add((a, b) if a < b else (b, a))
        if skipped:
            LOGGER.warning(
                "skipped %d oversized LSH bucket(s) (>%d docs); lower shingle_size or "
                "raise num_bands if duplicate recall looks low",
                skipped,
                max_bucket_size,
            )
        return sorted(pairs)

    @staticmethod
    def jaccard(left: np.ndarray, right: np.ndarray) -> float:
        """Exact Jaccard similarity between two sorted shingle-hash arrays."""
        if left.size == 0 or right.size == 0:
            return 0.0
        union = np.union1d(left, right).size
        if union == 0:
            return 0.0
        intersection = np.intersect1d(left, right, assume_unique=True).size
        return float(intersection) / float(union)

    # -- clustering ----------------------------------------------------------
    def clusters(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Union-find over verified candidate pairs; returns labels and stats."""
        if self._signatures is None:
            raise RuntimeError("NearDuplicateIndex.fit must be called before clusters")
        n_docs = self._signatures.shape[0]
        parent = np.arange(n_docs, dtype=np.int64)

        def find(node: int) -> int:
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != root:
                parent[node], node = root, parent[node]
            return root

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)

        pairs = self.candidate_pairs()
        matched = 0
        for left, right in pairs:
            if self.jaccard(self._shingle_hashes[left], self._shingle_hashes[right]) >= (
                self.config.jaccard_threshold
            ):
                union(left, right)
                matched += 1

        roots = np.asarray([find(index) for index in range(n_docs)], dtype=np.int64)
        _, labels = np.unique(roots, return_inverse=True)
        sizes = np.bincount(labels)
        stats = {
            "documents": int(n_docs),
            "candidate_pairs": len(pairs),
            "verified_pairs": int(matched),
            "clusters": int(sizes.size),
            "multi_document_clusters": int((sizes > 1).sum()),
            "documents_in_multi_clusters": int(sizes[sizes > 1].sum()),
            "largest_cluster": int(np.max(sizes)) if sizes.size else 0,
            "shingle_size": self.config.shingle_size,
            "jaccard_threshold": self.config.jaccard_threshold,
            "num_hashes": self.config.num_hashes,
        }
        return labels.astype(np.int64), stats


def _union_group_ids(*group_arrays: np.ndarray) -> np.ndarray:
    """Merge several integer group partitions into their transitive union.

    Combining partitions is *not* the same as concatenating their ids: a row may
    share an exact-content group with one row and a near-duplicate group with
    another, and all three belong in the same leakage cluster. Union-find makes
    that transitive closure explicit.
    """
    if not group_arrays:
        return np.zeros(0, dtype=np.int64)
    n_rows = len(group_arrays[0])
    parent = list(range(n_rows))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for array in group_arrays:
        first_seen: dict[int, int] = {}
        for index, raw in enumerate(np.asarray(array, dtype=np.int64).tolist()):
            if raw in first_seen:
                left, right = find(first_seen[raw]), find(index)
                if left != right:
                    parent[max(left, right)] = min(left, right)
            else:
                first_seen[raw] = index

    roots = np.asarray([find(index) for index in range(n_rows)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def assign_duplicate_clusters(
    frame: pd.DataFrame, config: NearDuplicateConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a ``dup_cluster`` column that groups exact and near duplicates.

    Two documents end up in the same cluster when they are exactly equal after
    normalisation *or* when their token 4-gram Jaccard similarity clears the
    configured threshold. Clusters are the unit of splitting, so no copy of an
    article can straddle train/validation/test.
    """
    frame = frame.copy()
    exact_codes, _ = pd.factorize(frame["match_key"].astype(str), sort=True)
    exact_codes = exact_codes.astype(np.int64)
    if len(frame) == 0 or not config.enabled:
        frame["dup_cluster"] = exact_codes
        return frame, {
            "enabled": False,
            "documents": len(frame),
            "clusters": len(np.unique(exact_codes)),
            "exact_duplicate_groups": len(np.unique(exact_codes)),
        }

    with timer() as elapsed:
        index = NearDuplicateIndex(config).fit(frame["match_key"].astype(str).tolist())
        near_labels, stats = index.clusters()

    combined = _union_group_ids(exact_codes, near_labels.astype(np.int64))
    frame["dup_cluster"] = combined
    sizes = np.bincount(combined)
    stats.update(
        {
            "enabled": True,
            "exact_duplicate_groups": len(np.unique(exact_codes)),
            "combined_clusters": int(sizes.size),
            "combined_multi_document_clusters": int((sizes > 1).sum()),
            "combined_documents_in_multi_clusters": int(sizes[sizes > 1].sum()),
            "documents": len(frame),
            "seconds": round(elapsed.seconds, 3),
        }
    )
    LOGGER.info(
        "duplicate clustering: %d docs -> %d clusters (%d multi-doc covering %d docs, %.1fs)",
        stats["documents"],
        stats["combined_clusters"],
        stats["combined_multi_document_clusters"],
        stats["combined_documents_in_multi_clusters"],
        stats["seconds"],
    )
    return frame, stats


def _assign_clusters_to_splits(
    frame: pd.DataFrame,
    *,
    label_column: str,
    group_column: str,
    ratios: dict[str, float],
    seed: int,
) -> pd.Series:
    """Greedy, label-stratified assignment of whole clusters to splits."""
    rng = np.random.default_rng(seed)
    splits = list(ratios)
    labels = sorted(frame[label_column].astype(str).unique().tolist())

    # Each cluster is homogeneous in practice; assert it to catch surprises.
    cluster_labels = frame.groupby(group_column)[label_column].nunique()
    if (cluster_labels > 1).any():
        LOGGER.warning(
            "%d duplicate cluster(s) contain more than one label; they are assigned using the "
            "majority label and their presence is reported in the leakage audit",
            int((cluster_labels > 1).sum()),
        )
    cluster_majority = frame.groupby(group_column)[label_column].agg(
        lambda values: values.value_counts().idxmax()
    )
    cluster_size = frame.groupby(group_column).size()

    label_totals = frame[label_column].value_counts().to_dict()
    targets = {
        (split, str(label)): ratios[split] * float(label_totals.get(label, 0))
        for split in splits
        for label in labels
    }
    assigned: dict[tuple[str, str], float] = dict.fromkeys(targets, 0.0)

    assignment: dict[Any, str] = {}
    for label in labels:
        members = [
            (cluster, int(size))
            for cluster, size in cluster_size.items()
            if str(cluster_majority.loc[cluster]) == label
        ]
        order: np.ndarray = rng.permutation(len(members))
        members = [members[i] for i in order]
        members.sort(key=lambda item: -item[1])
        for cluster, size in members:
            deficits = {
                split: targets[(split, label)] - assigned[(split, label)] for split in splits
            }
            choice = max(splits, key=lambda split: (deficits[split], rng.random()))
            assignment[cluster] = choice
            assigned[(choice, label)] += size

    return frame[group_column].map(assignment)


@dataclass(slots=True)
class SplitResult:
    """The three data splits plus everything needed to audit them."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    summary: dict[str, Any] = field(default_factory=dict)
    duplicate_stats: dict[str, Any] = field(default_factory=dict)
    leakage_audit: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "summary": self.summary,
            "duplicate_stats": self.duplicate_stats,
            "leakage_audit": self.leakage_audit,
        }

    def splits(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def grouped_stratified_split(
    frame: pd.DataFrame,
    *,
    label_column: str,
    group_column: str,
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split ``frame`` keeping every duplicate cluster intact and labels balanced."""
    assignment = _assign_clusters_to_splits(
        frame,
        label_column=label_column,
        group_column=group_column,
        ratios=ratios,
        seed=seed,
    )
    result: dict[str, pd.DataFrame] = {}
    for split in ratios:
        subset = frame.loc[(assignment == split).to_numpy()].copy()
        result[split] = subset.reset_index(drop=True)
    return result


def _stratified_split(
    frame: pd.DataFrame, *, label_column: str, ratios: dict[str, float], seed: int
) -> dict[str, pd.DataFrame]:
    """Plain stratified split (no grouping). Used only for ablation comparisons."""
    from sklearn.model_selection import train_test_split

    holdout_size = ratios["validation"] + ratios["test"]
    train, rest = train_test_split(
        frame,
        test_size=holdout_size,
        random_state=seed,
        stratify=frame[label_column],
        shuffle=True,
    )
    relative_test = ratios["test"] / holdout_size
    validation, test = train_test_split(
        rest,
        test_size=relative_test,
        random_state=seed,
        stratify=rest[label_column],
        shuffle=True,
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _temporal_split(
    frame: pd.DataFrame,
    *,
    ratios: dict[str, float],
    date_column: str = "date",
) -> dict[str, pd.DataFrame]:
    """Chronological split: earliest data trains, latest data tests."""
    dated = frame.dropna(subset=[date_column]).sort_values(date_column, kind="mergesort")
    undated = frame.loc[frame[date_column].isna()]
    if len(dated) < 100:
        raise ValueError(
            f"temporal split requested but only {len(dated)} row(s) have a parsed date; "
            "check dataset.date_formats in configs/data.yaml"
        )
    n_total = len(dated)
    n_test = round(n_total * ratios["test"])
    n_validation = round(n_total * ratios["validation"])
    n_train = n_total - n_test - n_validation
    parts = {
        "train": dated.iloc[:n_train],
        "validation": dated.iloc[n_train : n_train + n_validation],
        "test": dated.iloc[n_train + n_validation :],
    }
    if not undated.empty:
        LOGGER.warning(
            "%d row(s) without a parsed date are appended to the training split for the "
            "temporal strategy",
            len(undated),
        )
        parts["train"] = pd.concat([parts["train"], undated], ignore_index=True)
    return {name: part.reset_index(drop=True) for name, part in parts.items()}


def assert_no_group_leakage(splits: dict[str, pd.DataFrame], group_column: str) -> None:
    """Raise ``AssertionError`` if any group appears in more than one split."""
    seen: dict[Any, str] = {}
    for name, frame in splits.items():
        for group in frame[group_column].unique().tolist():
            if group in seen and seen[group] != name:
                raise AssertionError(
                    f"leakage: group {group!r} appears in both {seen[group]!r} and {name!r}"
                )
            seen[group] = name


def split_summary(splits: dict[str, pd.DataFrame], label_column: str) -> dict[str, Any]:
    """Per-split row counts, label counts and overlapping-content counts."""
    summary: dict[str, Any] = {}
    total = sum(len(frame) for frame in splits.values())
    for name, frame in splits.items():
        counts = {str(k): int(v) for k, v in frame[label_column].value_counts().items()}
        summary[name] = {
            "rows": len(frame),
            "share": round(len(frame) / max(total, 1), 4),
            "label_counts": counts,
            "label_shares": {k: round(v / max(len(frame), 1), 4) for k, v in counts.items()},
            "mean_document_chars": round(float(frame["row_length"].mean()), 1)
            if "row_length" in frame
            else None,
        }
    return summary


def leakage_audit(
    splits: dict[str, pd.DataFrame],
    *,
    group_column: str = "dup_cluster",
    match_column: str = "match_key",
) -> dict[str, Any]:
    """Independently verify that splitting did not leak content across splits."""
    audit: dict[str, Any] = {"group_column": group_column, "checks": {}}

    group_sets = {
        name: set(frame[group_column].tolist())
        for name, frame in splits.items()
        if group_column in frame
    }
    names = list(group_sets)
    overlaps: dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = group_sets[left] & group_sets[right]
            overlaps[f"{left}|{right}"] = len(shared)
    audit["checks"]["duplicate_cluster_overlap"] = overlaps
    audit["checks"]["duplicate_cluster_overlap_total"] = int(sum(overlaps.values()))

    if match_column in splits.get("train", pd.DataFrame()).columns:
        key_sets = {
            name: set(frame[match_column].astype(str).tolist()) for name, frame in splits.items()
        }
        exact_overlaps = {}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                exact_overlaps[f"{left}|{right}"] = len(key_sets[left] & key_sets[right])
        audit["checks"]["exact_match_key_overlap"] = exact_overlaps
        audit["checks"]["exact_match_key_overlap_total"] = int(sum(exact_overlaps.values()))

    audit["passed"] = (
        audit["checks"]["duplicate_cluster_overlap_total"] == 0
        and audit["checks"].get("exact_match_key_overlap_total", 0) == 0
    )
    return audit


def make_splits(
    frame: pd.DataFrame,
    config: DataConfig,
    *,
    strategy: Literal["grouped_stratified", "stratified", "temporal"] | None = None,
) -> SplitResult:
    """Cluster the corpus and produce leakage-safe train/validation/test splits."""
    splitting = config.splitting
    strategy = strategy or splitting.strategy
    label_column = "label"
    ratios = {
        "train": splitting.train_size,
        "validation": splitting.val_size,
        "test": splitting.test_size,
    }

    working = frame
    duplicate_stats: dict[str, Any] = {"enabled": False}
    group_column = "dup_cluster"
    working, duplicate_stats = assign_duplicate_clusters(frame, splitting.near_duplicate)

    if strategy == "temporal":
        splits = _temporal_split(working, ratios=ratios, date_column=splitting.temporal.date_column)
    elif strategy == "stratified":
        splits = _stratified_split(
            working, label_column=label_column, ratios=ratios, seed=config.seed
        )
    elif strategy == "grouped_stratified":
        splits = grouped_stratified_split(
            working,
            label_column=label_column,
            group_column=group_column,
            ratios=ratios,
            seed=config.seed,
        )
    else:  # pragma: no cover - guarded by Literal typing
        raise ValueError(f"unknown split strategy: {strategy}")

    audit = leakage_audit(splits, group_column=group_column)
    if strategy == "grouped_stratified" and not audit["passed"]:
        raise AssertionError(f"split leakage detected: {audit}")

    summary = split_summary(splits, label_column)
    summary["strategy"] = strategy
    summary["seed"] = config.seed
    summary["group_column"] = group_column
    summary["ratios_requested"] = ratios

    LOGGER.info(
        "splits: %s",
        {
            name: values["rows"]
            for name, values in summary.items()
            if isinstance(values, dict) and "rows" in values
        },
    )
    return SplitResult(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        strategy=strategy,
        summary=summary,
        duplicate_stats=duplicate_stats,
        leakage_audit=audit,
    )
