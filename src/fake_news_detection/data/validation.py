"""Dataset validation, quality reporting and auditable record exclusion.

Nothing is dropped silently. Every excluded row is attributed to exactly one
reason (the first rule that matched), the counts are reported, and a small
sanitised sample of each exclusion class is written to the quality report.

The module also surfaces *dataset artifacts* - systematic differences between
the positive and negative classes that a model can exploit without learning
anything about factual accuracy. Those findings drive the caveats in the model
card, so they are treated as first-class output rather than a footnote.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fake_news_detection.config import DataConfig
from fake_news_detection.data.ingestion import IngestionReport

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ValidationReport",
    "exclusion_summary",
    "normalise_for_matching",
    "validate_and_clean",
]

_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^0-9a-z\u00c0-\u024f ]+")
_REUTERS_RE = re.compile(r"\breuters\b", re.IGNORECASE)


def normalise_for_matching(text: pd.Series) -> pd.Series:
    """Build a comparison key for duplicate detection.

    Case, punctuation and whitespace differences are removed so that
    cosmetically edited copies of the same article collapse together.
    """
    return (
        text.astype("string")
        .str.replace(_WHITESPACE_RE, " ", regex=True)
        .str.strip()
        .str.lower()
        .str.replace(_NON_WORD_RE, "", regex=True)
        .str.replace(_WHITESPACE_RE, " ", regex=True)
        .fillna("")
    )


@dataclass(slots=True)
class RuleOutcome:
    """Counts and examples produced by a single validation rule."""

    rule: str
    description: str
    affected: int
    action: str  # "excluded" | "flagged"
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ValidationReport:
    """Machine-readable summary of the cleaning stage."""

    dataset_name: str
    dataset_version: str
    dataset_fingerprint: str
    rows_in: int
    rows_out: int
    rows_excluded: int
    exclusion_counts: dict[str, int]
    class_counts_before: dict[str, int]
    class_counts_after: dict[str, int]
    class_imbalance_ratio: float
    duplicate_rows_exact: int
    conflicting_label_groups: int
    conflicting_label_rows: int
    text_length_by_class: dict[str, dict[str, float]]
    title_length_by_class: dict[str, dict[str, float]]
    subject_distribution: dict[str, dict[str, int]]
    temporal_range_by_class: dict[str, dict[str, Any]]
    artifact_findings: list[dict[str, Any]] = field(default_factory=list)
    rules: list[RuleOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rules"] = [asdict(rule) for rule in self.rules]
        return payload


def _length_quantiles(series: pd.Series) -> dict[str, float]:
    lengths = series.astype("string").fillna("").str.len().to_numpy(dtype="float64")
    if lengths.size == 0:
        return {"mean": 0.0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    quantiles = np.percentile(lengths, [5, 50, 95])
    return {
        "mean": round(float(lengths.mean()), 2),
        "min": float(lengths.min()),
        "p05": round(float(quantiles[0]), 2),
        "p50": round(float(quantiles[1]), 2),
        "p95": round(float(quantiles[2]), 2),
        "max": float(lengths.max()),
    }


def _examples(frame: pd.DataFrame, index: pd.Index, limit: int) -> list[dict[str, Any]]:
    """Return a small, truncated sample of rows for the quality report.

    Only the headline (truncated) and length metadata are recorded; full article
    bodies are deliberately not copied into reports.
    """
    if limit <= 0 or len(index) == 0:
        return []
    subset = frame.loc[index[:limit]]
    examples: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        title = str(row.get("title", ""))
        examples.append(
            {
                "row_id": str(row.get("row_id", "")),
                "source": str(row.get("source", "")),
                "label": str(row.get("label", "")),
                "title": (title[:110] + "…") if len(title) > 110 else title,
                "text_chars": len(str(row.get("text", ""))),
            }
        )
    return examples


def exclusion_summary(reasons: pd.Series) -> dict[str, int]:
    """Count non-null entries of an exclusion-reason series."""
    counts = reasons.dropna().value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _artifact_findings(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect systematic, label-correlated artifacts in the raw corpus."""
    findings: list[dict[str, Any]] = []
    labels = sorted(str(x) for x in frame["label"].dropna().unique().tolist())

    per_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        subset = frame[frame["label"] == label]
        if subset.empty:
            continue
        text = subset["text"].astype("string").fillna("")
        subjects = subset["subject"].astype("string").fillna("").value_counts(normalize=True)
        per_label[label] = {
            "rows": len(subset),
            "share_of_corpus": round(len(subset) / max(len(frame), 1), 4),
            "empty_body_rate": round(float((text.str.strip() == "").mean()), 4),
            "reuters_wire_marker_rate": round(float(text.str.contains(_REUTERS_RE).mean()), 4),
            "top_subjects": {str(k): round(float(v), 4) for k, v in subjects.head(6).items()},
        }
    findings.extend({"label": label, **values} for label, values in per_label.items())

    # Disjoint subject vocabularies are a classic shortcut signal.
    subject_sets = {
        label: set(frame.loc[frame["label"] == label, "subject"].dropna().unique().tolist())
        for label in labels
    }
    if len(subject_sets) >= 2:
        intersection = set.intersection(*(subject_sets[key] for key in subject_sets))
        findings.append(
            {
                "finding": "subject_vocabulary_overlap",
                "intersection": sorted(str(s) for s in intersection),
                "per_label": {
                    key: sorted(str(s) for s in subject_sets[key]) for key in subject_sets
                },
                "interpretation": (
                    "An empty subject intersection means the label can be recovered from the "
                    "subject field alone, which will not generalise to unseen sources."
                    if not intersection
                    else "Subjects overlap across classes; lower subject-driven shortcut risk."
                ),
            }
        )

    empty_rates = {label: values["empty_body_rate"] for label, values in per_label.items()}
    if len(empty_rates) >= 2 and max(empty_rates.values()) - min(empty_rates.values()) > 0.02:
        findings.append(
            {
                "finding": "missing_body_imbalance",
                "empty_body_rate_by_label": empty_rates,
                "interpretation": (
                    "Rows without a body are not evenly distributed across labels, so retaining "
                    "them would let the model classify using body presence alone."
                ),
            }
        )

    reuters_rates = {
        label: values["reuters_wire_marker_rate"] for label, values in per_label.items()
    }
    if reuters_rates and max(reuters_rates.values()) > 0.5:
        findings.append(
            {
                "finding": "wire_marker_imbalance",
                "reuters_marker_rate_by_label": reuters_rates,
                "interpretation": (
                    "The Reuters wire marker is concentrated in one class. A bag-of-words model "
                    "will learn it as a shortcut, which inflates in-distribution accuracy and "
                    "does not transfer to other publishers."
                ),
            }
        )
    return findings


def validate_and_clean(
    frame: pd.DataFrame,
    config: DataConfig,
    ingestion: IngestionReport | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate ``frame`` and return ``(clean_frame, report)``.

    The returned frame contains only rows that passed every rule, plus
    ``match_key`` (used by the splitter for exact-duplicate grouping) and
    ``row_length``.
    """
    rules_cfg = config.validation
    labels = config.dataset.labels
    limit = rules_cfg.max_reported_examples

    reasons = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    outcomes: list[RuleOutcome] = []

    def mark(
        mask: pd.Series | np.ndarray, rule: str, description: str, action: str = "excluded"
    ) -> int:
        applicable = pd.Series(np.asarray(mask), index=frame.index).fillna(False).astype(bool)
        applicable &= reasons.isna()
        affected = int(applicable.sum())
        if action == "excluded":
            reasons.loc[applicable] = rule
        outcomes.append(
            RuleOutcome(
                rule=rule,
                description=description,
                affected=affected,
                action=action,
                examples=_examples(frame, frame.index[applicable.to_numpy()], limit),
            )
        )
        return affected

    title = frame["title"].astype("string").fillna("")
    text = frame["text"].astype("string").fillna("")
    title_stripped = title.str.strip()
    text_stripped = text.str.strip()

    # --- rules, in priority order -------------------------------------------
    mark(~frame["label"].isin(labels), "invalid_label", f"label not in {labels}")

    both_empty = (title_stripped == "") & (text_stripped == "")
    mark(both_empty, "empty_content", "headline and body are both empty")

    if rules_cfg.drop_empty_bodies:
        mark(
            text_stripped.str.len() < rules_cfg.min_text_chars,
            "short_body",
            f"body shorter than min_text_chars={rules_cfg.min_text_chars}",
        )

    total_len = title_stripped.str.len() + text_stripped.str.len()
    mark(
        total_len < rules_cfg.min_total_chars,
        "short_total",
        f"combined length below min_total_chars={rules_cfg.min_total_chars}",
    )
    mark(
        title_stripped.str.len() > rules_cfg.max_title_chars,
        "title_too_long",
        "headline over limit",
    )
    mark(text_stripped.str.len() > rules_cfg.max_text_chars, "text_too_long", "body over limit")

    match_key = normalise_for_matching(title_stripped + " " + text_stripped)
    usable = reasons.isna()

    # Conflicting labels are detected before de-duplication so that a single
    # article carrying two different labels is never used silently.
    usable_index = frame.index[usable.to_numpy()]
    key_usable = match_key.loc[usable_index]
    nunique = frame.loc[usable_index].groupby(key_usable, dropna=True)["label"].nunique()
    conflicting_keys = set(nunique[nunique > 1].index)
    conflicting_index = usable_index[key_usable.isin(conflicting_keys).to_numpy()]
    conflicting_groups = len(conflicting_keys)

    if conflicting_index.empty:
        mark(
            pd.Series(False, index=frame.index),
            "conflicting_label",
            "identical text, multiple labels",
        )
    elif rules_cfg.conflicting_label_policy == "majority":
        subset_labels = frame.loc[conflicting_index, "label"]
        subset_keys = match_key.loc[conflicting_index]
        majority = subset_labels.groupby(subset_keys).agg(
            lambda values: values.value_counts().idxmax()
        )
        agreement = subset_labels.groupby(subset_keys).agg(
            lambda values: float(values.value_counts(normalize=True).iloc[0])
        )
        row_majority = subset_keys.map(majority)
        row_agreement = subset_keys.map(agreement)
        keep = (subset_labels.to_numpy() == row_majority.to_numpy()) & (
            row_agreement.to_numpy() >= rules_cfg.majority_min_agreement
        )
        drop_index = conflicting_index[~keep]
        mark(
            frame.index.isin(drop_index),
            "conflicting_label",
            "identical text carries more than one label (majority rule applied)",
        )
    else:
        mark(
            frame.index.isin(conflicting_index),
            "conflicting_label",
            "identical text carries more than one label (dropped by policy)",
        )

    if rules_cfg.drop_exact_duplicates:
        usable = reasons.isna()
        key = pd.Series(np.where(usable, match_key, pd.NA), index=frame.index, dtype="object")
        duplicated = usable & key.duplicated(keep="first")
        mark(
            duplicated,
            "duplicate_text",
            "identical normalised title+body already seen (first occurrence kept)",
        )

    clean = frame.loc[reasons.isna()].copy()
    clean["match_key"] = match_key.loc[clean.index].to_numpy()
    clean["row_length"] = (title_stripped + text_stripped).loc[clean.index].str.len().to_numpy()
    clean = clean.reset_index(drop=True)

    class_before = {
        str(k): int(v)
        for k, v in frame["label"].value_counts().to_dict().items()
        if str(k) in labels
    }
    class_after = {
        str(k): int(v)
        for k, v in clean["label"].value_counts().to_dict().items()
        if str(k) in labels
    }
    counts = np.asarray(sorted(class_after.values()), dtype="float64")
    imbalance = (
        float(counts.max() / counts.min()) if counts.size > 1 and counts.min() > 0 else float("nan")
    )

    length_by_class = {
        str(label): _length_quantiles(clean.loc[clean["label"] == label, "text"])
        for label in labels
        if (clean["label"] == label).any()
    }
    title_length_by_class = {
        str(label): _length_quantiles(clean.loc[clean["label"] == label, "title"])
        for label in labels
        if (clean["label"] == label).any()
    }
    subject_distribution = {
        str(label): {
            str(k): int(v)
            for k, v in clean.loc[clean["label"] == label, "subject"].value_counts().items()
        }
        for label in labels
        if (clean["label"] == label).any()
    }
    temporal: dict[str, dict[str, Any]] = {}
    for label in labels:
        subset = clean.loc[clean["label"] == label, "date"].dropna()
        temporal[str(label)] = {
            "parsed": int(subset.shape[0]),
            "min": str(subset.min().date()) if not subset.empty else None,
            "max": str(subset.max().date()) if not subset.empty else None,
        }

    exclusions = exclusion_summary(reasons)
    warnings: list[str] = list(ingestion.warnings) if ingestion else []
    if not np.isnan(imbalance) and imbalance > 1.5:
        warnings.append(f"class imbalance ratio is {imbalance:.2f}; consider class_weight tuning")
    unparsed = int(clean["date"].isna().sum())
    if unparsed:
        warnings.append(f"{unparsed} retained row(s) have an unparsable or missing date")

    report = ValidationReport(
        dataset_name=config.dataset.name,
        dataset_version=config.dataset.version,
        dataset_fingerprint=ingestion.fingerprint if ingestion else "",
        rows_in=len(frame),
        rows_out=len(clean),
        rows_excluded=int(len(frame) - len(clean)),
        exclusion_counts=exclusions,
        class_counts_before=class_before,
        class_counts_after=class_after,
        class_imbalance_ratio=round(imbalance, 4) if not np.isnan(imbalance) else -1.0,
        duplicate_rows_exact=int(exclusions.get("duplicate_text", 0)),
        conflicting_label_groups=conflicting_groups,
        conflicting_label_rows=len(conflicting_index),
        text_length_by_class=length_by_class,
        title_length_by_class=title_length_by_class,
        subject_distribution=subject_distribution,
        temporal_range_by_class=temporal,
        artifact_findings=_artifact_findings(frame),
        rules=outcomes,
        warnings=warnings,
    )
    LOGGER.info(
        "validation kept %d/%d rows (%d excluded)",
        report.rows_out,
        report.rows_in,
        report.rows_excluded,
    )
    return clean, report
