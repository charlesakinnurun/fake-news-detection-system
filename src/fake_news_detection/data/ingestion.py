"""Ingestion of raw labelled news CSVs into one canonical schema.

The loader is deliberately schema-agnostic: source-specific column names are
mapped through :class:`~fake_news_detection.config.ColumnMapping`, per-source
labels are declared in configuration, and arbitrary label vocabularies can be
normalised with ``dataset.label_map``. Adding a differently-shaped corpus
therefore requires configuration only, not code.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fake_news_detection.config import ColumnMapping, DataConfig, DataSourceConfig, resolve_path
from fake_news_detection.utils import sha256_file, sha256_text, utc_now_iso

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CANONICAL_COLUMNS",
    "IngestionReport",
    "SourceReport",
    "load_dataset",
    "load_source",
    "parse_dates",
]

CANONICAL_COLUMNS = ("row_id", "title", "text", "subject", "date", "date_raw", "label", "source")

_REPLACEMENT_CHAR = "\ufffd"


@dataclass(slots=True)
class SourceReport:
    """Per-file ingestion statistics."""

    path: str
    label: str
    description: str
    rows: int
    columns: list[str]
    sha256: str
    bytes: int
    encoding: str
    encoding_fallback: bool = False
    fields_with_replacement_chars: int = 0
    missing_title: int = 0
    missing_text: int = 0


@dataclass(slots=True)
class IngestionReport:
    """Aggregate ingestion statistics, serialisable to JSON."""

    dataset_name: str
    dataset_version: str
    ingested_at: str
    total_rows: int
    sources: list[SourceReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [asdict(source) for source in self.sources]
        return payload


def parse_dates(series: pd.Series, formats: list[str]) -> pd.Series:
    """Parse a mixed-format date column into timezone-naive ``datetime64``.

    Formats are tried in order, which is faster and more predictable than letting
    pandas infer a format per row.
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.astype("string").str.strip().replace({"": pd.NA})
    for fmt in formats:
        pending = result.isna() & remaining.notna()
        if not pending.any():
            break
        result.loc[pending] = pd.to_datetime(remaining[pending], format=fmt, errors="coerce")
    return result


def _read_csv(path: Path, encoding: str) -> tuple[pd.DataFrame, bool, str]:
    """Read a CSV, falling back to latin-1 when the declared encoding fails."""
    try:
        return (
            pd.read_csv(path, encoding=encoding, dtype="string", low_memory=False),
            False,
            encoding,
        )
    except UnicodeDecodeError as exc:  # pragma: no cover - depends on the input file
        LOGGER.warning(
            "encoding %s failed for %s (%s); retrying with latin-1", encoding, path.name, exc
        )
        frame = pd.read_csv(path, encoding="latin-1", dtype="string", low_memory=False)
        return frame, True, "latin-1"


def _require_columns(frame: pd.DataFrame, mapping: ColumnMapping, path: Path) -> None:
    required = {mapping.title, mapping.text}
    if mapping.label:
        required.add(mapping.label)
    missing = sorted(col for col in required if col and col not in frame.columns)
    if missing:
        raise ValueError(
            f"{path.name}: required column(s) {missing} not found; available columns: "
            f"{list(frame.columns)}. Update dataset.columns in configs/data.yaml."
        )


def _column(frame: pd.DataFrame, name: str | None) -> pd.Series:
    """Return a string column, or an empty placeholder when unmapped/missing."""
    if name and name in frame.columns:
        return frame[name].astype("string").fillna("")
    return pd.Series([""] * len(frame), index=frame.index, dtype="string")


def load_source(
    source: DataSourceConfig,
    dataset: DataConfig,
    *,
    base: Path | None = None,
) -> tuple[pd.DataFrame, SourceReport]:
    """Load a single labelled CSV into the canonical schema."""
    mapping = dataset.dataset.columns
    path = resolve_path(source.path, base)
    if not path.exists():
        raise FileNotFoundError(
            f"raw dataset file not found: {path}\n"
            "Run `python scripts/download_data.py` (or `make data`) to fetch it, or point "
            "dataset.sources in configs/data.yaml at an existing CSV."
        )

    frame, fallback, used_encoding = _read_csv(path, dataset.dataset.encoding)
    _require_columns(frame, mapping, path)

    title = _column(frame, mapping.title)
    text = _column(frame, mapping.text)
    subject = _column(frame, mapping.subject)
    date_raw = _column(frame, mapping.date)

    if mapping.label and mapping.label in frame.columns:
        raw_labels = frame[mapping.label].astype("string").fillna("").str.strip()
        label_map = {key.strip().upper(): value for key, value in dataset.dataset.label_map.items()}
        labels = raw_labels.str.upper().map(lambda value: label_map.get(value, value))
    else:
        labels = pd.Series([source.label] * len(frame), index=frame.index, dtype="string")

    replacement_fields = int(
        title.str.contains(_REPLACEMENT_CHAR, regex=False).fillna(False).sum()
        + text.str.contains(_REPLACEMENT_CHAR, regex=False).fillna(False).sum()
    )

    canonical = pd.DataFrame(
        {
            "title": title,
            "text": text,
            "subject": subject,
            "date_raw": date_raw,
            "label": labels.astype("string"),
            "source": Path(source.path).name,
        }
    )
    canonical["date"] = parse_dates(canonical["date_raw"], dataset.dataset.date_formats)
    canonical["row_id"] = [
        f"{dataset.dataset.name}:{Path(source.path).stem}:{index}"
        for index in range(len(canonical))
    ]
    canonical = canonical.loc[:, list(CANONICAL_COLUMNS)]

    report = SourceReport(
        path=str(path),
        label=source.label,
        description=source.description,
        rows=len(canonical),
        columns=list(frame.columns),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        encoding=used_encoding,
        encoding_fallback=fallback,
        fields_with_replacement_chars=replacement_fields,
        missing_title=int((canonical["title"].str.strip() == "").sum()),
        missing_text=int((canonical["text"].str.strip() == "").sum()),
    )
    return canonical, report


def load_dataset(
    config: DataConfig, base: Path | None = None
) -> tuple[pd.DataFrame, IngestionReport]:
    """Load, validate and concatenate every configured source."""
    frames: list[pd.DataFrame] = []
    reports: list[SourceReport] = []
    warnings: list[str] = []

    for source in config.dataset.sources:
        frame, source_report = load_source(source, config, base=base)
        if source_report.encoding_fallback:
            warnings.append(
                f"{source_report.path}: decoded with latin-1 fallback (non-UTF-8 bytes)"
            )
        if source_report.fields_with_replacement_chars:
            warnings.append(
                f"{source_report.path}: {source_report.fields_with_replacement_chars} field(s) "
                "contain U+FFFD replacement characters (possible encoding damage)"
            )
        if source_report.label not in config.dataset.labels:
            raise ValueError(
                f"{source_report.path}: source label {source_report.label!r} is not declared in "
                f"dataset.labels={config.dataset.labels}"
            )
        frames.append(frame)
        reports.append(source_report)

    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = combined["label"].astype("string").str.strip()

    fingerprint_payload = "|".join(
        sorted(f"{item.sha256}:{item.rows}:{item.label}" for item in reports)
    )
    ingestion_report = IngestionReport(
        dataset_name=config.dataset.name,
        dataset_version=config.dataset.version,
        ingested_at=utc_now_iso(),
        total_rows=len(combined),
        sources=reports,
        warnings=warnings,
        fingerprint=sha256_text(fingerprint_payload),
    )
    LOGGER.info(
        "ingested %d rows from %d source(s); fingerprint=%s",
        ingestion_report.total_rows,
        len(reports),
        ingestion_report.fingerprint[:12],
    )
    return combined, ingestion_report
