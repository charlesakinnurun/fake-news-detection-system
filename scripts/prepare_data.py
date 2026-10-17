#!/usr/bin/env python
"""Build the validated, leakage-safe train/validation/test splits.

Pipeline: ingest -> validate -> cluster duplicates -> split -> persist.

Outputs (all under the directories declared in ``configs/data.yaml`` defaults):

* ``data/processed/{split}.parquet``      - the three splits
* ``data/processed/dataset_manifest.json`` - fingerprint, sizes, split audit
* ``reports/data_quality_report.json``     - machine-readable quality report
* ``reports/data_quality_report.md``       - human-readable quality report

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --ablation   # also build a plain stratified split
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from fake_news_detection.config import load_data_config, repo_root
from fake_news_detection.data.ingestion import load_dataset
from fake_news_detection.data.splitting import make_splits
from fake_news_detection.data.validation import validate_and_clean
from fake_news_detection.monitoring.logging import configure_logging
from fake_news_detection.utils import (
    display_path,
    ensure_dir,
    human_bytes,
    utc_now_iso,
    write_json,
)

LOGGER = logging.getLogger("prepare_data")

SPLIT_COLUMNS = [
    "row_id",
    "title",
    "text",
    "subject",
    "date",
    "label",
    "source",
    "match_key",
    "row_length",
    "dup_cluster",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-config", default=None, help="path to data.yaml")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["grouped_stratified", "stratified", "temporal"],
        help="override splitting.strategy",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="additionally build a plain stratified split to quantify duplicate leakage",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _quality_markdown(
    report: dict[str, Any], ingestion: dict[str, Any], splits: dict[str, Any]
) -> str:
    lines: list[str] = ["# Data quality report", ""]
    lines.append(f"- dataset: `{report['dataset_name']}` v{report['dataset_version']}")
    lines.append(f"- dataset fingerprint: `{report['dataset_fingerprint']}`")
    lines.append(f"- rows in: **{report['rows_in']:,}**")
    lines.append(f"- rows retained: **{report['rows_out']:,}**")
    lines.append(f"- rows excluded: **{report['rows_excluded']:,}**")
    lines.append("")

    lines.append("## Class distribution")
    lines.append("")
    lines.append("| split | rows | " + " | ".join(sorted(report["class_counts_after"])) + " |")
    lines.append("|---" * (len(report["class_counts_after"]) + 2) + "|")
    for name, info in splits.items():
        if not isinstance(info, dict) or "label_counts" not in info:
            continue
        counts = info.get("label_counts", {})
        cells = [counts.get(label, 0) for label in sorted(report["class_counts_after"])]
        lines.append(
            f"| {name} | {info.get('rows', 0):,} | " + " | ".join(str(c) for c in cells) + " |"
        )
    lines.append("")
    lines.append(f"Class imbalance ratio (max/min): **{report['class_imbalance_ratio']}**")
    lines.append("")

    lines.append("## Excluded records by reason")
    lines.append("")
    lines.append("| reason | rows |")
    lines.append("|---|---|")
    for reason, count in sorted(report["exclusion_counts"].items(), key=lambda item: -item[1]):
        lines.append(f"| `{reason}` | {count:,} |")
    lines.append("")

    lines.append("## Validation rules")
    lines.append("")
    lines.append("| rule | affected | action | description |")
    lines.append("|---|---|---|---|")
    for rule in report["rules"]:
        lines.append(
            f"| `{rule['rule']}` | {rule['affected']:,} | {rule['action']} | {rule['description']} |"
        )
    lines.append("")

    lines.append("## Duplicate and conflicting content")
    lines.append("")
    lines.append(f"- exact duplicate rows removed: **{report['duplicate_rows_exact']:,}**")
    lines.append(f"- groups with conflicting labels: **{report['conflicting_label_groups']:,}**")
    lines.append(f"- rows in those groups: **{report['conflicting_label_rows']:,}**")
    lines.append("")

    lines.append("## Text length by class (characters)")
    lines.append("")
    lines.append("| class | min | p05 | p50 | p95 | max | mean |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, stats in report["text_length_by_class"].items():
        lines.append(
            f"| {label} | {stats['min']:.0f} | {stats['p05']:.0f} | {stats['p50']:.0f} | "
            f"{stats['p95']:.0f} | {stats['max']:.0f} | {stats['mean']:.0f} |"
        )
    lines.append("")

    lines.append("## Temporal coverage")
    lines.append("")
    lines.append("| class | parsed dates | min | max |")
    lines.append("|---|---|---|---|")
    for label, stats in report["temporal_range_by_class"].items():
        lines.append(f"| {label} | {stats['parsed']:,} | {stats['min']} | {stats['max']} |")
    lines.append("")

    lines.append("## Dataset artifacts (why the reported accuracy is optimistic)")
    lines.append("")
    for finding in report["artifact_findings"]:
        if "label" in finding:
            lines.append(
                f"- **{finding['label']}** — rows: {finding['rows']:,}; empty body rate: "
                f"{finding['empty_body_rate']}; Reuters wire marker rate: "
                f"{finding['reuters_wire_marker_rate']}"
            )
        elif finding.get("finding") == "subject_vocabulary_overlap":
            lines.append(f"- **subject vocabulary**: {finding['interpretation']}")
            lines.append(f"  - per label: `{finding['per_label']}`")
        elif finding.get("finding") == "missing_body_imbalance":
            lines.append(
                f"- **missing-body imbalance**: {finding['empty_body_rate_by_label']} — "
                f"{finding['interpretation']}"
            )
        elif finding.get("finding") == "wire_marker_imbalance":
            lines.append(
                f"- **wire-marker imbalance**: {finding['reuters_marker_rate_by_label']} — "
                f"{finding['interpretation']}"
            )
    lines.append("")

    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")

    lines.append("## Ingestion")
    lines.append("")
    lines.append("| file | rows | bytes | sha256 | encoding |")
    lines.append("|---|---|---|---|---|")
    for source in ingestion["sources"]:
        lines.append(
            f"| {Path(source['path']).name} | {source['rows']:,} | {source['bytes']:,} | "
            f"`{source['sha256'][:16]}…` | {source['encoding']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, json_output=False, service="prepare_data")
    root = repo_root()
    config = load_data_config(args.data_config)

    output_dir = ensure_dir(root / args.output_dir)
    reports_dir = ensure_dir(root / args.reports_dir)

    frame, ingestion = load_dataset(config)
    clean, validation = validate_and_clean(frame, config, ingestion)
    if clean.empty:
        LOGGER.error("no rows survived validation; nothing to write")
        return 1

    result = make_splits(clean, config, strategy=args.strategy)

    for name, split in result.splits().items():
        path = output_dir / f"{name}.parquet"
        split.loc[:, [c for c in SPLIT_COLUMNS if c in split.columns]].to_parquet(path, index=False)
        LOGGER.info(
            "wrote %s (%s rows, %s)",
            display_path(path, root),
            len(split),
            human_bytes(path.stat().st_size),
        )

    ablation: dict[str, Any] | None = None
    if args.ablation:
        plain = make_splits(clean, config, strategy="stratified")
        ablation = {
            "strategy": plain.strategy,
            "leakage_audit": plain.leakage_audit,
            "duplicate_stats": plain.duplicate_stats,
            "summary": plain.summary,
            "note": (
                "A plain stratified split places near-duplicate articles on both sides of the "
                "split boundary. This is the leakage that the grouped strategy prevents."
            ),
        }
        LOGGER.info(
            "ablation: plain stratified split shares %d duplicate cluster(s) across splits",
            plain.leakage_audit["checks"]["duplicate_cluster_overlap_total"],
        )

    manifest = {
        "created_at": utc_now_iso(),
        "dataset": {
            "name": config.dataset.name,
            "version": config.dataset.version,
            "fingerprint": ingestion.fingerprint,
            "citation": config.dataset.citation,
            "license_note": config.dataset.license_note,
            "homepage": config.dataset.homepage,
        },
        "seed": config.seed,
        "preprocessing": config.preprocessing.model_dump(),
        "splitting": result.as_dict(),
        "rows_prepared": len(clean),
        "ingestion": ingestion.to_dict(),
        "ablation_plain_stratified": ablation,
        "artifacts": {
            "train": "data/processed/train.parquet",
            "validation": "data/processed/validation.parquet",
            "test": "data/processed/test.parquet",
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(
        reports_dir / "data_quality_report.json",
        {
            "validation": validation.to_dict(),
            "ingestion": ingestion.to_dict(),
            "splitting": result.as_dict(),
            "ablation_plain_stratified": ablation,
        },
    )
    markdown = _quality_markdown(validation.to_dict(), ingestion.to_dict(), result.summary)
    (reports_dir / "data_quality_report.md").write_text(markdown, "utf-8")

    LOGGER.info(
        "quality report: %s",
        display_path(reports_dir / "data_quality_report.md", root),
    )
    LOGGER.info("done. Next: python scripts/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
