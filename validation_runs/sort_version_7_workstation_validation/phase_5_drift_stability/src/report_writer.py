"""Report and CSV writers for SORT Version 7 Phase 5."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


DRIFT_PROFILE_FIELDS = [
    "test_type",
    "state_id",
    "reference_class",
    "iteration",
    "drift_value",
    "normalized_drift_value",
    "drift_label",
    "stability_label",
    "perturbation_strength",
    "rescaling_factor",
    "metric_status",
]


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def replace_rows(
    existing_rows: list[dict[str, Any]],
    test_type: str,
    replacement_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [row for row in existing_rows if row.get("test_type") != test_type] + replacement_rows


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "maximum": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))
