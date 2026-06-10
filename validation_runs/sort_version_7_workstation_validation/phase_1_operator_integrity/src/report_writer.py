"""Report and residual output helpers for SORT Version 7 Phase 1."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "test_type",
    "operator_id_i",
    "operator_id_j",
    "operator_id_k",
    "residual",
    "norm_type",
    "passed",
]


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_residual_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def summarize_residuals(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "maximum_residual": 0.0,
            "mean_residual": 0.0,
            "median_residual": 0.0,
        }
    return {
        "count": len(values),
        "maximum_residual": max(values),
        "mean_residual": statistics.fmean(values),
        "median_residual": statistics.median(values),
    }


def load_json_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def read_residual_csv(path: str | Path) -> list[dict[str, str]]:
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
    retained = [row for row in existing_rows if row.get("test_type") != test_type]
    return retained + replacement_rows


def build_or_update_report(
    *,
    phase_root: Path,
    setup: dict[str, Any],
    registry: dict[str, Any],
    section_name: str,
    section_summary: dict[str, Any],
) -> dict[str, Any]:
    output_path = phase_root / setup["outputs"]["operator_integrity_report"]
    existing = load_json_report(output_path)
    run_manifest_path = (phase_root / setup["inputs"]["run_manifest"]).resolve()
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    report = {
        "run_id": run_manifest.get("run_id", "unavailable"),
        "sort_version": "Version 7",
        "phase": "Phase 1 — Operator Integrity",
        "repository": {
            "name": "gregorwegener/SORT",
            "canonical_url": "https://github.com/gregorwegener/SORT",
            "validation_root": "validation_runs/sort_version_7_workstation_validation",
            "phase_path": "validation_runs/sort_version_7_workstation_validation/phase_1_operator_integrity",
        },
        "phase_0_references": setup["inputs"],
        "phase_0_run_id": run_manifest.get("run_id", "unavailable"),
        "phase_0_git": run_manifest.get("git", {}),
        "setup_reference": "config/phase_1_setup.json",
        "registry_reference": "input/operator_registry.json",
        "operator_count": registry["operator_count"],
        "matrix_dimension": registry["matrix_dimension"],
        "norm_type": setup["norm_type"],
        "tolerances": setup["tolerances"],
        "idempotency_summary": existing.get("idempotency_summary"),
        "balance_summary": existing.get("balance_summary"),
        "jacobi_summary": existing.get("jacobi_summary"),
        "overall_passed": False,
        "gate_1_passed": False,
        "non_claims": setup["non_claims"],
    }
    report[section_name] = section_summary

    required_sections = [
        report.get("idempotency_summary"),
        report.get("balance_summary"),
        report.get("jacobi_summary"),
    ]
    all_sections_present = all(section is not None for section in required_sections)
    all_sections_passed = all(
        bool(section.get("passed")) for section in required_sections if section is not None
    )
    report["overall_passed"] = all_sections_present and all_sections_passed
    report["gate_1_passed"] = report["overall_passed"]
    return report
