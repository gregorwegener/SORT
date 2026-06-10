"""Report and CSV writers for SORT Version 7 Phase 3."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


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


def summarize_residuals(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "maximum_residual": 0.0,
            "mean_residual": 0.0,
            "median_residual": 0.0,
            "standard_deviation": 0.0,
        }
    return {
        "count": len(values),
        "maximum_residual": max(values),
        "mean_residual": statistics.fmean(values),
        "median_residual": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
    }


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


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_or_update_projector_report(
    *,
    phase_root: Path,
    setup: dict[str, Any],
    registry: dict[str, Any],
    section_name: str,
    section_summary: dict[str, Any],
) -> dict[str, Any]:
    output_path = phase_root / setup["outputs"]["projector_validation"]
    existing = load_report(output_path)
    phase0_manifest = json.loads(
        (phase_root / setup["inputs"]["run_manifest"]).resolve().read_text(encoding="utf-8")
    )
    phase1_report = json.loads(
        (phase_root / setup["inputs"]["operator_integrity_report"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    phase2_report = json.loads(
        (phase_root / setup["inputs"]["kernel_norm_report"])
        .resolve()
        .read_text(encoding="utf-8")
    )

    report = {
        "run_id": phase0_manifest.get("run_id", "unavailable"),
        "sort_version": "Version 7",
        "phase": "Phase 3 — Global Projector",
        "repository": {
            "name": "gregorwegener/SORT",
            "canonical_url": "https://github.com/gregorwegener/SORT",
            "validation_root": "validation_runs/sort_version_7_workstation_validation",
            "phase_path": "validation_runs/sort_version_7_workstation_validation/phase_3_global_projector",
        },
        "phase_0_references": {
            "env_spec": setup["inputs"]["env_spec"],
            "seed_config": setup["inputs"]["seed_config"],
            "run_manifest": setup["inputs"]["run_manifest"],
            "run_id": phase0_manifest.get("run_id", "unavailable"),
        },
        "phase_1_references": {
            "operator_registry": "../phase_1_operator_integrity/input/operator_registry.json",
            "operator_integrity_report": setup["inputs"]["operator_integrity_report"],
            "operator_integrity_gate_passed": phase1_report.get("gate_1_passed"),
        },
        "phase_2_references": {
            "kernel_norm_report": setup["inputs"]["kernel_norm_report"],
            "projection_residuals": setup["inputs"]["projection_residuals"],
            "projection_kernel_gate_passed": phase2_report.get("gate_2_passed"),
        },
        "setup_reference": "config/phase_3_setup.json",
        "operator_registry_reference": setup["inputs"]["operator_registry"],
        "projection_operator_reference": setup["inputs"]["projection_operator_ref"],
        "global_projector_construction_summary": {
            "symbol": setup["global_projector"]["symbol"],
            "construction_rule": setup["global_projector"]["construction_rule"],
            "operator_count": registry.get("operator_count"),
            "matrix_dimension": registry.get("matrix_dimension"),
            "description": setup["global_projector"]["description"],
        },
        "idempotency_summary": existing.get("idempotency_summary"),
        "closure_summary": existing.get("closure_summary"),
        "composition_summary": existing.get("composition_summary"),
        "norm_type": setup["norm_type"],
        "tolerances": setup["tolerances"],
        "overall_passed": False,
        "gate_3_passed": False,
        "non_claims": setup["non_claims"],
    }
    report[section_name] = section_summary

    required_sections = [
        report.get("idempotency_summary"),
        report.get("closure_summary"),
        report.get("composition_summary"),
    ]
    all_sections_present = all(section is not None for section in required_sections)
    all_sections_passed = all(
        bool(section.get("passed")) for section in required_sections if section is not None
    )
    report["overall_passed"] = all_sections_present and all_sections_passed
    report["gate_3_passed"] = report["overall_passed"]
    return report
