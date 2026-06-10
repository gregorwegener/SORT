"""Report and CSV writers for SORT Version 7 Phase 2."""

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
        }
    return {
        "count": len(values),
        "maximum_residual": max(values),
        "mean_residual": statistics.fmean(values),
        "median_residual": statistics.median(values),
    }


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_or_update_report(
    *,
    phase_root: Path,
    setup: dict[str, Any],
    section_name: str,
    section_summary: dict[str, Any],
) -> dict[str, Any]:
    output_path = phase_root / setup["outputs"]["kernel_norm_report"]
    existing = load_report(output_path)
    phase0_manifest = json.loads(
        (phase_root / setup["inputs"]["run_manifest"]).resolve().read_text(encoding="utf-8")
    )
    phase1_report = json.loads(
        (phase_root / setup["inputs"]["operator_integrity_report"])
        .resolve()
        .read_text(encoding="utf-8")
    )

    report = {
        "run_id": phase0_manifest.get("run_id", "unavailable"),
        "sort_version": "Version 7",
        "phase": "Phase 2 — Projection Kernel",
        "repository": {
            "name": "gregorwegener/SORT",
            "canonical_url": "https://github.com/gregorwegener/SORT",
            "validation_root": "validation_runs/sort_version_7_workstation_validation",
            "phase_path": "validation_runs/sort_version_7_workstation_validation/phase_2_projection_kernel",
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
        "setup_reference": "config/phase_2_setup.json",
        "kernel_definition_reference": "input/kernel_definitions.yaml",
        "operator_registry_reference": setup["inputs"]["operator_registry"],
        "kernel_normalization_summary": existing.get("kernel_normalization_summary"),
        "projection_idempotency_summary": existing.get("projection_idempotency_summary"),
        "scale_grid_summary": {
            "mode": setup["k_grid"]["mode"],
            "values": setup["k_grid"]["values"],
            "xi_grid_mode": setup["xi_grid"]["mode"],
            "xi_grid_count": len(setup["xi_grid"]["values"]),
        },
        "norm_type": setup["norm_type"],
        "tolerances": setup["tolerances"],
        "overall_passed": False,
        "gate_2_passed": False,
        "non_claims": setup["non_claims"],
    }
    report[section_name] = section_summary

    required_sections = [
        report.get("kernel_normalization_summary"),
        report.get("projection_idempotency_summary"),
    ]
    all_sections_present = all(section is not None for section in required_sections)
    all_sections_passed = all(
        bool(section.get("passed")) for section in required_sections if section is not None
    )
    report["overall_passed"] = all_sections_present and all_sections_passed
    report["gate_2_passed"] = report["overall_passed"]
    return report
