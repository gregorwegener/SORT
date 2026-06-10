"""Report and CSV writers for SORT Version 7 Phase 4."""

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


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "maximum": 0.0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_or_update_fixed_point_report(
    *,
    phase_root: Path,
    setup: dict[str, Any],
    section_name: str,
    section_summary: dict[str, Any],
) -> dict[str, Any]:
    output_path = phase_root / setup["outputs"]["fixed_point_metrics"]
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
    phase3_report = json.loads(
        (phase_root / setup["inputs"]["projector_validation"])
        .resolve()
        .read_text(encoding="utf-8")
    )

    report = {
        "run_id": phase0_manifest.get("run_id", "unavailable"),
        "sort_version": "Version 7",
        "phase": "Phase 4 — Fixed-Point Structure",
        "repository": {
            "name": "gregorwegener/SORT",
            "canonical_url": "https://github.com/gregorwegener/SORT",
            "validation_root": "validation_runs/sort_version_7_workstation_validation",
            "phase_path": "validation_runs/sort_version_7_workstation_validation/phase_4_fixed_point",
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
            "projection_kernel_gate_passed": phase2_report.get("gate_2_passed"),
        },
        "phase_3_references": {
            "projector_validation": setup["inputs"]["projector_validation"],
            "closure_test_report": setup["inputs"]["closure_test_report"],
            "global_projector_gate_passed": phase3_report.get("gate_3_passed"),
        },
        "setup_reference": "config/phase_4_setup.json",
        "projection_operator_reference": setup["inputs"]["projection_operator_ref"],
        "global_projector_reference": setup["inputs"]["global_projector_ref"],
        "iteration_rule": setup["iteration"]["iteration_rule"],
        "convergence_summary": existing.get("convergence_summary"),
        "fixed_point_classification": existing.get("fixed_point_classification"),
        "norm_invariance_summary": existing.get("norm_invariance_summary"),
        "iteration_stability_summary": existing.get("iteration_stability_summary"),
        "norm_type": setup["norm_type"],
        "tolerances": setup["tolerances"],
        "overall_passed": False,
        "gate_4_passed": False,
        "non_claims": setup["non_claims"],
    }
    report[section_name] = section_summary
    if section_name == "convergence_summary":
        report["fixed_point_classification"] = section_summary.get("classification_counts", {})

    required_sections = [
        report.get("convergence_summary"),
        report.get("norm_invariance_summary"),
        report.get("iteration_stability_summary"),
    ]
    all_sections_present = all(section is not None for section in required_sections)
    all_sections_passed = all(
        bool(section.get("passed")) for section in required_sections if section is not None
    )
    report["overall_passed"] = all_sections_present and all_sections_passed
    report["gate_4_passed"] = report["overall_passed"]
    return report
