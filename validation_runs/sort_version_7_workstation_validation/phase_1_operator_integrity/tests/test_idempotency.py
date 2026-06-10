"""Run Phase 1 per-operator idempotency validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_1_operator_integrity.src.commutators import _matmul
from phase_1_operator_integrity.src.norms import residual_norm
from phase_1_operator_integrity.src.operator_registry import (
    get_all_operators,
    load_registry,
)
from phase_1_operator_integrity.src.report_writer import (
    build_or_update_report,
    replace_rows,
    read_residual_csv,
    summarize_residuals,
    write_json_report,
    write_residual_csv,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup = json.loads((phase_root / "config" / "phase_1_setup.json").read_text(encoding="utf-8"))
    registry = load_registry(phase_root / setup["inputs"]["operator_registry"])
    tolerance = float(setup["tolerances"]["epsilon_idempotency"])
    norm_type = setup["norm_type"]

    residuals: list[float] = []
    rows = []
    failed_operator_ids = []

    for operator in get_all_operators(registry):
        operator_id = operator["operator_id"]
        matrix = operator["matrix"]
        squared = _matmul(matrix, matrix)
        residual = residual_norm(squared, matrix, norm_type)
        passed = residual < tolerance
        residuals.append(residual)
        if not passed:
            failed_operator_ids.append(operator_id)
        rows.append(
            {
                "test_type": "idempotency",
                "operator_id_i": operator_id,
                "operator_id_j": "",
                "operator_id_k": "",
                "residual": f"{residual:.17g}",
                "norm_type": norm_type,
                "passed": str(passed).lower(),
            }
        )

    summary = summarize_residuals(residuals)
    summary.update(
        {
            "tolerance": tolerance,
            "passed_count": len(residuals) - len(failed_operator_ids),
            "failed_count": len(failed_operator_ids),
            "failed_operator_ids": failed_operator_ids,
            "passed": not failed_operator_ids,
        }
    )

    report = build_or_update_report(
        phase_root=phase_root,
        setup=setup,
        registry=registry,
        section_name="idempotency_summary",
        section_summary=summary,
    )
    report_path = phase_root / setup["outputs"]["operator_integrity_report"]
    csv_path = phase_root / setup["outputs"]["operator_residuals"]
    write_json_report(report_path, report)
    write_residual_csv(csv_path, replace_rows(read_residual_csv(csv_path), "idempotency", rows))

    print("Phase 1 idempotency validation passed" if summary["passed"] else "Phase 1 idempotency validation failed")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
