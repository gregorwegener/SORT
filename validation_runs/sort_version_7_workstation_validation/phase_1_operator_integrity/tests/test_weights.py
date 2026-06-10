"""Run Phase 1 structural balance validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_1_operator_integrity.src.operator_registry import get_weights, load_registry
from phase_1_operator_integrity.src.report_writer import (
    build_or_update_report,
    replace_rows,
    read_residual_csv,
    write_json_report,
    write_residual_csv,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup = json.loads((phase_root / "config" / "phase_1_setup.json").read_text(encoding="utf-8"))
    registry = load_registry(phase_root / setup["inputs"]["operator_registry"])
    tolerance = float(setup["tolerances"]["epsilon_balance"])
    norm_type = setup["norm_type"]

    weight_sum = sum(get_weights(registry))
    residual = abs(weight_sum)
    passed = residual < tolerance

    summary = {
        "weight_sum": weight_sum,
        "absolute_residual": residual,
        "tolerance": tolerance,
        "passed": passed,
    }
    rows = [
        {
            "test_type": "balance",
            "operator_id_i": "",
            "operator_id_j": "",
            "operator_id_k": "",
            "residual": f"{residual:.17g}",
            "norm_type": norm_type,
            "passed": str(passed).lower(),
        }
    ]

    report = build_or_update_report(
        phase_root=phase_root,
        setup=setup,
        registry=registry,
        section_name="balance_summary",
        section_summary=summary,
    )
    report_path = phase_root / setup["outputs"]["operator_integrity_report"]
    csv_path = phase_root / setup["outputs"]["operator_residuals"]
    write_json_report(report_path, report)
    write_residual_csv(csv_path, replace_rows(read_residual_csv(csv_path), "balance", rows))

    print("Phase 1 structural balance validation passed" if passed else "Phase 1 structural balance validation failed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
