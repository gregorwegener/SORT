"""Run Phase 3 global projector idempotency validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from phase_3_global_projector.src.global_projector import (
    construct_global_projector,
    load_projector_config,
)
from phase_3_global_projector.src.projector_idempotency import (
    projector_idempotency_residual,
    statewise_projector_residual,
    summarize_projector_residuals,
)
from phase_3_global_projector.src.report_writer import (
    build_or_update_projector_report,
    read_csv,
    replace_rows,
    write_csv,
    write_json_report,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup = load_projector_config(phase_root / "config" / "phase_3_setup.json")
    registry = json.loads((phase_root / setup["inputs"]["operator_registry"]).read_text(encoding="utf-8"))
    seed_config = json.loads((phase_root / setup["synthetic_states"]["seed_reference"]).resolve().read_text(encoding="utf-8"))
    projector = construct_global_projector(registry, setup)

    norm_type = setup["norm_type"]
    tolerance = float(setup["tolerances"]["epsilon_projector"])
    states = _synthetic_states(
        int(setup["synthetic_states"]["state_count"]),
        int(setup["synthetic_states"]["state_dimension"]),
        int(seed_config["global_seed"]),
    )

    direct_residual = projector_idempotency_residual(projector, norm_type)
    direct_passed = direct_residual < tolerance
    state_residuals = []
    failed_state_ids = []
    rows = [
        {
            "test_type": "direct_projector",
            "state_id": "",
            "residual": f"{direct_residual:.17g}",
            "norm_type": norm_type,
            "passed": str(direct_passed).lower(),
        }
    ]

    for state_id, state in enumerate(states, start=1):
        residual = statewise_projector_residual(state, projector, norm_type)
        passed = residual < tolerance
        state_residuals.append(residual)
        if not passed:
            failed_state_ids.append(state_id)
        rows.append(
            {
                "test_type": "statewise_projector",
                "state_id": state_id,
                "residual": f"{residual:.17g}",
                "norm_type": norm_type,
                "passed": str(passed).lower(),
            }
        )

    summary = summarize_projector_residuals(state_residuals)
    summary.update(
        {
            "direct_projector_residual": direct_residual,
            "direct_projector_passed": direct_passed,
            "epsilon_projector": tolerance,
            "norm_type": norm_type,
            "state_count": len(states),
            "seed_reference": setup["synthetic_states"]["seed_reference"],
            "global_seed": seed_config["global_seed"],
            "passed_count": len(state_residuals) - len(failed_state_ids),
            "failed_count": len(failed_state_ids),
            "failed_state_ids": failed_state_ids,
            "passed": direct_passed and not failed_state_ids,
        }
    )

    report = build_or_update_projector_report(
        phase_root=phase_root,
        setup=setup,
        registry=registry,
        section_name="idempotency_summary",
        section_summary=summary,
    )
    csv_path = phase_root / setup["outputs"]["projector_residuals"]
    without_direct = replace_rows(read_csv(csv_path), "direct_projector", [rows[0]])
    merged_rows = replace_rows(without_direct, "statewise_projector", rows[1:])
    write_csv(csv_path, merged_rows, ["test_type", "state_id", "residual", "norm_type", "passed"])
    write_json_report(phase_root / setup["outputs"]["projector_validation"], report)

    print("Phase 3 projector idempotency validation passed" if summary["passed"] else "Phase 3 projector idempotency validation failed")
    return 0 if summary["passed"] else 1


def _synthetic_states(state_count: int, state_dimension: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(state_dimension)] for _ in range(state_count)]


if __name__ == "__main__":
    raise SystemExit(main())
