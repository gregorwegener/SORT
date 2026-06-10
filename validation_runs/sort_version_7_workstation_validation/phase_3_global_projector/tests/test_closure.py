"""Run Phase 3 closure validation for projected synthetic states."""

from __future__ import annotations

import json
import random
from pathlib import Path

from phase_3_global_projector.src.closure_checker import (
    check_admissible_state,
    closure_violation_rate,
)
from phase_3_global_projector.src.global_projector import (
    apply_global_projector,
    construct_global_projector,
    load_projector_config,
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
    states = _synthetic_states(
        int(setup["synthetic_states"]["state_count"]),
        int(setup["synthetic_states"]["state_dimension"]),
        int(seed_config["global_seed"]),
    )

    results = []
    rows = []
    borderline_cases = []
    for state_id, state in enumerate(states, start=1):
        projected = apply_global_projector(state, projector)
        result = check_admissible_state(projected, setup["admissible_space"], norm_type)
        result["state_id"] = state_id
        results.append(result)
        if result["borderline"]:
            borderline_cases.append({"state_id": state_id, "norm": result["norm"]})
        residual = max(0.0, result["norm"] - result["max_norm"]) if result["finite"] else float("inf")
        rows.append(
            {
                "test_type": "closure",
                "state_id": state_id,
                "residual": f"{residual:.17g}",
                "norm_type": norm_type,
                "passed": str(result["passed"]).lower(),
            }
        )

    violation_count = sum(1 for result in results if not result["passed"])
    violation_rate = closure_violation_rate(results)
    summary = {
        "admissible_space": setup["admissible_space"],
        "synthetic_state_count": len(states),
        "finite_state_count": sum(1 for result in results if result["finite"]),
        "norm_bound_pass_count": sum(1 for result in results if result["within_bound"]),
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "borderline_cases": borderline_cases,
        "epsilon_closure": float(setup["tolerances"]["epsilon_closure"]),
        "norm_type": norm_type,
        "seed_reference": setup["synthetic_states"]["seed_reference"],
        "global_seed": seed_config["global_seed"],
        "passed": violation_count == 0,
    }

    report = build_or_update_projector_report(
        phase_root=phase_root,
        setup=setup,
        registry=registry,
        section_name="closure_summary",
        section_summary=summary,
    )
    closure_report = {
        "run_id": report["run_id"],
        "sort_version": "Version 7",
        "phase": "Phase 3 — Global Projector",
        "admissible_space_definition": setup["admissible_space"],
        "synthetic_state_count": len(states),
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "borderline_cases": borderline_cases,
        "passed": summary["passed"],
        "non_claims": setup["non_claims"],
    }
    csv_path = phase_root / setup["outputs"]["projector_residuals"]
    write_csv(
        csv_path,
        replace_rows(read_csv(csv_path), "closure", rows),
        ["test_type", "state_id", "residual", "norm_type", "passed"],
    )
    write_json_report(phase_root / setup["outputs"]["projector_validation"], report)
    write_json_report(phase_root / setup["outputs"]["closure_test_report"], closure_report)

    print("Phase 3 closure validation passed" if summary["passed"] else "Phase 3 closure validation failed")
    return 0 if summary["passed"] else 1


def _synthetic_states(state_count: int, state_dimension: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(state_dimension)] for _ in range(state_count)]


if __name__ == "__main__":
    raise SystemExit(main())
