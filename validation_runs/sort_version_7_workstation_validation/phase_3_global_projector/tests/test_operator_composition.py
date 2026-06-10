"""Run Phase 3 declared operator-composition stability validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from phase_3_global_projector.src.composition_checker import composition_residual
from phase_3_global_projector.src.global_projector import (
    construct_global_projector,
    load_projector_config,
)
from phase_3_global_projector.src.report_writer import (
    build_or_update_projector_report,
    summarize_residuals,
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
    tolerance = float(setup["tolerances"]["epsilon_composition"])
    states = _synthetic_states(
        int(setup["synthetic_states"]["state_count"]),
        int(setup["synthetic_states"]["state_dimension"]),
        int(seed_config["global_seed"]),
    )
    operators = {operator["operator_id"]: operator["matrix"] for operator in registry["operators"]}

    rows = []
    residuals = []
    unstable_pairs = set()
    pairs = setup["operator_composition"]["pairs"]
    for operator_id_i, operator_id_j in pairs:
        A = operators[int(operator_id_i)]
        B = operators[int(operator_id_j)]
        for state_id, state in enumerate(states, start=1):
            residual = composition_residual(projector, A, B, state, norm_type)
            passed = residual < tolerance
            residuals.append(residual)
            if not passed:
                unstable_pairs.add((operator_id_i, operator_id_j))
            rows.append(
                {
                    "operator_id_i": operator_id_i,
                    "operator_id_j": operator_id_j,
                    "state_id": state_id,
                    "residual": f"{residual:.17g}",
                    "norm_type": norm_type,
                    "passed": str(passed).lower(),
                }
            )

    summary = summarize_residuals(residuals)
    summary.update(
        {
            "mode": setup["operator_composition"]["mode"],
            "assume_global_commutativity": setup["operator_composition"]["assume_global_commutativity"],
            "global_commutativity_statement": "Global commutativity is not assumed by this Phase 3 composition check.",
            "evaluated_operator_pairs": pairs,
            "pair_count": len(pairs),
            "state_count_per_pair": len(states),
            "expected_row_count": len(pairs) * len(states),
            "epsilon_composition": tolerance,
            "norm_type": norm_type,
            "passed_count": len(residuals) - len(unstable_pairs),
            "failed_count": sum(1 for residual in residuals if residual >= tolerance),
            "unstable_pairs": [list(pair) for pair in sorted(unstable_pairs)],
            "passed": not unstable_pairs,
        }
    )

    report = build_or_update_projector_report(
        phase_root=phase_root,
        setup=setup,
        registry=registry,
        section_name="composition_summary",
        section_summary=summary,
    )
    write_csv(
        phase_root / setup["outputs"]["composition_stability"],
        rows,
        ["operator_id_i", "operator_id_j", "state_id", "residual", "norm_type", "passed"],
    )
    write_json_report(phase_root / setup["outputs"]["projector_validation"], report)

    print("Phase 3 operator-composition validation passed" if summary["passed"] else "Phase 3 operator-composition validation failed")
    return 0 if summary["passed"] else 1


def _synthetic_states(state_count: int, state_dimension: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(state_dimension)] for _ in range(state_count)]


if __name__ == "__main__":
    raise SystemExit(main())
