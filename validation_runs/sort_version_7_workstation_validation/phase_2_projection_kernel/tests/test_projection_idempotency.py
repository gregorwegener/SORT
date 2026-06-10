"""Run Phase 2 projection idempotency validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from phase_2_projection_kernel.src.kernel_definition import kernel_profile, load_kernel_definition
from phase_2_projection_kernel.src.projection_metrics import summarize_residuals
from phase_2_projection_kernel.src.projection_operator import projection_idempotency_residual
from phase_2_projection_kernel.src.report_writer import (
    build_or_update_report,
    write_csv,
    write_json_report,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup = json.loads((phase_root / "config" / "phase_2_setup.json").read_text(encoding="utf-8"))
    kernel_definition = load_kernel_definition(phase_root / setup["inputs"]["kernel_definitions"])
    seed_config = json.loads(
        (phase_root / setup["synthetic_states"]["seed_reference"])
        .resolve()
        .read_text(encoding="utf-8")
    )

    sigma_0 = float(kernel_definition["kernel"]["scale_parameter"]["value"])
    xi_values = [float(value) for value in setup["xi_grid"]["values"]]
    k_values = [float(value) for value in setup["k_grid"]["values"]]
    state_count = int(setup["synthetic_states"]["state_count"])
    state_dimension = int(setup["synthetic_states"]["state_dimension"])
    seed = int(seed_config["global_seed"])
    tolerance = float(setup["tolerances"]["epsilon_projection"])
    norm_type = setup["norm_type"]

    states = _synthetic_states(state_count, state_dimension, seed)
    rows = []
    residuals = []
    failed_cases = []

    for state_id, state in enumerate(states, start=1):
        for k_value in k_values:
            kernel_values = kernel_profile(xi_values, sigma_0, k_value)
            residual = projection_idempotency_residual(state, kernel_values, norm_type)
            passed = residual < tolerance
            residuals.append(residual)
            if not passed:
                failed_cases.append({"state_id": state_id, "k_value": k_value, "residual": residual})
            rows.append(
                {
                    "state_id": state_id,
                    "k_value": k_value,
                    "residual": f"{residual:.17g}",
                    "norm_type": norm_type,
                    "passed": str(passed).lower(),
                }
            )

    summary = summarize_residuals(residuals)
    summary.update(
        {
            "epsilon_projection": tolerance,
            "norm_type": norm_type,
            "state_count": state_count,
            "state_dimension": state_dimension,
            "seed_reference": setup["synthetic_states"]["seed_reference"],
            "global_seed": seed,
            "k_values": k_values,
            "passed_count": len(residuals) - len(failed_cases),
            "failed_count": len(failed_cases),
            "failed_cases": failed_cases,
            "passed": not failed_cases,
        }
    )

    report = build_or_update_report(
        phase_root=phase_root,
        setup=setup,
        section_name="projection_idempotency_summary",
        section_summary=summary,
    )
    write_json_report(phase_root / setup["outputs"]["kernel_norm_report"], report)
    write_csv(
        phase_root / setup["outputs"]["projection_residuals"],
        rows,
        ["state_id", "k_value", "residual", "norm_type", "passed"],
    )

    print("Phase 2 projection idempotency validation passed" if summary["passed"] else "Phase 2 projection idempotency validation failed")
    return 0 if summary["passed"] else 1


def _synthetic_states(state_count: int, state_dimension: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [rng.uniform(-1.0, 1.0) for _ in range(state_dimension)]
        for _ in range(state_count)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
