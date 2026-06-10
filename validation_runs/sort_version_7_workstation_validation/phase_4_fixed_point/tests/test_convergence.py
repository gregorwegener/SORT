"""Run Phase 4 convergence validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_3_global_projector.src.global_projector import construct_global_projector
from phase_4_fixed_point.src.convergence_analysis import (
    classify_convergence,
    detect_oscillation,
    successive_differences,
    summarize_convergence,
)
from phase_4_fixed_point.src.fixed_point_iteration import (
    apply_H,
    apply_pi_kappa,
    generate_initial_states,
    kernel_profile_from_phase2_config,
    run_fixed_point_batch,
)
from phase_4_fixed_point.src.report_writer import (
    build_or_update_fixed_point_report,
    write_csv,
    write_json_report,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup, registry, seed_config, phase2_setup, phase3_setup = _load_inputs(phase_root)
    projector = construct_global_projector(registry, phase3_setup)
    kernel_values = kernel_profile_from_phase2_config(
        phase2_setup,
        float(setup["iteration"]["k_value"]),
    )
    projection_fn = lambda state: apply_H(apply_pi_kappa(state, kernel_values), projector)
    states = generate_initial_states(setup, seed_config)
    series_batch = run_fixed_point_batch(states, projection_fn, setup)

    rows = []
    final_residuals = []
    maximum_residuals = []
    classification_counts = {label: 0 for label in setup["convergence"]["classification_labels"]}
    failed_state_ids = []
    divergent_state_ids = []

    for state_id, series in enumerate(series_batch, start=1):
        differences = successive_differences(series, setup["norm_type"])
        classification = (
            "oscillatory"
            if detect_oscillation(series, setup["norm_type"])
            else classify_convergence(differences, setup)
        )
        classification_counts[classification] += 1
        summary = summarize_convergence(differences)
        final_residuals.append(float(summary["final_residual"]))
        maximum_residuals.append(float(summary["maximum_residual"]))
        if classification == "divergent":
            divergent_state_ids.append(state_id)
        if classification not in {"convergent", "neutral"}:
            failed_state_ids.append(state_id)
        for entry in series:
            rows.append(
                {
                    "state_id": state_id,
                    "iteration": entry["iteration"],
                    "residual": f"{entry['residual']:.17g}",
                    "norm_value": f"{entry['norm_value']:.17g}",
                    "classification": classification,
                }
            )

    passed = not failed_state_ids
    summary = {
        "state_count": len(series_batch),
        "convergence_threshold": float(setup["tolerances"]["epsilon_convergence"]),
        "criterion": setup["convergence"]["criterion"],
        "iteration_rule": setup["iteration"]["iteration_rule"],
        "k_value": float(setup["iteration"]["k_value"]),
        "max_iterations": int(setup["iteration"]["max_iterations"]),
        "min_iterations": int(setup["iteration"]["min_iterations"]),
        "final_maximum_residual": max(final_residuals),
        "maximum_residual": max(maximum_residuals),
        "mean_final_residual": sum(final_residuals) / len(final_residuals),
        "classification_counts": classification_counts,
        "failed_state_ids": failed_state_ids,
        "divergent_state_ids": divergent_state_ids,
        "stop_reasons": sorted({series[-1]["stop_reason"] for series in series_batch}),
        "passed": passed,
    }

    report = build_or_update_fixed_point_report(
        phase_root=phase_root,
        setup=setup,
        section_name="convergence_summary",
        section_summary=summary,
    )
    write_csv(
        phase_root / setup["outputs"]["convergence_series"],
        rows,
        ["state_id", "iteration", "residual", "norm_value", "classification"],
    )
    write_json_report(phase_root / setup["outputs"]["fixed_point_metrics"], report)

    print("Phase 4 convergence validation passed" if passed else "Phase 4 convergence validation failed")
    return 0 if passed else 1


def _load_inputs(phase_root: Path):
    setup = json.loads((phase_root / "config" / "phase_4_setup.json").read_text(encoding="utf-8"))
    registry = json.loads((phase_root / setup["inputs"]["operator_registry"]).read_text(encoding="utf-8"))
    projection_ref = json.loads((phase_root / setup["inputs"]["projection_operator_ref"]).read_text(encoding="utf-8"))
    global_ref = json.loads((phase_root / setup["inputs"]["global_projector_ref"]).read_text(encoding="utf-8"))
    seed_config = json.loads((phase_root / setup["synthetic_states"]["seed_reference"]).resolve().read_text(encoding="utf-8"))
    phase2_setup = json.loads(
        (phase_root / projection_ref["phase_2_dependencies"]["phase_2_setup"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    phase3_setup = json.loads(
        (phase_root / global_ref["phase_3_dependencies"]["phase_3_setup"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    return setup, registry, seed_config, phase2_setup, phase3_setup


if __name__ == "__main__":
    raise SystemExit(main())
