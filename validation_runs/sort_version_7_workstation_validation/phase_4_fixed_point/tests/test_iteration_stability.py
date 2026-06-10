"""Run Phase 4 deterministic repeatability and perturbation-response validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_3_global_projector.src.global_projector import construct_global_projector
from phase_4_fixed_point.src.fixed_point_iteration import (
    apply_H,
    apply_pi_kappa,
    generate_initial_states,
    kernel_profile_from_phase2_config,
    run_fixed_point_batch,
)
from phase_4_fixed_point.src.report_writer import (
    build_or_update_fixed_point_report,
    write_json_report,
)
from phase_4_fixed_point.src.stability_classifier import (
    apply_perturbation,
    classify_stability,
    compare_repeat_runs,
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
    states_a = generate_initial_states(setup, seed_config)
    states_b = generate_initial_states(setup, seed_config)
    series_a = run_fixed_point_batch(states_a, projection_fn, setup)
    series_b = run_fixed_point_batch(states_b, projection_fn, setup)

    repeatability_residuals = [
        compare_repeat_runs(series_a[index], series_b[index], setup["norm_type"])
        for index in range(len(series_a))
    ]
    epsilon_repeatability = float(setup["tolerances"]["epsilon_repeatability"])
    repeatability_passed = max(repeatability_residuals) < epsilon_repeatability

    scale = float(setup["stability"]["perturbation_scale"])
    classification_counts = {label: 0 for label in setup["stability"]["classification_labels"]}
    perturbation_cases = []
    divergent_cases = []
    for index, state in enumerate(states_a, start=1):
        perturbed_state = apply_perturbation(
            state,
            scale,
            int(seed_config["global_seed"]) + index,
        )
        perturbed_series = run_fixed_point_batch([perturbed_state], projection_fn, setup)[0]
        result = classify_stability(series_a[index - 1], perturbed_series, setup)
        classification_counts[result["classification"]] += 1
        case = {
            "state_id": index,
            "terminal_distance": result["terminal_distance"],
            "classification": result["classification"],
            "passed": result["passed"],
        }
        perturbation_cases.append(case)
        if not result["passed"]:
            divergent_cases.append(case)

    stability_passed = repeatability_passed and not divergent_cases
    summary = {
        "repeatability_summary": {
            "repeat_same_seed": setup["stability"]["repeat_same_seed"],
            "epsilon_repeatability": epsilon_repeatability,
            "maximum_residual": max(repeatability_residuals),
            "mean_residual": sum(repeatability_residuals) / len(repeatability_residuals),
            "passed": repeatability_passed,
        },
        "perturbation_summary": {
            "perturbation_test": setup["stability"]["perturbation_test"],
            "perturbation_scale": scale,
            "classification_rule": setup["stability"]["classification_rule"],
            "classification_counts": classification_counts,
            "cases": perturbation_cases,
            "divergent_cases": divergent_cases,
            "passed": not divergent_cases,
        },
        "stability_classification_counts": classification_counts,
        "divergent_cases": divergent_cases,
        "passed": stability_passed,
    }

    report = build_or_update_fixed_point_report(
        phase_root=phase_root,
        setup=setup,
        section_name="iteration_stability_summary",
        section_summary=summary,
    )
    stability_report = {
        "run_id": report["run_id"],
        "sort_version": "Version 7",
        "phase": "Phase 4 — Fixed-Point Structure",
        "repeatability_summary": summary["repeatability_summary"],
        "perturbation_summary": summary["perturbation_summary"],
        "perturbation_scale": scale,
        "stability_classification_counts": classification_counts,
        "divergent_cases": divergent_cases,
        "passed": stability_passed,
        "non_claims": setup["non_claims"],
    }
    write_json_report(phase_root / setup["outputs"]["fixed_point_metrics"], report)
    write_json_report(phase_root / setup["outputs"]["iteration_stability"], stability_report)

    print("Phase 4 iteration-stability validation passed" if stability_passed else "Phase 4 iteration-stability validation failed")
    return 0 if stability_passed else 1


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
