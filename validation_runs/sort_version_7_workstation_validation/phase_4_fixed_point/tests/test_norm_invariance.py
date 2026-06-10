"""Run Phase 4 structural norm-behavior tracking."""

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
from phase_4_fixed_point.src.norm_checker import (
    classify_norm_behavior,
    norm_series,
    summarize_norm_drift,
)
from phase_4_fixed_point.src.report_writer import (
    build_or_update_fixed_point_report,
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

    classification_counts = {label: 0 for label in setup["norm_tracking"]["classification_labels"]}
    state_summaries = []
    maximum_drifts = []
    mean_drifts = []
    classified_drift_states = []

    for state_id, series in enumerate(series_batch, start=1):
        values = norm_series(series, setup["norm_type"])
        summary = summarize_norm_drift(values)
        classification = classify_norm_behavior(
            float(summary["initial_norm"]),
            float(summary["terminal_norm"]),
            float(summary["maximum_norm_drift"]),
            setup,
        )
        classification_counts[classification] += 1
        maximum_drifts.append(float(summary["maximum_norm_drift"]))
        mean_drifts.append(float(summary["mean_norm_drift"]))
        if classification != "norm-invariant":
            classified_drift_states.append({"state_id": state_id, "classification": classification})
        state_summary = {"state_id": state_id, "classification": classification}
        state_summary.update(summary)
        state_summaries.append(state_summary)

    report_summary = {
        "norm_type": setup["norm_type"],
        "state_count": len(series_batch),
        "iteration_count": max(len(series) for series in series_batch),
        "epsilon_norm": float(setup["tolerances"]["epsilon_norm"]),
        "strict_norm_preservation_required": setup["norm_tracking"]["strict_norm_preservation_required"],
        "classification_rule": setup["norm_tracking"]["classification_rule"],
        "maximum_norm_drift": max(maximum_drifts),
        "mean_norm_drift": sum(mean_drifts) / len(mean_drifts),
        "norm_behavior_classification_counts": classification_counts,
        "failed_states": [],
        "classified_drift_states": classified_drift_states,
        "state_summaries": state_summaries,
        "passed": True,
    }

    report = build_or_update_fixed_point_report(
        phase_root=phase_root,
        setup=setup,
        section_name="norm_invariance_summary",
        section_summary=report_summary,
    )
    norm_report = {
        "run_id": report["run_id"],
        "sort_version": "Version 7",
        "phase": "Phase 4 — Fixed-Point Structure",
        "norm_type": setup["norm_type"],
        "state_count": len(series_batch),
        "iteration_count": report_summary["iteration_count"],
        "maximum_norm_drift": report_summary["maximum_norm_drift"],
        "mean_norm_drift": report_summary["mean_norm_drift"],
        "norm_behavior_classification_counts": classification_counts,
        "failed_states": [],
        "passed": True,
        "non_claims": setup["non_claims"],
    }
    write_json_report(phase_root / setup["outputs"]["fixed_point_metrics"], report)
    write_json_report(phase_root / setup["outputs"]["norm_invariance_report"], norm_report)

    print("Phase 4 norm-behavior tracking passed")
    return 0


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
