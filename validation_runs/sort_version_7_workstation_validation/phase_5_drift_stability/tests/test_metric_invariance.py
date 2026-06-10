"""Run Phase 5 drift metric scale-behavior validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_3_global_projector.src.global_projector import construct_global_projector
from phase_5_drift_stability.src.drift_metric import (
    build_metric_definition,
    classify_drift,
    drift_value,
    normalized_drift_value,
    write_drift_metric_definition,
)
from phase_5_drift_stability.src.metric_invariance import (
    compare_metric_under_rescaling,
    summarize_invariance,
)
from phase_5_drift_stability.src.report_writer import (
    DRIFT_PROFILE_FIELDS,
    load_json,
    read_csv,
    replace_rows,
    write_csv,
    write_json_report,
)
from phase_5_drift_stability.src.synthetic_reference_generator import generate_kernel_vector


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup, registry, phase3_setup, run_manifest, catalog = _load_inputs(phase_root)
    projector = _projector_context(setup, registry, phase3_setup)
    references = catalog["references"]

    all_results = []
    rows = []
    for reference in references:
        results = compare_metric_under_rescaling(
            reference["state"],
            projector,
            setup["norm_type"],
            [float(value) for value in setup["metric_invariance"]["rescaling_factors"]],
        )
        for result in results:
            enriched = {
                "state_id": reference["state_id"],
                "reference_class": reference["reference_class"],
            }
            enriched.update(result)
            all_results.append(enriched)
            rows.append(
                {
                    "test_type": "metric_invariance",
                    "state_id": reference["state_id"],
                    "reference_class": reference["reference_class"],
                    "iteration": "rescale",
                    "drift_value": f"{result['transformed_drift']:.17g}",
                    "normalized_drift_value": f"{result['transformed_normalized_drift']:.17g}",
                    "drift_label": classify_drift(result["transformed_drift"], setup["drift_thresholds"]),
                    "stability_label": "untested",
                    "perturbation_strength": "",
                    "rescaling_factor": result["rescaling_factor"],
                    "metric_status": "passed" if result["passed"] else "failed",
                }
            )

    invariance_summary = summarize_invariance(all_results)
    invariance_summary.update(
        {
            "rescaling_factors": setup["metric_invariance"]["rescaling_factors"],
            "expected_raw_behavior": setup["metric_invariance"]["expected_raw_behavior"],
            "expected_normalized_behavior": setup["metric_invariance"]["expected_normalized_behavior"],
            "results": all_results,
        }
    )

    existing_metric = load_json(phase_root / setup["outputs"]["drift_metric_definition"])
    existing_stability = load_json(phase_root / setup["outputs"]["stability_response"])
    monotonicity_summary = existing_metric.get("drift_monotonicity_summary")
    stability_passed = bool(existing_stability.get("passed"))
    gate_5_passed = (
        bool(monotonicity_summary and monotonicity_summary.get("passed"))
        and stability_passed
        and bool(invariance_summary["passed"])
    )
    metric_definition = build_metric_definition(
        run_id=run_manifest.get("run_id", "unavailable"),
        config=setup,
        monotonicity_summary=monotonicity_summary,
        invariance_summary=invariance_summary,
        gate_5_passed=gate_5_passed,
    )
    existing_stability["gate_5_passed"] = gate_5_passed

    drift_csv = phase_root / setup["outputs"]["drift_profiles"]
    write_csv(
        drift_csv,
        replace_rows(read_csv(drift_csv), "metric_invariance", rows),
        DRIFT_PROFILE_FIELDS,
    )
    write_drift_metric_definition(
        phase_root / setup["outputs"]["drift_metric_definition"],
        metric_definition,
    )
    write_json_report(phase_root / setup["outputs"]["stability_response"], existing_stability)

    print("Phase 5 metric invariance validation passed" if invariance_summary["passed"] else "Phase 5 metric invariance validation failed")
    return 0 if invariance_summary["passed"] else 1


def _load_inputs(phase_root: Path):
    setup = json.loads((phase_root / "config" / "phase_5_setup.json").read_text(encoding="utf-8"))
    registry = json.loads((phase_root / setup["inputs"]["operator_registry"]).read_text(encoding="utf-8"))
    global_ref = json.loads((phase_root / setup["inputs"]["global_projector_ref"]).read_text(encoding="utf-8"))
    phase3_setup = json.loads(
        (phase_root / global_ref["phase_3_dependencies"]["phase_3_setup"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    run_manifest = json.loads((phase_root / setup["inputs"]["run_manifest"]).resolve().read_text(encoding="utf-8"))
    catalog = json.loads((phase_root / setup["outputs"]["synthetic_reference_catalog"]).read_text(encoding="utf-8"))
    return setup, registry, phase3_setup, run_manifest, catalog


def _projector_context(setup, registry, phase3_setup):
    context = dict(setup)
    context["_kernel_vector"] = generate_kernel_vector(setup)
    context["_global_projector"] = construct_global_projector(registry, phase3_setup)
    return context


if __name__ == "__main__":
    raise SystemExit(main())
