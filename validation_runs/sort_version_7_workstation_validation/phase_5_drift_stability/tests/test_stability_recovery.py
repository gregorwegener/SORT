"""Run Phase 5 perturbation-response and stability classification validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_3_global_projector.src.global_projector import construct_global_projector
from phase_5_drift_stability.src.drift_metric import (
    classify_drift,
    drift_value,
    normalized_drift_value,
)
from phase_5_drift_stability.src.report_writer import (
    DRIFT_PROFILE_FIELDS,
    read_csv,
    replace_rows,
    write_csv,
    write_json_report,
)
from phase_5_drift_stability.src.stability_response import (
    projection_response,
    run_response,
    summarize_stability,
)
from phase_5_drift_stability.src.synthetic_reference_generator import generate_kernel_vector


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup, registry, phase3_setup, run_manifest, catalog = _load_inputs(phase_root)
    projector = _projector_context(setup, registry, phase3_setup)
    references = catalog["references"]

    rows = []
    responses = []
    for reference in references:
        for strength_index, strength in enumerate(setup["perturbation_strengths"], start=1):
            runtime_config = dict(setup)
            runtime_config["_response_seed"] = (
                int(catalog["global_seed"]) + reference["state_id"] * 1000 + strength_index
            )
            response = run_response(
                reference["state"],
                float(strength),
                projection_response,
                projector,
                runtime_config,
            )
            response_record = {
                "state_id": reference["state_id"],
                "reference_class": reference["reference_class"],
                "perturbation_strength": float(strength),
                "initial_drift": response["initial_drift"],
                "terminal_drift": response["terminal_drift"],
                "response_classification": response["response_classification"],
                "stability_label": response["stability_label"],
                "trajectory": response["trajectory"],
            }
            responses.append(response_record)
            for entry in response["trajectory"]:
                drift_label = classify_drift(entry["drift_value"], setup["drift_thresholds"])
                rows.append(
                    {
                        "test_type": "stability_recovery",
                        "state_id": reference["state_id"],
                        "reference_class": reference["reference_class"],
                        "iteration": entry["iteration"],
                        "drift_value": f"{entry['drift_value']:.17g}",
                        "normalized_drift_value": f"{normalized_drift_value(reference['state'], projector, setup['norm_type']):.17g}",
                        "drift_label": drift_label,
                        "stability_label": response["stability_label"],
                        "perturbation_strength": strength,
                        "rescaling_factor": "",
                        "metric_status": response["response_classification"],
                    }
                )

    summary = summarize_stability(responses)
    stability_report = {
        "run_id": run_manifest.get("run_id", "unavailable"),
        "sort_version": "Version 7",
        "phase": "Phase 5 — Drift and Stability",
        "perturbation_strengths": setup["perturbation_strengths"],
        "state_count": len(references),
        "response_classifications": sorted({item["response_classification"] for item in responses}),
        "stability_classification_counts": summary["stability_classification_counts"],
        "responses": responses,
        "divergent_cases": summary["divergent_cases"],
        "passed": summary["passed"],
        "gate_5_stability_component_passed": summary["passed"],
        "non_claims": setup["non_claims"],
    }

    drift_csv = phase_root / setup["outputs"]["drift_profiles"]
    write_csv(
        drift_csv,
        replace_rows(read_csv(drift_csv), "stability_recovery", rows),
        DRIFT_PROFILE_FIELDS,
    )
    write_json_report(phase_root / setup["outputs"]["stability_response"], stability_report)

    print("Phase 5 stability recovery validation passed" if summary["passed"] else "Phase 5 stability recovery validation failed")
    return 0 if summary["passed"] else 1


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
