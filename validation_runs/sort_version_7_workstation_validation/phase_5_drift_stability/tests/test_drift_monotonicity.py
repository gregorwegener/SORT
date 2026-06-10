"""Run Phase 5 drift monotonicity validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_3_global_projector.src.global_projector import construct_global_projector
from phase_5_drift_stability.src.drift_metric import (
    build_metric_definition,
    drift_profile,
    drift_value,
    normalized_drift_value,
    write_drift_metric_definition,
)
from phase_5_drift_stability.src.report_writer import (
    DRIFT_PROFILE_FIELDS,
    read_csv,
    replace_rows,
    write_csv,
    write_json_report,
)
from phase_5_drift_stability.src.synthetic_reference_generator import (
    build_reference_catalog,
    generate_all_references,
    generate_kernel_vector,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup, registry, seed_config, phase3_setup, run_manifest = _load_inputs(phase_root)
    projector = _projector_context(setup, registry, phase3_setup)
    references = generate_all_references(setup, int(seed_config["global_seed"]))

    rows = drift_profile(references, projector, setup)
    class_stats = {}
    for reference_class in setup["synthetic_references"]["classes"]:
        class_refs = [item for item in references if item["reference_class"] == reference_class]
        drift_values = [
            drift_value(item["state"], projector, setup["norm_type"])
            for item in class_refs
        ]
        normalized_values = [
            normalized_drift_value(item["state"], projector, setup["norm_type"])
            for item in class_refs
        ]
        class_stats[reference_class] = {
            "count": len(class_refs),
            "mean_drift": sum(drift_values) / len(drift_values),
            "maximum_drift": max(drift_values),
            "mean_normalized_drift": sum(normalized_values) / len(normalized_values),
            "maximum_normalized_drift": max(normalized_values),
        }

    ordering = (
        class_stats["projector_invariant"]["mean_drift"]
        <= class_stats["slightly_violated"]["mean_drift"]
        <= class_stats["strongly_violated"]["mean_drift"]
    )
    failed_comparisons = [] if ordering else ["projector_invariant <= slightly_violated <= strongly_violated"]
    monotonicity_summary = {
        "expected_ordering": "projector_invariant <= slightly_violated <= strongly_violated",
        "class_statistics": class_stats,
        "ordering_status": ordering,
        "failed_class_comparisons": failed_comparisons,
        "passed": ordering,
    }

    catalog = build_reference_catalog(
        run_id=run_manifest.get("run_id", "unavailable"),
        config=setup,
        seed=int(seed_config["global_seed"]),
        references=references,
    )
    metric_definition = build_metric_definition(
        run_id=run_manifest.get("run_id", "unavailable"),
        config=setup,
        monotonicity_summary=monotonicity_summary,
        invariance_summary=None,
        gate_5_passed=False,
    )

    drift_csv = phase_root / setup["outputs"]["drift_profiles"]
    write_csv(
        drift_csv,
        replace_rows(read_csv(drift_csv), "drift_monotonicity", rows),
        DRIFT_PROFILE_FIELDS,
    )
    write_json_report(phase_root / setup["outputs"]["synthetic_reference_catalog"], catalog)
    write_drift_metric_definition(
        phase_root / setup["outputs"]["drift_metric_definition"],
        metric_definition,
    )

    print("Phase 5 drift monotonicity validation passed" if ordering else "Phase 5 drift monotonicity validation failed")
    return 0 if ordering else 1


def _load_inputs(phase_root: Path):
    setup = json.loads((phase_root / "config" / "phase_5_setup.json").read_text(encoding="utf-8"))
    registry = json.loads((phase_root / setup["inputs"]["operator_registry"]).read_text(encoding="utf-8"))
    global_ref = json.loads((phase_root / setup["inputs"]["global_projector_ref"]).read_text(encoding="utf-8"))
    phase3_setup = json.loads(
        (phase_root / global_ref["phase_3_dependencies"]["phase_3_setup"])
        .resolve()
        .read_text(encoding="utf-8")
    )
    seed_config = json.loads((phase_root / setup["synthetic_references"]["seed_reference"]).resolve().read_text(encoding="utf-8"))
    run_manifest = json.loads((phase_root / setup["inputs"]["run_manifest"]).resolve().read_text(encoding="utf-8"))
    return setup, registry, seed_config, phase3_setup, run_manifest


def _projector_context(setup, registry, phase3_setup):
    context = dict(setup)
    context["_kernel_vector"] = generate_kernel_vector(setup)
    context["_global_projector"] = construct_global_projector(registry, phase3_setup)
    return context


if __name__ == "__main__":
    raise SystemExit(main())
