"""Structural drift metric for SORT Version 7 Phase 5."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from phase_5_drift_stability.src.synthetic_reference_generator import generate_kernel_vector


def effective_projector(state: Sequence[float], config: dict[str, Any]) -> list[float]:
    kernel_vector = config.get("_kernel_vector") or generate_kernel_vector(config)
    coefficient = sum(float(kernel_vector[index]) * float(state[index]) for index in range(len(kernel_vector)))
    projected = [coefficient * float(value) for value in kernel_vector]
    global_projector = config.get("_global_projector")
    if global_projector is None:
        return projected
    return _apply_matrix(global_projector, projected)


def drift_value(state: Sequence[float], projector: dict[str, Any], norm_type: str) -> float:
    projected = effective_projector(state, projector)
    return _vector_norm(
        [projected[index] - float(state[index]) for index in range(len(projected))],
        norm_type,
    )


def normalized_drift_value(state: Sequence[float], projector: dict[str, Any], norm_type: str) -> float:
    drift = drift_value(state, projector, norm_type)
    state_norm = _vector_norm(state, norm_type)
    return drift / max(state_norm, 1.0e-12)


def classify_drift(value: float, thresholds: dict[str, float]) -> str:
    if not math.isfinite(value):
        return "structural"
    if value <= float(thresholds["negligible_max"]):
        return "negligible"
    if value <= float(thresholds["moderate_max"]):
        return "moderate"
    return "structural"


def drift_profile(
    states: list[dict[str, Any]],
    projector: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for reference in states:
        drift = drift_value(reference["state"], projector, config["norm_type"])
        normalized = normalized_drift_value(reference["state"], projector, config["norm_type"])
        rows.append(
            {
                "test_type": "drift_monotonicity",
                "state_id": reference["state_id"],
                "reference_class": reference["reference_class"],
                "iteration": 0,
                "drift_value": f"{drift:.17g}",
                "normalized_drift_value": f"{normalized:.17g}",
                "drift_label": classify_drift(drift, config["drift_thresholds"]),
                "stability_label": "untested",
                "perturbation_strength": "",
                "rescaling_factor": "",
                "metric_status": "baseline",
            }
        )
    return rows


def write_drift_metric_definition(path: str | Path, definition: dict[str, Any]) -> None:
    definition_path = Path(path)
    definition_path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.write_text(
        json.dumps(definition, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def build_metric_definition(
    *,
    run_id: str,
    config: dict[str, Any],
    monotonicity_summary: dict[str, Any] | None = None,
    invariance_summary: dict[str, Any] | None = None,
    gate_5_passed: bool = False,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sort_version": "Version 7",
        "phase": "Phase 5 — Drift and Stability",
        "formal_metric_definition": config["drift_metric"]["definition"],
        "effective_projector_definition": config["effective_drift_projector"]["definition"],
        "effective_projector_rationale": config["effective_drift_projector"]["rationale"],
        "norm_type": config["norm_type"],
        "threshold_policy": config["drift_thresholds"],
        "dimensionless": config["drift_metric"]["dimensionless"],
        "raw_drift_scale_behavior": config["metric_invariance"]["expected_raw_behavior"],
        "normalized_drift_scale_behavior": config["metric_invariance"]["expected_normalized_behavior"],
        "scale_robustness_statement": config["drift_metric"]["scale_behavior"],
        "metric_invariance_summary": invariance_summary,
        "drift_monotonicity_summary": monotonicity_summary,
        "passed": bool(monotonicity_summary and monotonicity_summary.get("passed"))
        and (invariance_summary is None or bool(invariance_summary.get("passed"))),
        "gate_5_passed": gate_5_passed,
        "non_claims": config["non_claims"],
    }


def _apply_matrix(matrix: Sequence[Sequence[float]], state: Sequence[float]) -> list[float]:
    return [
        sum(float(matrix[row][column]) * float(state[column]) for column in range(len(state)))
        for row in range(len(matrix))
    ]


def _vector_norm(values: Sequence[float], norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in values))
    if norm_key == "max":
        return max((abs(float(value)) for value in values), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")
