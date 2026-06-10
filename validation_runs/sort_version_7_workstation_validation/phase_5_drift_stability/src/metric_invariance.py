"""Drift metric scale-behavior checks for SORT Version 7 Phase 5."""

from __future__ import annotations

from typing import Any, Sequence

from phase_5_drift_stability.src.drift_metric import (
    drift_value,
    normalized_drift_value,
)


def rescale_state(state: Sequence[float], factor: float) -> list[float]:
    return [float(factor) * float(value) for value in state]


def compare_metric_under_rescaling(
    state: Sequence[float],
    projector: dict[str, Any],
    norm_type: str,
    factors: list[float],
) -> list[dict[str, Any]]:
    baseline_drift = drift_value(state, projector, norm_type)
    baseline_normalized = normalized_drift_value(state, projector, norm_type)
    results = []
    for factor in factors:
        transformed = rescale_state(state, factor)
        transformed_drift = drift_value(transformed, projector, norm_type)
        transformed_normalized = normalized_drift_value(transformed, projector, norm_type)
        raw_expected = abs(float(factor)) * baseline_drift
        raw_residual = abs(transformed_drift - raw_expected)
        normalized_residual = abs(transformed_normalized - baseline_normalized)
        passed = raw_residual <= 1.0e-9 and normalized_residual <= 1.0e-9
        results.append(
            {
                "rescaling_factor": factor,
                "baseline_drift": baseline_drift,
                "transformed_drift": transformed_drift,
                "baseline_normalized_drift": baseline_normalized,
                "transformed_normalized_drift": transformed_normalized,
                "raw_homogeneity_residual": raw_residual,
                "normalized_invariance_residual": normalized_residual,
                "expected_behavior": "raw_homogeneous_normalized_invariant",
                "passed": passed,
            }
        )
    return results


def summarize_invariance(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [result for result in results if not result["passed"]]
    return {
        "result_count": len(results),
        "maximum_raw_homogeneity_residual": max(
            (result["raw_homogeneity_residual"] for result in results),
            default=0.0,
        ),
        "maximum_normalized_invariance_residual": max(
            (result["normalized_invariance_residual"] for result in results),
            default=0.0,
        ),
        "failed_cases": failed,
        "passed": not failed,
    }
