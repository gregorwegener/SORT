"""Perturbation-response utilities for SORT Version 7 Phase 5."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

from phase_5_drift_stability.src.drift_metric import (
    classify_drift,
    drift_value,
    effective_projector,
)


def apply_perturbation(state: Sequence[float], strength: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [float(value) + rng.uniform(-float(strength), float(strength)) for value in state]


def run_response(
    state: Sequence[float],
    perturbation_strength: float,
    projection_fn: Callable[[Sequence[float], dict[str, Any]], list[float]],
    projector: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    current = apply_perturbation(
        state,
        perturbation_strength,
        int(config.get("_response_seed", 117666)),
    )
    trajectory = []
    initial_drift = drift_value(current, projector, config["norm_type"])
    for iteration in range(1, int(config["projection_iterations"]) + 1):
        current = projection_fn(current, projector)
        terminal_drift = drift_value(current, projector, config["norm_type"])
        trajectory.append({"iteration": iteration, "drift_value": terminal_drift})
    response = {
        "initial_drift": initial_drift,
        "terminal_drift": trajectory[-1]["drift_value"] if trajectory else initial_drift,
        "trajectory": trajectory,
        "perturbation_strength": perturbation_strength,
    }
    response["stability_label"] = classify_stability(response, config)
    response["response_classification"] = (
        "divergence" if response["stability_label"] == "unstable" else "return"
    )
    return response


def classify_stability(response: dict[str, Any], config: dict[str, Any]) -> str:
    terminal_drift = float(response["terminal_drift"])
    if not math.isfinite(terminal_drift):
        return "unstable"
    drift_label = classify_drift(terminal_drift, config["drift_thresholds"])
    if drift_label == "negligible":
        return "stable"
    if drift_label == "moderate":
        return "marginal"
    return "unstable"


def summarize_stability(responses: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"stable": 0, "marginal": 0, "unstable": 0}
    divergent_cases = []
    for response in responses:
        label = response["stability_label"]
        counts[label] += 1
        if label == "unstable":
            divergent_cases.append(response)
    return {
        "response_count": len(responses),
        "stability_classification_counts": counts,
        "divergent_cases": divergent_cases,
        "passed": not divergent_cases,
    }


def projection_response(state: Sequence[float], projector: dict[str, Any]) -> list[float]:
    return effective_projector(state, projector)
