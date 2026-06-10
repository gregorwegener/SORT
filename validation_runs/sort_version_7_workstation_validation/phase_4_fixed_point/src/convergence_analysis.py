"""Convergence analysis for SORT Version 7 Phase 4 iteration series."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence


def successive_differences(series: list[Any], norm_type: str) -> list[float]:
    states = [_extract_state(entry) for entry in series]
    differences = []
    for index in range(1, len(states)):
        differences.append(
            _vector_norm(
                [states[index][column] - states[index - 1][column] for column in range(len(states[index]))],
                norm_type,
            )
        )
    return differences


def classify_convergence(differences: list[float], config: dict[str, Any]) -> str:
    if not differences:
        return "neutral"
    epsilon = float(config["tolerances"]["epsilon_convergence"])
    if differences[-1] < epsilon:
        return "convergent"
    if len(differences) >= 4 and all(differences[index] > differences[index - 1] for index in range(1, len(differences))):
        return "divergent"
    return "neutral"


def summarize_convergence(differences: list[float]) -> dict[str, float | int]:
    if not differences:
        return {
            "count": 0,
            "final_residual": 0.0,
            "maximum_residual": 0.0,
            "mean_residual": 0.0,
            "median_residual": 0.0,
        }
    return {
        "count": len(differences),
        "final_residual": differences[-1],
        "maximum_residual": max(differences),
        "mean_residual": statistics.fmean(differences),
        "median_residual": statistics.median(differences),
    }


def detect_oscillation(series: list[Any], norm_type: str) -> bool:
    states = [_extract_state(entry) for entry in series]
    if len(states) < 4:
        return False
    epsilon = 1e-12
    for index in range(2, len(states)):
        distance_to_two_back = _vector_norm(
            [states[index][column] - states[index - 2][column] for column in range(len(states[index]))],
            norm_type,
        )
        distance_to_previous = _vector_norm(
            [states[index][column] - states[index - 1][column] for column in range(len(states[index]))],
            norm_type,
        )
        if distance_to_two_back < epsilon and distance_to_previous >= epsilon:
            return True
    return False


def _extract_state(entry: Any) -> Sequence[float]:
    if isinstance(entry, dict):
        return entry["state"]
    return entry


def _vector_norm(values: Sequence[float], norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in values))
    if norm_key == "max":
        return max((abs(float(value)) for value in values), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")
