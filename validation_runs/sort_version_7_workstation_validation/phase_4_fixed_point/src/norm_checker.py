"""Norm tracking utilities for SORT Version 7 Phase 4."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence


def state_norm(state: Sequence[float], norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in state))
    if norm_key == "max":
        return max((abs(float(value)) for value in state), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def norm_series(iteration_series: list[Any], norm_type: str) -> list[float]:
    return [state_norm(_extract_state(entry), norm_type) for entry in iteration_series]


def norm_drift(norm_values: list[float]) -> list[float]:
    if not norm_values:
        return []
    initial = norm_values[0]
    return [abs(value - initial) for value in norm_values]


def summarize_norm_drift(norm_values: list[float]) -> dict[str, float | int]:
    drift_values = norm_drift(norm_values)
    if not drift_values:
        return {
            "iteration_count": 0,
            "initial_norm": 0.0,
            "terminal_norm": 0.0,
            "maximum_norm_drift": 0.0,
            "mean_norm_drift": 0.0,
        }
    return {
        "iteration_count": len(norm_values),
        "initial_norm": norm_values[0],
        "terminal_norm": norm_values[-1],
        "maximum_norm_drift": max(drift_values),
        "mean_norm_drift": statistics.fmean(drift_values),
    }


def classify_norm_behavior(
    initial_norm: float,
    terminal_norm: float,
    max_drift: float,
    config: dict[str, Any],
) -> str:
    epsilon = float(config["tolerances"]["epsilon_norm"])
    strict_required = bool(config["norm_tracking"]["strict_norm_preservation_required"])
    if max_drift < epsilon:
        return "norm-invariant"
    if not strict_required and terminal_norm <= initial_norm:
        return "contractive-projection"
    return "norm-drifting"


def _extract_state(entry: Any) -> Sequence[float]:
    if isinstance(entry, dict):
        return entry["state"]
    return entry
