"""Repeatability and perturbation classification for SORT Version 7 Phase 4."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


def compare_repeat_runs(
    series_a: list[Any],
    series_b: list[Any],
    norm_type: str,
) -> float:
    terminal_a = _extract_state(series_a[-1])
    terminal_b = _extract_state(series_b[-1])
    if len(terminal_a) != len(terminal_b):
        raise ValueError("Terminal states must have matching dimensions.")
    return _vector_norm(
        [terminal_a[index] - terminal_b[index] for index in range(len(terminal_a))],
        norm_type,
    )


def apply_perturbation(state: Sequence[float], scale: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [float(value) + rng.uniform(-float(scale), float(scale)) for value in state]


def classify_stability(
    reference_series: list[Any],
    perturbed_series: list[Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    norm_type = config["norm_type"]
    scale = float(config["stability"]["perturbation_scale"])
    distance = compare_repeat_runs(reference_series, perturbed_series, norm_type)
    if not math.isfinite(distance) or distance > 10.0 * scale:
        label = "divergent"
    elif distance <= scale:
        label = "attractor"
    else:
        label = "neutral"
    return {
        "terminal_distance": distance,
        "classification": label,
        "passed": label != "divergent",
    }


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
