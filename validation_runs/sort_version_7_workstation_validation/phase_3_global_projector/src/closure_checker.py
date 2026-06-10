"""Closure checks for SORT Version 7 Phase 3 projected synthetic states."""

from __future__ import annotations

import math
from typing import Any, Sequence


def is_finite_state(state: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in state)


def state_norm(state: Sequence[float], norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in state))
    if norm_key == "max":
        return max((abs(float(value)) for value in state), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def check_admissible_state(
    state: Sequence[float],
    admissible_space: dict[str, Any],
    norm_type: str,
) -> dict[str, Any]:
    finite = is_finite_state(state)
    norm_value = state_norm(state, norm_type) if finite else float("inf")
    max_norm = float(admissible_space["max_norm"])
    within_bound = norm_value <= max_norm
    margin = max_norm - norm_value
    borderline = finite and within_bound and margin <= max_norm * 1e-9
    return {
        "finite": finite,
        "norm": norm_value,
        "max_norm": max_norm,
        "within_bound": within_bound,
        "borderline": borderline,
        "passed": finite and within_bound,
    }


def closure_violation_rate(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    violations = sum(1 for result in results if not result.get("passed"))
    return violations / len(results)
