"""Operator-composition checks for SORT Version 7 Phase 3."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

from phase_3_global_projector.src.global_projector import apply_global_projector


Matrix = Sequence[Sequence[float]]


def apply_operator(operator_matrix: Matrix, state: Sequence[float]) -> list[float]:
    if len(operator_matrix) != len(state):
        raise ValueError("Operator and state dimensions are incompatible.")
    return [
        sum(float(operator_matrix[row][column]) * float(state[column]) for column in range(len(state)))
        for row in range(len(operator_matrix))
    ]


def compose(A: Matrix, B: Matrix, state: Sequence[float]) -> list[float]:
    return apply_operator(A, apply_operator(B, state))


def composition_residual(
    projector: Matrix,
    A: Matrix,
    B: Matrix,
    state: Sequence[float],
    norm_type: str,
) -> float:
    left = apply_global_projector(compose(A, B, state), projector)
    right = apply_global_projector(compose(B, A, state), projector)
    return _vector_norm([left[index] - right[index] for index in range(len(left))], norm_type)


def evaluate_operator_pair(
    projector: Matrix,
    A: Matrix,
    B: Matrix,
    states: list[Sequence[float]],
    norm_type: str,
) -> dict[str, Any]:
    residuals = [composition_residual(projector, A, B, state, norm_type) for state in states]
    return {
        "state_count": len(states),
        "residuals": residuals,
        "maximum_residual": max(residuals) if residuals else 0.0,
        "mean_residual": statistics.fmean(residuals) if residuals else 0.0,
        "median_residual": statistics.median(residuals) if residuals else 0.0,
    }


def _vector_norm(vector: Sequence[float], norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm_key == "max":
        return max((abs(float(value)) for value in vector), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")
