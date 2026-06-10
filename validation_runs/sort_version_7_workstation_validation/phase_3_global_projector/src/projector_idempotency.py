"""Global projector idempotency metrics for SORT Version 7 Phase 3."""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from phase_3_global_projector.src.global_projector import (
    apply_global_projector,
    apply_global_projector_twice,
)


Matrix = Sequence[Sequence[float]]


def matrix_multiply(A: Matrix, B: Matrix) -> list[list[float]]:
    if not A or not B or len(A[0]) != len(B):
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    return [
        [
            sum(float(A[row][inner]) * float(B[inner][column]) for inner in range(len(B)))
            for column in range(len(B[0]))
        ]
        for row in range(len(A))
    ]


def matrix_subtract(A: Matrix, B: Matrix) -> list[list[float]]:
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same shape.")
    return [
        [float(A[row][column]) - float(B[row][column]) for column in range(len(A[0]))]
        for row in range(len(A))
    ]


def projector_idempotency_residual(projector: Matrix, norm_type: str) -> float:
    return _matrix_norm(matrix_subtract(matrix_multiply(projector, projector), projector), norm_type)


def statewise_projector_residual(
    state: Sequence[float],
    projector: Matrix,
    norm_type: str,
) -> float:
    projected_once = apply_global_projector(state, projector)
    projected_twice = apply_global_projector_twice(state, projector)
    residual = [
        projected_twice[index] - projected_once[index]
        for index in range(len(projected_once))
    ]
    return _vector_norm(residual, norm_type)


def summarize_projector_residuals(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "maximum_residual": 0.0,
            "mean_residual": 0.0,
            "median_residual": 0.0,
            "standard_deviation": 0.0,
        }
    return {
        "count": len(values),
        "maximum_residual": max(values),
        "mean_residual": statistics.fmean(values),
        "median_residual": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
    }


def _matrix_norm(matrix: Matrix, norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key == "frobenius":
        return math.sqrt(sum(float(value) ** 2 for row in matrix for value in row))
    if norm_key == "max":
        return max((abs(float(value)) for row in matrix for value in row), default=0.0)
    if norm_key == "l2":
        return _spectral_norm_power_iteration(matrix)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def _vector_norm(vector: Sequence[float], norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm_key == "max":
        return max((abs(float(value)) for value in vector), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def _spectral_norm_power_iteration(matrix: Matrix) -> float:
    rows = len(matrix)
    columns = len(matrix[0])
    vector = [1.0 / math.sqrt(columns)] * columns
    for _ in range(100):
        av = [
            sum(float(matrix[row][column]) * vector[column] for column in range(columns))
            for row in range(rows)
        ]
        at_av = [
            sum(float(matrix[row][column]) * av[row] for row in range(rows))
            for column in range(columns)
        ]
        norm = math.sqrt(sum(value * value for value in at_av))
        if norm == 0.0:
            return 0.0
        vector = [value / norm for value in at_av]
    av = [
        sum(float(matrix[row][column]) * vector[column] for column in range(columns))
        for row in range(rows)
    ]
    return math.sqrt(sum(value * value for value in av))
