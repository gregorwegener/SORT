"""Matrix norm utilities for SORT Version 7 Phase 1."""

from __future__ import annotations

import math
from typing import Sequence


Matrix = Sequence[Sequence[float]]


def matrix_norm(matrix: Matrix, norm_type: str = "frobenius") -> float:
    norm_key = norm_type.lower()
    _validate_rectangular(matrix)
    if norm_key == "frobenius":
        return math.sqrt(sum(float(value) ** 2 for row in matrix for value in row))
    if norm_key == "max":
        return max((abs(float(value)) for row in matrix for value in row), default=0.0)
    if norm_key == "l2":
        return _spectral_norm(matrix)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def residual_norm(lhs: Matrix, rhs: Matrix, norm_type: str = "frobenius") -> float:
    _validate_same_shape(lhs, rhs)
    residual = [
        [float(lhs[row][column]) - float(rhs[row][column]) for column in range(len(lhs[0]))]
        for row in range(len(lhs))
    ]
    return matrix_norm(residual, norm_type)


def _validate_rectangular(matrix: Matrix) -> None:
    if not matrix:
        raise ValueError("Matrix must not be empty.")
    width = len(matrix[0])
    if width == 0:
        raise ValueError("Matrix rows must not be empty.")
    for row in matrix:
        if len(row) != width:
            raise ValueError("Matrix rows must have consistent length.")


def _validate_same_shape(lhs: Matrix, rhs: Matrix) -> None:
    _validate_rectangular(lhs)
    _validate_rectangular(rhs)
    if len(lhs) != len(rhs) or len(lhs[0]) != len(rhs[0]):
        raise ValueError("Matrices must have the same shape.")


def _spectral_norm(matrix: Matrix) -> float:
    try:
        import numpy as np  # type: ignore[import-not-found]

        return float(np.linalg.norm(np.array(matrix, dtype=float), 2))
    except Exception:
        return _spectral_norm_power_iteration(matrix)


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
