"""Algebraic utilities for SORT Version 7 Phase 1."""

from __future__ import annotations

from typing import Sequence


Matrix = Sequence[Sequence[float]]


def commutator(A: Matrix, B: Matrix) -> list[list[float]]:
    return _subtract(_matmul(A, B), _matmul(B, A))


def jacobi_residual(A: Matrix, B: Matrix, C: Matrix) -> list[list[float]]:
    first = commutator(A, commutator(B, C))
    second = commutator(B, commutator(C, A))
    third = commutator(C, commutator(A, B))
    return _add(_add(first, second), third)


def _matmul(A: Matrix, B: Matrix) -> list[list[float]]:
    if not A or not B:
        raise ValueError("Matrices must not be empty.")
    rows_a = len(A)
    cols_a = len(A[0])
    rows_b = len(B)
    cols_b = len(B[0])
    if cols_a != rows_b:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    return [
        [
            sum(float(A[row][inner]) * float(B[inner][column]) for inner in range(cols_a))
            for column in range(cols_b)
        ]
        for row in range(rows_a)
    ]


def _subtract(A: Matrix, B: Matrix) -> list[list[float]]:
    _validate_same_shape(A, B)
    return [
        [float(A[row][column]) - float(B[row][column]) for column in range(len(A[0]))]
        for row in range(len(A))
    ]


def _add(A: Matrix, B: Matrix) -> list[list[float]]:
    _validate_same_shape(A, B)
    return [
        [float(A[row][column]) + float(B[row][column]) for column in range(len(A[0]))]
        for row in range(len(A))
    ]


def _validate_same_shape(A: Matrix, B: Matrix) -> None:
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same shape.")
