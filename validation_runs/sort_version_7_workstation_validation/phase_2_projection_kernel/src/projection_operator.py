"""Rank-one kernel projection interface for SORT Version 7 Phase 2."""

from __future__ import annotations

from typing import Sequence

from phase_2_projection_kernel.src.kernel_normalization import normalize_kernel
from phase_2_projection_kernel.src.projection_metrics import projection_residual


def kernel_projection_vector(kernel_values: Sequence[float]) -> list[float]:
    return normalize_kernel(kernel_values)


def apply_projection(state: Sequence[float], kernel_values: Sequence[float]) -> list[float]:
    if len(state) != len(kernel_values):
        raise ValueError("State and kernel vectors must have the same dimension.")
    u = kernel_projection_vector(kernel_values)
    coefficient = sum(u[index] * float(state[index]) for index in range(len(u)))
    return [u_value * coefficient for u_value in u]


def apply_projection_twice(state: Sequence[float], kernel_values: Sequence[float]) -> list[float]:
    return apply_projection(apply_projection(state, kernel_values), kernel_values)


def projection_idempotency_residual(
    state: Sequence[float],
    kernel_values: Sequence[float],
    norm_type: str,
) -> float:
    projected_once = apply_projection(state, kernel_values)
    projected_twice = apply_projection(projected_once, kernel_values)
    return projection_residual(projected_once, projected_twice, norm_type)
