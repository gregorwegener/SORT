"""Projection metric utilities for SORT Version 7 Phase 2."""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

from phase_2_projection_kernel.src.kernel_definition import kernel_value


def vector_norm(vector: Sequence[float], norm_type: str = "frobenius") -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm_key == "max":
        return max((abs(float(value)) for value in vector), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def projection_residual(
    projected_once: Sequence[float],
    projected_twice: Sequence[float],
    norm_type: str,
) -> float:
    if len(projected_once) != len(projected_twice):
        raise ValueError("Projected vectors must have the same dimension.")
    difference = [
        float(projected_twice[index]) - float(projected_once[index])
        for index in range(len(projected_once))
    ]
    return vector_norm(difference, norm_type)


def summarize_residuals(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "maximum_residual": 0.0,
            "mean_residual": 0.0,
            "median_residual": 0.0,
        }
    return {
        "count": len(values),
        "maximum_residual": max(values),
        "mean_residual": statistics.fmean(values),
        "median_residual": statistics.median(values),
    }


def kernel_profile_rows(
    k_values: Sequence[float],
    xi_values: Sequence[float],
    sigma_0: float,
) -> list[dict[str, Any]]:
    rows = []
    for k_value in k_values:
        for xi_value in xi_values:
            rows.append(
                {
                    "k_value": float(k_value),
                    "xi_value": float(xi_value),
                    "kernel_value": kernel_value(float(xi_value), sigma_0, float(k_value)),
                }
            )
    return rows
