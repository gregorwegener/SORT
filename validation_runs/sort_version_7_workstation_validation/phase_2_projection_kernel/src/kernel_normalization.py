"""Kernel normalization utilities for SORT Version 7 Phase 2."""

from __future__ import annotations

import math
from typing import Sequence


def vector_l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def normalize_kernel(values: Sequence[float]) -> list[float]:
    norm = vector_l2_norm(values)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero kernel vector.")
    return [float(value) / norm for value in values]


def kernel_norm(values: Sequence[float]) -> float:
    return vector_l2_norm(values)


def kernel_norm_residual(values: Sequence[float], target: float) -> float:
    return abs(kernel_norm(values) - float(target))
