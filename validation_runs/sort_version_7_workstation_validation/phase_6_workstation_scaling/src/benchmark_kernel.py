"""Fixed Phase 6 workstation scaling workload."""

from __future__ import annotations

import math
from typing import Any, Callable

try:
    import numpy as np
except Exception:  # pragma: no cover - exercised only when numpy is unavailable.
    np = None


def generate_grid_state(grid_size: int, seed: int):
    element_count = int(grid_size) ** 3
    if np is not None:
        rng = np.random.default_rng(int(seed) + int(grid_size))
        return rng.standard_normal(element_count, dtype=np.float64)
    if element_count > 250000:
        raise RuntimeError("NumPy unavailable and grid is too large for safe list fallback.")
    import random

    rng = random.Random(int(seed) + int(grid_size))
    return [rng.uniform(-1.0, 1.0) for _ in range(element_count)]


def run_projection_iterations(state, projection_fn: Callable[[Any], Any], iterations: int):
    current = state
    for _ in range(int(iterations)):
        current = projection_fn(current)
    return current


def run_global_projector_iterations(state, projector_fn: Callable[[Any], Any], iterations: int):
    current = state
    for _ in range(int(iterations)):
        current = projector_fn(current)
    return current


def compute_final_residual(initial_state, final_state, norm_type: str) -> float:
    if np is not None and hasattr(initial_state, "shape"):
        difference = final_state - initial_state
        if norm_type.upper() in {"L2", "FROBENIUS"}:
            return float(np.linalg.norm(difference))
        if norm_type.lower() == "max":
            return float(np.max(np.abs(difference)))
    difference = [float(final_state[index]) - float(initial_state[index]) for index in range(len(initial_state))]
    if norm_type.upper() in {"L2", "FROBENIUS"}:
        return math.sqrt(sum(value * value for value in difference))
    if norm_type.lower() == "max":
        return max((abs(value) for value in difference), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def run_benchmark_kernel(
    grid_size: int,
    projection_fn: Callable[[Any], Any],
    projector_fn: Callable[[Any], Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if config["benchmark_kernel"]["early_stopping"]:
        raise ValueError("Phase 6 benchmark kernel must not use early stopping.")
    if config["benchmark_kernel"]["adaptive_parameters"]:
        raise ValueError("Phase 6 benchmark kernel must not use adaptive parameters.")
    initial_state = generate_grid_state(grid_size, seed)
    after_projection = run_projection_iterations(
        initial_state,
        projection_fn,
        int(config["benchmark_kernel"]["projection_iterations"]),
    )
    final_state = run_global_projector_iterations(
        after_projection,
        projector_fn,
        int(config["benchmark_kernel"]["global_projector_iterations"]),
    )
    final_residual = compute_final_residual(
        initial_state,
        final_state,
        config["runtime"]["norm_type"],
    )
    return {
        "grid_size": int(grid_size),
        "element_count": int(grid_size) ** 3,
        "final_residual": final_residual,
    }


def rank_one_projection(state):
    if np is not None and hasattr(state, "shape"):
        mean_value = float(np.mean(state))
        return np.full_like(state, mean_value)
    mean_value = sum(float(value) for value in state) / len(state)
    return [mean_value for _ in state]


def identity_projector(state):
    if np is not None and hasattr(state, "copy"):
        return state.copy()
    return list(state)


def estimate_memory_mb(grid_size: int) -> float:
    element_count = int(grid_size) ** 3
    return element_count * 8.0 * 4.0 / (1024.0 * 1024.0)


def numpy_available() -> bool:
    return np is not None
