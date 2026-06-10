"""Fixed-point iteration utilities for SORT Version 7 Phase 4."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence


SIGMA_0 = 0.00190643


def generate_initial_states(config: dict[str, Any], seed_config: dict[str, Any]) -> list[list[float]]:
    state_count = int(config["synthetic_states"]["initial_state_count"])
    state_dimension = int(config["synthetic_states"]["state_dimension"])
    rng = random.Random(int(seed_config["global_seed"]))
    return [
        [rng.uniform(-1.0, 1.0) for _ in range(state_dimension)]
        for _ in range(state_count)
    ]


def kernel_profile_from_phase2_config(
    phase2_setup: dict[str, Any],
    k_value: float,
) -> list[float]:
    xi_values = [float(value) for value in phase2_setup["xi_grid"]["values"]]
    sigma_0 = float(phase2_setup.get("sigma_0", SIGMA_0))
    return [
        math.exp(-0.5 * (sigma_0 * float(k_value) * xi_value) ** 2)
        for xi_value in xi_values
    ]


def apply_pi_kappa(state: Sequence[float], kernel_values: Sequence[float]) -> list[float]:
    if len(state) != len(kernel_values):
        raise ValueError("State and kernel profile dimensions must match.")
    kernel_norm = _vector_norm(kernel_values, "frobenius")
    if kernel_norm == 0.0:
        raise ValueError("Cannot apply projection with a zero kernel profile.")
    u = [float(value) / kernel_norm for value in kernel_values]
    coefficient = sum(u[index] * float(state[index]) for index in range(len(u)))
    return [u_value * coefficient for u_value in u]


def apply_H(state: Sequence[float], global_projector: Sequence[Sequence[float]]) -> list[float]:
    if len(state) != len(global_projector):
        raise ValueError("State dimension must match global projector dimension.")
    return [
        sum(float(global_projector[row][column]) * float(state[column]) for column in range(len(state)))
        for row in range(len(global_projector))
    ]


def iterate_projection(
    initial_state: Sequence[float],
    projection_fn: Callable[[Sequence[float]], list[float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    norm_type = config["norm_type"]
    epsilon = float(config["tolerances"]["epsilon_convergence"])
    max_iterations = int(config["iteration"]["max_iterations"])
    min_iterations = int(config["iteration"]["min_iterations"])
    stop_when_converged = bool(config["iteration"]["stop_when_converged"])

    current = [float(value) for value in initial_state]
    series = [
        {
            "iteration": 0,
            "state": current,
            "residual": 0.0,
            "norm_value": _vector_norm(current, norm_type),
            "stop_reason": "initial_state",
        }
    ]

    for iteration in range(1, max_iterations + 1):
        next_state = projection_fn(current)
        residual = _vector_norm(
            [next_state[index] - current[index] for index in range(len(current))],
            norm_type,
        )
        stop_reason = "max_iterations"
        if iteration >= min_iterations and residual < epsilon and stop_when_converged:
            stop_reason = "converged_after_min_iterations"
        series.append(
            {
                "iteration": iteration,
                "state": next_state,
                "residual": residual,
                "norm_value": _vector_norm(next_state, norm_type),
                "stop_reason": stop_reason,
            }
        )
        current = next_state
        if stop_reason == "converged_after_min_iterations":
            break

    return series


def run_fixed_point_batch(
    states: list[Sequence[float]],
    projection_fn: Callable[[Sequence[float]], list[float]],
    config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    return [iterate_projection(state, projection_fn, config) for state in states]


def _vector_norm(values: Sequence[float], norm_type: str) -> float:
    norm_key = norm_type.lower()
    if norm_key in {"frobenius", "l2"}:
        return math.sqrt(sum(float(value) ** 2 for value in values))
    if norm_key == "max":
        return max((abs(float(value)) for value in values), default=0.0)
    raise ValueError(f"Unsupported norm type: {norm_type}")
