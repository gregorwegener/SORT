"""Kernel definition utilities for SORT Version 7 Phase 2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


SORT_VERSION = "Version 7"
PHASE = "Phase 2 — Projection Kernel"
SIGMA_0 = 0.00190643


class KernelDefinitionError(ValueError):
    """Raised when the Phase 2 kernel definition is malformed."""


def load_kernel_definition(path: str | Path) -> dict[str, Any]:
    definition_path = Path(path)
    if not definition_path.exists():
        raise KernelDefinitionError(f"Kernel definition file not found: {definition_path}")
    definition = _parse_phase_2_kernel_yaml(definition_path.read_text(encoding="utf-8"))
    validate_kernel_definition(definition)
    return definition


def validate_kernel_definition(definition: dict[str, Any]) -> None:
    if definition.get("sort_version") != SORT_VERSION:
        raise KernelDefinitionError("Kernel definition sort_version must be exactly 'Version 7'.")
    if definition.get("phase") != PHASE:
        raise KernelDefinitionError(f"Kernel definition phase must be exactly '{PHASE}'.")
    kernel = definition.get("kernel")
    if not isinstance(kernel, dict):
        raise KernelDefinitionError("Kernel definition must contain a kernel mapping.")
    if not kernel.get("kernel_name"):
        raise KernelDefinitionError("Kernel name is required.")
    if kernel.get("analytic_form") != "exp(-0.5 * (sigma_0 * xi)^2)":
        raise KernelDefinitionError("Kernel analytic form does not match Phase 2 declaration.")
    scale_parameter = kernel.get("scale_parameter")
    if not isinstance(scale_parameter, dict):
        raise KernelDefinitionError("Kernel scale_parameter is required.")
    if scale_parameter.get("symbol") != "sigma_0":
        raise KernelDefinitionError("Kernel scale parameter symbol must be sigma_0.")
    if abs(float(scale_parameter.get("value")) - SIGMA_0) > 0.0:
        raise KernelDefinitionError("Kernel sigma_0 must be exactly 0.00190643.")
    if not isinstance(kernel.get("normalization_domain"), dict):
        raise KernelDefinitionError("Kernel normalization_domain is required.")
    if kernel.get("projection_role") != "structural projection interface":
        raise KernelDefinitionError("Kernel projection_role must be declared.")
    if not definition.get("non_claims"):
        raise KernelDefinitionError("Kernel non_claims are required.")


def kernel_value(xi: float, sigma_0: float, k_value: float = 1.0) -> float:
    return math.exp(-0.5 * (float(sigma_0) * float(k_value) * float(xi)) ** 2)


def kernel_profile(
    xi_values: list[float],
    sigma_0: float,
    k_value: float = 1.0,
) -> list[float]:
    return [kernel_value(xi, sigma_0, k_value) for xi in xi_values]


def evaluate_over_k_grid(
    xi_values: list[float],
    k_grid: list[float],
    sigma_0: float,
) -> dict[float, list[float]]:
    return {
        float(k_value): kernel_profile(xi_values, sigma_0, float(k_value))
        for k_value in k_grid
    }


def _parse_phase_2_kernel_yaml(text: str) -> dict[str, Any]:
    """Parse the fixed Phase 2 YAML shape without external dependencies."""
    definition: dict[str, Any] = {"kernel": {}, "non_claims": []}
    stack: list[tuple[int, Any]] = [(-1, definition)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise KernelDefinitionError("YAML list item found outside a list.")
            parent.append(_parse_scalar(line[2:]))
            continue

        if ":" not in line:
            raise KernelDefinitionError(f"Cannot parse YAML line: {line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if raw_value == "":
            next_container: Any = [] if key == "non_claims" or key == "bounds" else {}
            if isinstance(parent, dict):
                parent[key] = next_container
            else:
                raise KernelDefinitionError("Nested mapping requires dictionary parent.")
            stack.append((indent, next_container))
        else:
            if not isinstance(parent, dict):
                raise KernelDefinitionError("Scalar mapping requires dictionary parent.")
            parent[key] = _parse_scalar(raw_value)

    return definition


def _parse_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value == "[]":
        return []
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
