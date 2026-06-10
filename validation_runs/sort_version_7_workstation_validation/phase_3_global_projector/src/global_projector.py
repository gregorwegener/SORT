"""Global projector construction for SORT Version 7 Phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


MATRIX_DIMENSION = 22


class ProjectorError(ValueError):
    """Raised when the declared Phase 3 projector cannot be constructed."""


def load_projector_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProjectorError(f"Projector config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectorError(f"Projector config is invalid JSON: {config_path}") from exc
    if config.get("sort_version") != "Version 7":
        raise ProjectorError("Projector config sort_version must be exactly 'Version 7'.")
    if config.get("phase") != "Phase 3 — Global Projector":
        raise ProjectorError("Projector config phase must be Phase 3 — Global Projector.")
    if config.get("global_projector", {}).get("construction_rule") != "sum_of_coordinate_projection_operators":
        raise ProjectorError("Unsupported global projector construction rule.")
    if config.get("operator_composition", {}).get("assume_global_commutativity") is not False:
        raise ProjectorError("Phase 3 must explicitly not assume global commutativity.")
    return config


def construct_global_projector(
    operator_registry: dict[str, Any],
    config: dict[str, Any],
) -> list[list[float]]:
    if operator_registry.get("operator_count") != MATRIX_DIMENSION:
        raise ProjectorError("Operator registry must declare exactly 22 operators.")
    operators = operator_registry.get("operators")
    if not isinstance(operators, list) or len(operators) != MATRIX_DIMENSION:
        raise ProjectorError("Operator registry must contain 22 operator entries.")
    if config.get("global_projector", {}).get("construction_rule") != "sum_of_coordinate_projection_operators":
        raise ProjectorError("Global projector construction rule is not declared.")

    projector = [[0.0 for _ in range(MATRIX_DIMENSION)] for _ in range(MATRIX_DIMENSION)]
    for expected_id, operator in enumerate(operators, start=1):
        if operator.get("operator_id") != expected_id:
            raise ProjectorError("Operator ordering must preserve IDs 1 through 22.")
        matrix = operator.get("matrix")
        _validate_square_matrix(matrix, MATRIX_DIMENSION, expected_id)
        for row in range(MATRIX_DIMENSION):
            for column in range(MATRIX_DIMENSION):
                projector[row][column] += float(matrix[row][column])

    _validate_square_matrix(projector, MATRIX_DIMENSION, 0)
    return projector


def apply_global_projector(
    state: Sequence[float],
    projector: Sequence[Sequence[float]],
) -> list[float]:
    if len(state) != len(projector):
        raise ProjectorError("State dimension must match projector dimension.")
    return [
        sum(float(projector[row][column]) * float(state[column]) for column in range(len(state)))
        for row in range(len(projector))
    ]


def apply_global_projector_twice(
    state: Sequence[float],
    projector: Sequence[Sequence[float]],
) -> list[float]:
    return apply_global_projector(apply_global_projector(state, projector), projector)


def _validate_square_matrix(matrix: Any, dimension: int, operator_id: int) -> None:
    label = "projector" if operator_id == 0 else f"operator {operator_id}"
    if not isinstance(matrix, list) or len(matrix) != dimension:
        raise ProjectorError(f"{label} matrix must have {dimension} rows.")
    for row in matrix:
        if not isinstance(row, list) or len(row) != dimension:
            raise ProjectorError(f"{label} matrix must be {dimension} by {dimension}.")
        for value in row:
            if not isinstance(value, (int, float)):
                raise ProjectorError(f"{label} matrix entries must be numeric.")
