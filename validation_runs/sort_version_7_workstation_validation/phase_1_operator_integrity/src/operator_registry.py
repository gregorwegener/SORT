"""Operator registry loading and validation for SORT Version 7 Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SORT_VERSION = "Version 7"
PHASE = "Phase 1 — Operator Integrity"
OPERATOR_COUNT = 22
MATRIX_DIMENSION = 22


class RegistryValidationError(ValueError):
    """Raised when the Phase 1 operator registry is malformed."""


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryValidationError(f"Registry file not found: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(f"Registry JSON is invalid: {registry_path}") from exc
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    if registry.get("sort_version") != SORT_VERSION:
        raise RegistryValidationError("Registry sort_version must be exactly 'Version 7'.")
    if registry.get("phase") != PHASE:
        raise RegistryValidationError(f"Registry phase must be exactly '{PHASE}'.")
    if registry.get("operator_count") != OPERATOR_COUNT:
        raise RegistryValidationError("Registry operator_count must be exactly 22.")
    if registry.get("matrix_dimension") != MATRIX_DIMENSION:
        raise RegistryValidationError("Registry matrix_dimension must be exactly 22.")
    if not registry.get("non_claims"):
        raise RegistryValidationError("Registry non_claims must be present.")

    operators = registry.get("operators")
    if not isinstance(operators, list):
        raise RegistryValidationError("Registry operators must be a list.")
    if len(operators) != OPERATOR_COUNT:
        raise RegistryValidationError("Registry must contain exactly 22 operators.")

    expected_ids = list(range(1, OPERATOR_COUNT + 1))
    actual_ids = [operator.get("operator_id") for operator in operators]
    if actual_ids != expected_ids:
        raise RegistryValidationError("Operator IDs must be ordered integers 1 through 22.")

    for operator in operators:
        operator_id = operator["operator_id"]
        expected_symbol = f"O_{operator_id}"
        expected_name = f"SORT Operator {operator_id:02d}"

        if operator.get("symbol") != expected_symbol:
            raise RegistryValidationError(
                f"Operator {operator_id} symbol must be {expected_symbol}."
            )
        if operator.get("name") != expected_name:
            raise RegistryValidationError(
                f"Operator {operator_id} name must be {expected_name}."
            )
        if operator.get("representation_type") != "matrix":
            raise RegistryValidationError(
                f"Operator {operator_id} representation_type must be 'matrix'."
            )
        if "c_weight" not in operator or not isinstance(operator["c_weight"], (int, float)):
            raise RegistryValidationError(
                f"Operator {operator_id} must declare numeric c_weight."
            )

        expected_weight = 1.0 if operator_id <= 11 else -1.0
        if float(operator["c_weight"]) != expected_weight:
            raise RegistryValidationError(
                f"Operator {operator_id} c_weight must be {expected_weight}."
            )

        matrix = operator.get("matrix")
        _validate_square_matrix(matrix, operator_id)


def _validate_square_matrix(matrix: Any, operator_id: int) -> None:
    if not isinstance(matrix, list):
        raise RegistryValidationError(f"Operator {operator_id} matrix must be a list.")
    if len(matrix) != MATRIX_DIMENSION:
        raise RegistryValidationError(
            f"Operator {operator_id} matrix must have 22 rows."
        )
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list):
            raise RegistryValidationError(
                f"Operator {operator_id} matrix row {row_index} must be a list."
            )
        if len(row) != MATRIX_DIMENSION:
            raise RegistryValidationError(
                f"Operator {operator_id} matrix row {row_index} must have 22 columns."
            )
        for column_index, value in enumerate(row):
            if not isinstance(value, (int, float)):
                raise RegistryValidationError(
                    "Operator "
                    f"{operator_id} matrix entry ({row_index}, {column_index}) "
                    "must be numeric."
                )


def get_operator(registry: dict[str, Any], operator_id: int) -> dict[str, Any]:
    if not isinstance(operator_id, int):
        raise RegistryValidationError("operator_id must be an integer.")
    for operator in registry["operators"]:
        if operator["operator_id"] == operator_id:
            return operator
    raise RegistryValidationError(f"Operator {operator_id} not found.")


def get_all_operators(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return list(registry["operators"])


def get_weights(registry: dict[str, Any]) -> list[float]:
    return [float(operator["c_weight"]) for operator in registry["operators"]]
