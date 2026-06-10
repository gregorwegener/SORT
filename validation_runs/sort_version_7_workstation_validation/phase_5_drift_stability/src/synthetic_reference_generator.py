"""Synthetic reference generation for SORT Version 7 Phase 5."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Sequence


SIGMA_0 = 0.00190643


def generate_kernel_vector(config: dict[str, Any]) -> list[float]:
    dimension = int(config["synthetic_references"]["state_dimension"])
    k_value = float(config["effective_drift_projector"]["k_value"])
    raw = [
        math.exp(-0.5 * (SIGMA_0 * k_value * float(index)) ** 2)
        for index in range(1, dimension + 1)
    ]
    norm = _vector_norm(raw)
    if norm == 0.0:
        raise ValueError("Cannot generate kernel vector from a zero kernel profile.")
    return [value / norm for value in raw]


def generate_projector_invariant_states(
    config: dict[str, Any],
    projector: Any,
) -> list[dict[str, Any]]:
    unit_vector = generate_kernel_vector(config)
    count = int(config["synthetic_references"]["reference_state_count_per_class"])
    states = []
    for index in range(count):
        scalar = 0.75 + (index + 1) / (count + 1)
        states.append(
            {
                "state_id": index + 1,
                "reference_class": "projector_invariant",
                "state_dimension": len(unit_vector),
                "construction_method": "scalar_multiple_of_kernel_projection_vector",
                "amplitude": 0.0,
                "state": [scalar * value for value in unit_vector],
            }
        )
    return states


def generate_slightly_violated_states(
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    return _generate_violated_states(
        config=config,
        seed=seed + 101,
        start_id=1 + int(config["synthetic_references"]["reference_state_count_per_class"]),
        reference_class="slightly_violated",
        amplitude=5.0e-7,
        construction_method="kernel_vector_plus_deterministic_orthogonal_perturbation_amplitude_5e-7",
    )


def generate_strongly_violated_states(
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    return _generate_violated_states(
        config=config,
        seed=seed + 202,
        start_id=1 + 2 * int(config["synthetic_references"]["reference_state_count_per_class"]),
        reference_class="strongly_violated",
        amplitude=1.0e-3,
        construction_method="kernel_vector_plus_deterministic_orthogonal_perturbation_amplitude_1e-3",
    )


def orthogonalize_against(
    vector: Sequence[float],
    unit_vector: Sequence[float],
) -> list[float]:
    projection = sum(float(vector[index]) * float(unit_vector[index]) for index in range(len(unit_vector)))
    orthogonal = [
        float(vector[index]) - projection * float(unit_vector[index])
        for index in range(len(unit_vector))
    ]
    norm = _vector_norm(orthogonal)
    if norm == 0.0:
        raise ValueError("Generated perturbation is not usable after orthogonalization.")
    return [value / norm for value in orthogonal]


def write_reference_catalog(path: str | Path, references: list[dict[str, Any]]) -> None:
    catalog_path = Path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(references, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_reference_catalog(
    *,
    run_id: str,
    config: dict[str, Any],
    seed: int,
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    class_counts = {
        reference_class: sum(1 for item in references if item["reference_class"] == reference_class)
        for reference_class in config["synthetic_references"]["classes"]
    }
    return {
        "run_id": run_id,
        "sort_version": "Version 7",
        "phase": "Phase 5 — Drift and Stability",
        "repository": {
            "name": "gregorwegener/SORT",
            "canonical_url": "https://github.com/gregorwegener/SORT",
            "validation_root": "validation_runs/sort_version_7_workstation_validation",
            "phase_path": "validation_runs/sort_version_7_workstation_validation/phase_5_drift_stability",
        },
        "synthetic_reference_classes": config["synthetic_references"]["classes"],
        "state_count_per_class": class_counts,
        "total_state_count": len(references),
        "state_dimension": config["synthetic_references"]["state_dimension"],
        "seed_reference": config["synthetic_references"]["seed_reference"],
        "global_seed": seed,
        "construction_methods": sorted({item["construction_method"] for item in references}),
        "references": references,
        "non_claims": config["non_claims"],
    }


def generate_all_references(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    references = []
    references.extend(generate_projector_invariant_states(config, None))
    references.extend(generate_slightly_violated_states(config, seed))
    references.extend(generate_strongly_violated_states(config, seed))
    return references


def _generate_violated_states(
    *,
    config: dict[str, Any],
    seed: int,
    start_id: int,
    reference_class: str,
    amplitude: float,
    construction_method: str,
) -> list[dict[str, Any]]:
    unit_vector = generate_kernel_vector(config)
    count = int(config["synthetic_references"]["reference_state_count_per_class"])
    rng = random.Random(seed)
    states = []
    for index in range(count):
        scalar = 0.75 + (index + 1) / (count + 1)
        raw = [rng.uniform(-1.0, 1.0) for _ in unit_vector]
        orthogonal = orthogonalize_against(raw, unit_vector)
        states.append(
            {
                "state_id": start_id + index,
                "reference_class": reference_class,
                "state_dimension": len(unit_vector),
                "construction_method": construction_method,
                "amplitude": amplitude,
                "state": [
                    scalar * unit_vector[column] + amplitude * orthogonal[column]
                    for column in range(len(unit_vector))
                ],
            }
        )
    return states


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))
