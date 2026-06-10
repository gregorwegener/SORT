"""Run Phase 1 Jacobi consistency validation."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from phase_1_operator_integrity.src.commutators import jacobi_residual
from phase_1_operator_integrity.src.norms import matrix_norm
from phase_1_operator_integrity.src.operator_registry import get_operator, load_registry
from phase_1_operator_integrity.src.report_writer import (
    build_or_update_report,
    replace_rows,
    read_residual_csv,
    summarize_residuals,
    write_json_report,
    write_residual_csv,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup = json.loads((phase_root / "config" / "phase_1_setup.json").read_text(encoding="utf-8"))
    registry = load_registry(phase_root / setup["inputs"]["operator_registry"])
    seed_config = json.loads((phase_root / setup["jacobi"]["seed_reference"]).resolve().read_text(encoding="utf-8"))

    tolerance = float(setup["tolerances"]["epsilon_jacobi"])
    norm_type = setup["norm_type"]
    mode = setup["jacobi"]["mode"]
    sample_count = int(setup["jacobi"]["sample_count"])
    triples = _triples(mode, sample_count, int(seed_config["global_seed"]))

    residuals: list[float] = []
    rows = []
    failed_triples = []

    for i, j, k in triples:
        A = get_operator(registry, i)["matrix"]
        B = get_operator(registry, j)["matrix"]
        C = get_operator(registry, k)["matrix"]
        residual = matrix_norm(jacobi_residual(A, B, C), norm_type)
        passed = residual < tolerance
        residuals.append(residual)
        if not passed:
            failed_triples.append([i, j, k])
        rows.append(
            {
                "test_type": "jacobi",
                "operator_id_i": i,
                "operator_id_j": j,
                "operator_id_k": k,
                "residual": f"{residual:.17g}",
                "norm_type": norm_type,
                "passed": str(passed).lower(),
            }
        )

    summary = summarize_residuals(residuals)
    summary.update(
        {
            "mode": mode,
            "sample_count": len(triples) if mode == "sampled" else None,
            "tolerance": tolerance,
            "seed_reference": setup["jacobi"]["seed_reference"],
            "global_seed": seed_config["global_seed"],
            "passed_count": len(residuals) - len(failed_triples),
            "failed_count": len(failed_triples),
            "failed_triples": failed_triples,
            "passed": not failed_triples,
        }
    )

    report = build_or_update_report(
        phase_root=phase_root,
        setup=setup,
        registry=registry,
        section_name="jacobi_summary",
        section_summary=summary,
    )
    report_path = phase_root / setup["outputs"]["operator_integrity_report"]
    csv_path = phase_root / setup["outputs"]["operator_residuals"]
    write_json_report(report_path, report)
    write_residual_csv(csv_path, replace_rows(read_residual_csv(csv_path), "jacobi", rows))

    print("Phase 1 Jacobi validation passed" if summary["passed"] else "Phase 1 Jacobi validation failed")
    return 0 if summary["passed"] else 1


def _triples(mode: str, sample_count: int, seed: int) -> list[tuple[int, int, int]]:
    operator_ids = range(1, 23)
    if mode == "full":
        return list(itertools.product(operator_ids, repeat=3))
    if mode == "sampled":
        rng = random.Random(seed)
        return [
            (
                rng.randint(1, 22),
                rng.randint(1, 22),
                rng.randint(1, 22),
            )
            for _ in range(sample_count)
        ]
    raise ValueError("Jacobi mode must be either 'full' or 'sampled'.")


if __name__ == "__main__":
    raise SystemExit(main())
