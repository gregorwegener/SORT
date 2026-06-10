"""Run Phase 2 kernel normalization validation."""

from __future__ import annotations

import json
from pathlib import Path

from phase_2_projection_kernel.src.kernel_definition import (
    SIGMA_0,
    kernel_profile,
    load_kernel_definition,
)
from phase_2_projection_kernel.src.kernel_normalization import (
    kernel_norm,
    kernel_norm_residual,
    normalize_kernel,
)
from phase_2_projection_kernel.src.projection_metrics import kernel_profile_rows
from phase_2_projection_kernel.src.report_writer import (
    build_or_update_report,
    write_csv,
    write_json_report,
)


def main() -> int:
    phase_root = Path(__file__).resolve().parents[1]
    setup = json.loads((phase_root / "config" / "phase_2_setup.json").read_text(encoding="utf-8"))
    kernel_definition = load_kernel_definition(phase_root / setup["inputs"]["kernel_definitions"])

    sigma_0 = float(kernel_definition["kernel"]["scale_parameter"]["value"])
    if sigma_0 != SIGMA_0:
        raise ValueError("Phase 2 sigma_0 does not match the declared structural reference value.")

    xi_values = [float(value) for value in setup["xi_grid"]["values"]]
    k_values = [float(value) for value in setup["k_grid"]["values"]]
    target = float(setup["integration"]["normalization_target"])
    tolerance = float(setup["tolerances"]["epsilon_norm"])

    scale_results = []
    residuals = []
    failed_k_values = []

    for k_value in k_values:
        raw_values = kernel_profile(xi_values, sigma_0, k_value)
        raw_norm = kernel_norm(raw_values)
        normalized_values = normalize_kernel(raw_values)
        normalized_norm = kernel_norm(normalized_values)
        residual = kernel_norm_residual(normalized_values, target)
        passed = residual < tolerance
        residuals.append(residual)
        if not passed:
            failed_k_values.append(k_value)
        scale_results.append(
            {
                "k_value": k_value,
                "raw_norm": raw_norm,
                "normalized_norm": normalized_norm,
                "residual": residual,
                "tolerance": tolerance,
                "passed": passed,
            }
        )

    summary = {
        "integration_method": setup["integration"]["method"],
        "domain_type": setup["integration"]["domain_type"],
        "normalization_target": target,
        "epsilon_norm": tolerance,
        "scale_results": scale_results,
        "maximum_residual": max(residuals),
        "failed_k_values": failed_k_values,
        "passed": not failed_k_values,
    }

    report = build_or_update_report(
        phase_root=phase_root,
        setup=setup,
        section_name="kernel_normalization_summary",
        section_summary=summary,
    )
    write_json_report(phase_root / setup["outputs"]["kernel_norm_report"], report)
    write_csv(
        phase_root / setup["outputs"]["kernel_profiles"],
        kernel_profile_rows(k_values, xi_values, sigma_0),
        ["k_value", "xi_value", "kernel_value"],
    )

    print("Phase 2 kernel normalization validation passed" if summary["passed"] else "Phase 2 kernel normalization validation failed")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
