# Phase 4 — Fixed-Point Structure

This directory contains the fixed-point and convergence validation artifacts for the SORT Version 7 workstation validation run.

Repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_4_fixed_point/
```

Phase 4 validates iterative projection behavior on deterministic synthetic states.

## Scope

Phase 4 validates:

- iterative projection behavior
- convergence residuals
- fixed-point classification
- norm-invariance tracking
- deterministic repeatability
- perturbation response

Phase 4 does not use empirical data, does not introduce Level-1 dynamics, does not validate a physical model, does not perform ASDV, and does not constitute SWORD execution.

Iteration in this phase is a numerical test instrument. It is not physical time evolution.

Convergence is not a physical relaxation claim.

Fixed-point stability labels are numerical classifications only.

## Inputs

Required Phase 0 files:

```text
../phase_0_setup/env_spec.yaml
../phase_0_setup/seed_config.json
../phase_0_setup/run_manifest.json
```

Required Phase 1 files:

```text
../phase_1_operator_integrity/input/operator_registry.json
../phase_1_operator_integrity/outputs/operator_integrity_report.json
```

Required Phase 2 files:

```text
../phase_2_projection_kernel/input/kernel_definitions.yaml
../phase_2_projection_kernel/outputs/kernel_norm_report.json
../phase_2_projection_kernel/outputs/projection_residuals.csv
../phase_2_projection_kernel/outputs/kernel_profiles.csv
```

Required Phase 3 files:

```text
../phase_3_global_projector/outputs/projector_validation.json
../phase_3_global_projector/outputs/projector_residuals.csv
../phase_3_global_projector/outputs/closure_test_report.json
```

## Configuration

```text
config/phase_4_setup.json
```

## Execution

Run from the validation root:

```bash
python -m phase_4_fixed_point.tests.test_convergence
python -m phase_4_fixed_point.tests.test_norm_invariance
python -m phase_4_fixed_point.tests.test_iteration_stability
```

## Outputs

```text
outputs/fixed_point_metrics.json
outputs/convergence_series.csv
outputs/norm_invariance_report.json
outputs/iteration_stability.json
```

## Gate 4 Completion

Gate 4 is complete when:

- `phase_4_setup.json` exists
- `fixed_point_metrics.json` exists
- `convergence_series.csv` exists
- `norm_invariance_report.json` exists
- `iteration_stability.json` exists
- Phase 0 references are present
- Phase 1 references are present
- Phase 2 references are present
- Phase 3 references are present
- no sub-version labels are present
- no empirical, ASDV, SWORD, dynamical, minimality, or necessity claim is introduced

After Gate 4 passes, Phase 5 — Drift and Stability may begin.
