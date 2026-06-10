# Phase 5 — Drift and Stability

This directory contains the drift and stability validation artifacts for the SORT Version 7 workstation validation run.

Repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_5_drift_stability/
```

Phase 5 defines and validates structural drift diagnostics on deterministic synthetic reference states.

Phase 5 uses the effective structural projection apparatus `H_eff(Psi) = H(pi_kappa(Psi))` for drift measurement because the Phase 3 global projector is the identity projector in the current workstation representation. This does not redefine Phase 2 or Phase 3.

## Scope

Phase 5 validates:

- synthetic reference-state generation
- structural drift metric definition
- drift-profile measurement
- perturbation response
- numerical stability classification
- metric behavior under declared synthetic transformations

Phase 5 does not use empirical data, does not perform production drift detection, does not validate a physical regime boundary, does not perform ASDV, and does not constitute SWORD execution.

Stability labels are numerical classifications only.

Drift labels are numerical classifications only.

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

Required Phase 4 files:

```text
../phase_4_fixed_point/outputs/fixed_point_metrics.json
../phase_4_fixed_point/outputs/convergence_series.csv
../phase_4_fixed_point/outputs/norm_invariance_report.json
../phase_4_fixed_point/outputs/iteration_stability.json
```

## Configuration

```text
config/phase_5_setup.json
```

## Execution

Run from the validation root:

```bash
python -m phase_5_drift_stability.tests.test_drift_monotonicity
python -m phase_5_drift_stability.tests.test_stability_recovery
python -m phase_5_drift_stability.tests.test_metric_invariance
```

## Outputs

```text
outputs/drift_profiles.csv
outputs/stability_response.json
outputs/synthetic_reference_catalog.json
outputs/drift_metric_definition.json
```

## Gate 5 Completion

Gate 5 is complete when:

- `phase_5_setup.json` exists
- `drift_profiles.csv` exists
- `stability_response.json` exists
- `synthetic_reference_catalog.json` exists
- `drift_metric_definition.json` exists
- Phase 0 references are present
- Phase 1 references are present
- Phase 2 references are present
- Phase 3 references are present
- Phase 4 references are present
- no sub-version labels are present
- no empirical, ASDV, SWORD, production, physical-regime, minimality, or necessity claim is introduced

After Gate 5 passes, Phase 6 — Workstation Scaling may begin.
