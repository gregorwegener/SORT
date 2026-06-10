# Phase 2 — Projection Kernel

This directory contains the projection-kernel validation artifacts for the SORT Version 7 workstation validation run.

Repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_2_projection_kernel/
```

Phase 2 validates the declared kernel family and projection interface used by the Version 7 workstation validation run.

## Scope

Phase 2 validates:

- kernel definition
- kernel normalization
- projection-interface behavior
- approximate projection idempotency
- scale-grid stability on synthetic states

Phase 2 does not use empirical data, does not fit parameters, does not validate a physical model, does not establish cross-domain `sigma_0` universality, does not perform ASDV, and does not constitute SWORD execution.

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

Required Phase 2 input:

```text
input/kernel_definitions.yaml
```

## Configuration

```text
config/phase_2_setup.json
```

## Execution

Run from the validation root:

```bash
python -m phase_2_projection_kernel.tests.test_kernel_normalization
python -m phase_2_projection_kernel.tests.test_projection_idempotency
```

## Outputs

```text
outputs/kernel_norm_report.json
outputs/kernel_profiles.csv
outputs/projection_residuals.csv
```

## Gate 2 Completion

Gate 2 is complete when:

- `kernel_definitions.yaml` exists
- `phase_2_setup.json` exists
- `kernel_norm_report.json` exists
- `kernel_profiles.csv` exists
- `projection_residuals.csv` exists
- Phase 0 references are present
- Phase 1 references are present
- no sub-version labels are present
- no empirical, ASDV, SWORD, or universality claim is introduced

After Gate 2 passes, Phase 3 — Global Projector may begin.
