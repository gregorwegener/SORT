# Phase 3 — Global Projector

This directory contains the global projector validation artifacts for the SORT Version 7 workstation validation run.

Repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_3_global_projector/
```

Phase 3 validates the declared global projector `H` used by the Version 7 workstation validation run.

## Scope

Phase 3 validates:

- construction of the declared global projector
- projector idempotency
- closure of projected synthetic states
- declared operator-composition stability under `H`

Phase 3 does not use empirical data, does not introduce Level-1 dynamics, does not validate a physical model, does not assume global commutativity, does not perform ASDV, and does not constitute SWORD execution.

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

## Configuration

```text
config/phase_3_setup.json
```

## Execution

Run from the validation root:

```bash
python -m phase_3_global_projector.tests.test_projector_idempotency
python -m phase_3_global_projector.tests.test_closure
python -m phase_3_global_projector.tests.test_operator_composition
```

## Outputs

```text
outputs/projector_validation.json
outputs/projector_residuals.csv
outputs/closure_test_report.json
outputs/composition_stability.csv
```

## Gate 3 Completion

Gate 3 is complete when:

- `phase_3_setup.json` exists
- `projector_validation.json` exists
- `projector_residuals.csv` exists
- `closure_test_report.json` exists
- `composition_stability.csv` exists
- Phase 0 references are present
- Phase 1 references are present
- Phase 2 references are present
- no sub-version labels are present
- no empirical, ASDV, SWORD, dynamical, global-commutativity, minimality, or necessity claim is introduced

After Gate 3 passes, Phase 4 — Fixed-Point Structure may begin.
