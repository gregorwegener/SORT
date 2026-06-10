# Phase 1 — Operator Integrity

This directory contains the operator integrity validation artifacts for the SORT Version 7 workstation validation run.

Repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_1_operator_integrity/
```

Phase 1 validates the declared 22-operator registry used by the Version 7 workstation validation run.

## Scope

Phase 1 validates:

- operator registry completeness
- per-operator idempotency
- structural balance of declared weights
- Jacobi consistency under the configured evaluation mode

Phase 1 does not prove operator minimality, uniqueness, structural necessity, generator independence, empirical validity, production validity, or SWORD execution.

## Inputs

Required Phase 0 files:

```text
../phase_0_setup/env_spec.yaml
../phase_0_setup/seed_config.json
../phase_0_setup/run_manifest.json
```

Required Phase 1 input:

```text
input/operator_registry.json
```

## Configuration

```text
config/phase_1_setup.json
```

## Execution

Run from the validation root:

```bash
python -m phase_1_operator_integrity.tests.test_idempotency
python -m phase_1_operator_integrity.tests.test_weights
python -m phase_1_operator_integrity.tests.test_jacobi
```

## Outputs

```text
outputs/operator_integrity_report.json
outputs/operator_residuals.csv
```

## Gate 1 Completion

Gate 1 is complete when:

- `operator_registry.json` exists and contains 22 declared operators
- `phase_1_setup.json` exists
- `operator_integrity_report.json` exists
- `operator_residuals.csv` exists
- idempotency summary is present
- balance summary is present
- Jacobi summary is present
- Phase 0 references are present
- no sub-version labels are present

After Gate 1 passes, Phase 2 — Projection Kernel may begin.
