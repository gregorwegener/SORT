# SORT Version 7 Workstation Validation

This repository folder contains the workstation validation artifacts for SORT Version 7. The purpose of these artifacts is to document deterministic, reproducible structural validation of the Level-0 SORT apparatus under controlled single-node conditions.

The validation run is maintained inside the official SORT GitHub repository under:

```text
validation_runs/sort_version_7_workstation_validation/
```

The validation runs operate on top of the frozen MOCK v4 structural reference architecture. They do not constitute a new MOCK version, do not implement SWORD, do not use empirical datasets, do not perform model fitting, and do not claim production or benchmark validation.

## Phase Structure

- `phase_0_setup/` — setup and reproducibility artifacts
- `phase_1_operator_integrity/` — reserved for Phase 1
- `phase_2_projection_kernel/` — reserved for Phase 2
- `phase_3_global_projector/` — reserved for Phase 3
- `phase_4_fixed_point/` — reserved for Phase 4
- `phase_5_drift_stability/` — reserved for Phase 5
- `phase_6_workstation_scaling/` — reserved for Phase 6
- `phase_7_artifact_freeze/` — reserved for Phase 7
- `manifests/` — reserved for validation manifests
- `artifacts/` — reserved for generated artifacts

Only Phase 0 is implemented at this stage.
