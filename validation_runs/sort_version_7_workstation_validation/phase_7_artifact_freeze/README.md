# Phase 7 — Artifact Freeze

This directory contains the artifact-freeze outputs for the SORT Version 7 workstation validation run.

Repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/
```

Phase 7 freezes the completed workstation validation run into a reproducible evidence package.

## Scope

Phase 7 performs:

- pre-freeze consistency audit
- artifact inventory
- SHA-256 hash generation
- reproducibility manifest creation
- ZIP package creation
- final non-claim verification

Phase 7 does not generate new scientific results. It does not perform empirical validation, SWORD execution, ASDV execution, production validation, benchmark validation, minimality proof, or structural necessity proof.

Phase 6 is included in this freeze.

## Required Inputs

Phase 7 requires completed outputs from:

```text
phase_0_setup/
phase_1_operator_integrity/
phase_2_projection_kernel/
phase_3_global_projector/
phase_4_fixed_point/
phase_5_drift_stability/
```

Phase 6 is included:

```text
phase_6_workstation_scaling/
```

## Configuration

```text
config/phase_7_setup.json
```

## Execution

Run from the Phase 7 folder:

```bash
python src/prefreeze_audit.py --config config/phase_7_setup.json
python src/artifact_inventory.py --config config/phase_7_setup.json
python src/hash_manifest.py --config config/phase_7_setup.json
python src/freeze_package.py --config config/phase_7_setup.json
```

Alternatively, run all steps:

```bash
python src/freeze_package.py --config config/phase_7_setup.json --run-all
```

## Outputs

```text
outputs/prefreeze_audit_report.json
outputs/artifact_inventory.csv
outputs/hashes.txt
outputs/repro_manifest.json
outputs/freeze_report.json
outputs/SORT_Version_7_Workstation_Validation.zip
```

## Self-Reference Handling

Phase 7 includes generated manifest files in the ZIP package. The files `artifact_inventory.csv`, `hashes.txt`, `repro_manifest.json`, and `freeze_report.json` are self-referential, so Phase 7 does not attempt mathematically impossible final self-hashes for those files.

The final ZIP SHA-256 hash and ZIP size are recorded in the repository-side `outputs/freeze_report.json` after ZIP creation.

## Gate 7 Completion

Gate 7 is complete when:

- `prefreeze_audit_report.json` exists
- `artifact_inventory.csv` exists
- `hashes.txt` exists
- `repro_manifest.json` exists
- `freeze_report.json` exists
- `SORT_Version_7_Workstation_Validation.zip` exists
- ZIP SHA-256 hash is recorded
- Phase 6 status is recorded as included
- no sub-version labels are present
- no empirical, SWORD, ASDV, production, benchmark, minimality, or necessity claim is introduced

After Gate 7 passes, the SORT Version 7 workstation validation package is frozen.
