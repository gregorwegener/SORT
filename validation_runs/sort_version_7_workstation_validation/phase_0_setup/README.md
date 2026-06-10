# Phase 0 — Setup and Reproducibility

This directory contains the setup and reproducibility artifacts for the SORT Version 7 workstation validation run.

The validation run is maintained inside the official SORT GitHub repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_0_setup/
```

Phase 0 establishes the deterministic environment used by all later validation phases. It records the software environment, hardware profile, Git state, repository state, and global seed configuration.

## Files

- `env_spec.yaml` — software, hardware, and repository environment specification
- `seed_config.json` — deterministic seed policy with global seed `117666`
- `run_manifest.json` — audit manifest for this validation run
- `collect_env.py` — local environment collection script
- `README.md` — this file

## Scope

Phase 0 produces no scientific results.

It does not perform empirical validation, model fitting, SWORD execution, ASDV analysis, benchmark testing, or production-system analysis.

The purpose is reproducibility and auditability only.

## Execution

Run from the repository root:

```bash
python validation_runs/sort_version_7_workstation_validation/phase_0_setup/collect_env.py
```

## Gate 0 Completion

Gate 0 is complete when:

- `env_spec.yaml` exists
- `seed_config.json` exists
- `run_manifest.json` exists
- the global seed is fixed to `117666`
- the Git state is recorded or explicitly marked unavailable
- the official repository is recorded as `gregorwegener/SORT`
- the validation root is recorded as `validation_runs/sort_version_7_workstation_validation`
- no placeholder fields remain unless explicitly marked unavailable

After Gate 0 passes, Phase 1 — Operator Integrity may begin.
