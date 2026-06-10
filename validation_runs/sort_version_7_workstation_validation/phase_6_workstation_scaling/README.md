# Phase 6 — Workstation Scaling

This directory contains the workstation scaling artifacts for the SORT Version 7 validation run.

Repository:

```text
gregorwegener/SORT
```

Repository path:

```text
validation_runs/sort_version_7_workstation_validation/phase_6_workstation_scaling/
```

Phase 6 records runtime, memory, grid-scaling, and thread-scaling behavior under controlled single-workstation conditions.

## Scope

Phase 6 records:

- deterministic workstation execution
- grid-size behavior
- thread-count behavior
- wall-time measurements
- peak-memory measurements
- final residual sanity checks
- raw execution logs

Phase 6 does not claim performance superiority, production readiness, HPC scalability, cloud readiness, empirical validity, ASDV validity, or SWORD execution.

Safe-gate mode is the default executed gate profile. It runs grid `128`, thread counts `1, 2, 4, 8, 16, 32`, and two measured repeats. Declared grid and thread combinations not executed by safe-gate are recorded as skipped.

Full mode is implemented but not executed automatically.

## Inputs

Required Phase 0 files:

```text
../phase_0_setup/env_spec.yaml
../phase_0_setup/seed_config.json
../phase_0_setup/run_manifest.json
```

Required Phase 2 files:

```text
../phase_2_projection_kernel/src/kernel_definition.py
../phase_2_projection_kernel/src/projection_operator.py
../phase_2_projection_kernel/outputs/kernel_norm_report.json
../phase_2_projection_kernel/outputs/projection_residuals.csv
```

Required Phase 3 files:

```text
../phase_3_global_projector/src/global_projector.py
../phase_3_global_projector/outputs/projector_validation.json
../phase_3_global_projector/outputs/projector_residuals.csv
```

Optional Phase 4 sanity-check files:

```text
../phase_4_fixed_point/outputs/fixed_point_metrics.json
../phase_4_fixed_point/outputs/convergence_series.csv
```

Optional Phase 5 sanity-check files:

```text
../phase_5_drift_stability/outputs/drift_profiles.csv
../phase_5_drift_stability/outputs/stability_response.json
```

## Configuration

```text
config/phase_6_setup.json
```

## Execution

Safe-gate run:

```bash
python src/benchmark_runner.py --config config/phase_6_setup.json --mode safe-gate
```

Single run:

```bash
python src/benchmark_runner.py --config config/phase_6_setup.json --mode single --grid 128 --threads 4
```

Grid scaling:

```bash
python src/benchmark_runner.py --config config/phase_6_setup.json --mode grid
```

Thread scaling:

```bash
python src/benchmark_runner.py --config config/phase_6_setup.json --mode threads
```

Full run:

```bash
python src/benchmark_runner.py --config config/phase_6_setup.json --mode full
```

Dry run:

```bash
python src/benchmark_runner.py --config config/phase_6_setup.json --mode safe-gate --dry-run
```

## Thread Control on Windows

Example for 8 threads:

```powershell
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set NUMEXPR_NUM_THREADS=8
```

The selected thread environment is logged by the runner.

## Outputs

```text
outputs/scaling_results.csv
outputs/runtime_profiles.json
outputs/memory_profiles.json
outputs/raw_logs/
```

## Gate 6 Completion

Gate 6 is complete when:

- `phase_6_setup.json` exists
- `scaling_results.csv` exists
- `runtime_profiles.json` exists
- `memory_profiles.json` exists
- raw logs exist
- Phase 0 references are present
- Phase 2 references are present
- Phase 3 references are present
- skipped runs, if any, are explicitly recorded
- no sub-version labels are present
- no performance superiority, production-readiness, HPC, ASDV, SWORD, empirical, minimality, or necessity claim is introduced

After Gate 6 passes, Phase 7 — Artifact Freeze may begin.
