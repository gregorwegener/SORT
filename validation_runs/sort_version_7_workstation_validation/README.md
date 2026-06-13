# SORT Version 7 Workstation Validation

**Status:** Pre-audit frozen artefact — superseded for current validation claims  
**Original execution status:** Completed and frozen  
**Corrective status:** Independent revalidation required  
**Protected repository snapshot:** `archive/pre-audit-2026-06-13`  
**Snapshot commit:** `a0efa78c573e9e5d1dc9c0d681586f858b936b00`

## Status notice

This folder preserves the original SORT Version 7 workstation validation artefacts. The run was completed and frozen before the independent numerical audit of the historical MOCK v3 calculation and the subsequent review of the validation architecture.

The artefacts remain public for provenance and reproducibility analysis. They are not deleted, rewritten, or retroactively corrected.

Following the audit, this run is classified as a **pre-audit validation artefact**. Its recorded pass states must not be cited as the current validation status of SORT until a corrected protocol has been executed and independently replayed.

The revalidation must address at least:

- separation of the global projector from the Gaussian damping kernel;
- independent testing of projector idempotency;
- removal of projection-idempotency claims for a non-idempotent Gaussian filter;
- explicit negative controls for algebraic and numerical tests;
- audit-safe array ownership and copy semantics;
- consistent parameter provenance across phases;
- regenerated manifests only after the corrected run is complete.

## Preservation rule

All original phase folders, outputs, reports, manifests, hashes, and ZIP packages in this path remain historical artefacts. They must not be overwritten.

The corrected run must use a separate path and a separate frozen package.

## Repository location

```text
validation_runs/sort_version_7_workstation_validation/
```

## Historical phase structure

- `phase_0_setup/` — setup and reproducibility artefacts
- `phase_1_operator_integrity/` — operator-integrity artefacts
- `phase_2_projection_kernel/` — projection-kernel artefacts
- `phase_3_global_projector/` — global-projector artefacts
- `phase_4_fixed_point/` — fixed-point artefacts
- `phase_5_drift_stability/` — drift and stability artefacts
- `phase_6_workstation_scaling/` — workstation-scaling artefacts
- `phase_7_artifact_freeze/` — original frozen package
- `manifests/` — original validation manifests
- `artifacts/` — original generated artefacts

## Related audit notice

See the repository-level file:

```text
AUDIT_STATUS.md
```
