# SORT Audit Status

**Date:** 2026-06-13  
**Status:** Corrective audit and revalidation initiated  
**Protected pre-audit snapshot:** `archive/pre-audit-2026-06-13`  
**Snapshot commit:** `a0efa78c573e9e5d1dc9c0d681586f858b936b00`

## Scope

An independent numerical audit identified issues that require a controlled corrective and revalidation process. The affected areas include the historical MOCK v3 validation interpretation, the distinction between the Gaussian kernel and an idempotent projector, the Hubble-drift sign convention, and the evidential status of the existing SORT Version 7 workstation validation run.

The historical artefacts remain preserved. They must not be silently overwritten or retroactively presented as if the corrected interpretation had already applied at the time of execution.

## Current status of major artefacts

| Artefact | Current status | Treatment |
| --- | --- | --- |
| MOCK v3 frozen outputs | Historical exploratory numerical snapshot | Preserve unchanged; accompany with audit and provenance notes |
| Canonical historical value `sigma0 = 0.00190642767773082` | Confirmed Golden-Run output | Retain as historical reference scale, not as a derived universal constant |
| MOCK v3 Gaussian kernel matrix | Numerically reproducible from the frozen value | Retain as historical kernel artefact |
| MOCK v3 idempotency claim | Not confirmed | Withdraw as validation evidence for the Gaussian filtering operator |
| Hubble-drift positive local-rate interpretation | Not confirmed as a direct engine prediction | Treat the stored value as a historical drift or attenuation amplitude pending a new response derivation |
| MOCK v4 | Frozen structural reference architecture | Remains preserved; does not constitute numerical execution evidence |
| SORT Version 7 workstation validation run | Pre-audit frozen artefact; superseded for current validation claims | Preserve unchanged and replace through a new audited validation run |
| SORT-AI Core-3 evidence release | Separate analysis-layer evidence protocol | Retain, but review provenance wording and avoid treating `sigma0` as independently validated by AI data |

## Validation-run notice

The existing SORT Version 7 workstation validation package and Zenodo record remain available for provenance. Following the audit, they are classified as a **pre-audit frozen artefact**. Their recorded pass states must not be cited as the current validation status of SORT until the corrected validation protocol has been executed and independently replayed.

A new audited run will separate:

- the global projector from the Gaussian damping kernel;
- projector idempotency from kernel stability and repeated damping;
- formal operator tests from registry completeness checks;
- expected outputs from generated outputs;
- historical reference parameters from newly derived quantities.

## Repository rule

No historical result files, manifests, ZIP packages, or archived outputs are to be overwritten during the corrective process. New calculations must be placed in a separately versioned validation or corrective-replay path.
