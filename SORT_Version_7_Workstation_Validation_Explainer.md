# SORT Version 7 Workstation Validation Run — Explainer

**Document status:** explanatory repository note  
**Scope:** explanation of the archived SORT Version 7 workstation validation artifact package  
**Frozen artifact:** `SORT_Version_7_Workstation_Validation.zip`  
**Zenodo DOI:** https://doi.org/10.5281/zenodo.20634212  
**Repository:** `gregorwegener/SORT`

---

## 1. Purpose of this note

This note explains the purpose, structure, and interpretation of the **SORT Version 7 Workstation Validation Run**.

The validation run is archived as a frozen artifact package:

```text
validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/outputs/SORT_Version_7_Workstation_Validation.zip
```

The package is identified by the following SHA-256 hash:

```text
2fc5e68551f70ac25e8970e51204de80c03d593003aab8123091366cac8df505
```

This explanatory note is **not itself part of the frozen ZIP package**. It is a post-freeze repository-level explanation intended to help readers understand what was validated, why the validation run matters, and how it differs from the frozen MOCK v4 reference architecture.

The canonical frozen artifact remains the ZIP package and its recorded hash.

---

## 2. What the validation run is

The SORT Version 7 Workstation Validation Run is a deterministic Level-0 structural validation sequence.

Its purpose is to test whether the declared SORT Version 7 operator, projection, kernel, fixed-point, drift, stability, workstation execution, and artifact-freeze structure can be executed reproducibly under controlled workstation conditions.

In compact form:

```text
SORT Version 7 definitions
→ deterministic validation inputs
→ phase-specific Python scripts
→ generated outputs
→ reports
→ hashes
→ frozen ZIP package
→ Zenodo DOI
```

The run is organized into eight phases:

| Phase | Gate | Main question |
|---|---:|---|
| Phase 0 — Setup and Reproducibility | Gate 0 | Is the execution environment documented? |
| Phase 1 — Operator Integrity | Gate 1 | Are the declared operators internally consistent? |
| Phase 2 — Projection Kernel | Gate 2 | Does the declared kernel behave as a valid projection interface? |
| Phase 3 — Global Projector | Gate 3 | Can the global projector be constructed and validated? |
| Phase 4 — Fixed-Point Structure | Gate 4 | Does repeated projection converge under the declared rule? |
| Phase 5 — Drift and Stability | Gate 5 | Does the framework distinguish drift and recover from perturbation? |
| Phase 6 — Workstation Scaling | Gate 6 | Can the validation apparatus execute on the declared workstation? |
| Phase 7 — Artifact Freeze | Gate 7 | Are the outputs inventoried, hashed, packaged, and frozen? |

All gates passed in the frozen validation package.

---

## 3. What the validation run is not

The validation run has a deliberately narrow scope.

It does **not** constitute:

- empirical validation;
- production validation;
- benchmark superiority evidence;
- customer-system validation;
- production telemetry analysis;
- SWORD execution;
- ASDV execution;
- HPC validation;
- cloud-readiness validation;
- a new MOCK version;
- a proof of minimality;
- a proof of structural necessity;
- a proof of uniqueness;
- a replacement for domain-specific Level-1 theories.

The validation run supports the following narrower statement:

> The declared SORT Version 7 Level-0 operator, projection, kernel, fixed-point, drift, stability, workstation execution, and artifact-freeze chain was implemented and passed deterministic validation gates under controlled workstation conditions.

---

## 4. Why this validation run matters

SORT Version 7 is positioned as a Level-0 structural assessment framework.

At this level, the relevant question is not whether SORT produces empirical predictions, replaces established dynamical theories, or optimizes a production system.

The relevant question is:

> Are the declared structural objects internally consistent, executable, reproducible, and packageable as a stable validation artifact?

The validation run answers this question at the computational validation level.

It provides a bridge between the conceptual framework and a reproducible artifact chain:

```text
mathematical structure
→ executable validation
→ reproducibility manifest
→ artifact inventory
→ hash manifest
→ frozen package
→ citation record
```

This matters because Version 7 is intended to consolidate SORT before later work on embedding competence, generator analysis, minimality analysis, and potential execution-layer development.

---

## 5. Relationship to MOCK v4

MOCK means **Model-Operator Consistency Kernel**.

MOCK v4 is the frozen structural reference architecture for SORT. It defines structural contracts, operator topology, projection conventions, domain boundaries, catalog conventions, and evidence-bundle primitives.

MOCK v4 is not an execution engine.

The Version 7 Workstation Validation Run is different:

| Layer | Role |
|---|---|
| MOCK v4 | Frozen structural reference architecture |
| SORT Version 7 Workstation Validation Run | Deterministic validation sequence executed on top of the reference structure |
| Future SWORD layer | Operator-resolved execution layer, not part of this validation run |

The validation run does not replace MOCK v4 and does not define a new MOCK version. It validates a deterministic Level-0 structural run sequence that uses the frozen architectural reference as the basis for reproducible testing.

---

## 6. Validation root

The validation run is located under:

```text
validation_runs/sort_version_7_workstation_validation/
```

The directory contains:

```text
phase_0_setup/
phase_1_operator_integrity/
phase_2_projection_kernel/
phase_3_global_projector/
phase_4_fixed_point/
phase_5_drift_stability/
phase_6_workstation_scaling/
phase_7_artifact_freeze/
manifests/
artifacts/
README.md
```

Each phase follows the same general pattern:

```text
config/      declared validation settings
input/       references to prior phase outputs or fixed inputs
src/         phase-specific validation scripts
outputs/     generated reports, metrics, and artifacts
README.md   local phase documentation
```

The phase README files are concise technical notes. This explainer provides the higher-level interpretation.

---

## 7. Phase-by-phase explanation

### 7.1 Phase 0 — Setup and Reproducibility

Phase 0 records the execution context.

It documents:

- run identifier;
- repository identity;
- local working copy;
- machine profile;
- operating system;
- Python version;
- Git metadata;
- seed configuration;
- validation scope.

The run identifier is:

```text
sort-version-7-workstation-validation-20260610T213622p0200
```

The recorded execution context includes:

```text
machine = ThinkStation
operating system = Windows
CPU = Intel Core i9-13900K
Python = 3.13.5
logical threads = 32
RAM ≈ 64 GB
```

Phase 0 produces no scientific result. Its purpose is traceability.

Without Phase 0, later outputs would exist as files, but the environment, seed, and run identity would not be sufficiently documented.

---

### 7.2 Phase 1 — Operator Integrity

Phase 1 validates the declared 22-operator structure.

The central checks are:

| Check | Purpose |
|---|---|
| Operator count | Confirms that the registry contains 22 declared operators |
| Matrix dimension | Confirms 22-dimensional representation |
| Idempotency | Confirms $\hat{O}_i^2=\hat{O}_i$ for each declared operator |
| Balance | Confirms the declared weight-balance condition |
| Jacobi sampling | Checks sampled algebraic consistency relations |

The key structural property is idempotency:

$$\hat{O}_i^2=\hat{O}_i$$

This means that once a structural projection has been applied, applying the same projection again does not change the result further.

In Phase 1, all 22 operators passed the idempotency check, the balance residual was zero, and the sampled Jacobi checks passed.

This phase validates operator integrity only. It does not prove minimality, uniqueness, or structural necessity.

---

### 7.3 Phase 2 — Projection Kernel

Phase 2 validates the declared projection-kernel interface.

The core object is the structural projection kernel $\kappa(k)$, used to define a projection interface $\pi_\kappa$.

The phase checks:

- kernel normalization;
- deterministic finite-grid behavior;
- projection idempotency;
- behavior across declared $k$-values;
- consistency with the declared kernel definition.

The tested $k$-values are:

```text
0.25, 0.5, 1.0, 2.0, 4.0
```

The phase uses synthetic states and a finite coordinate grid. It does not use empirical data and does not fit parameters.

The core projection idea is that the kernel profile defines a normalized vector, from which a rank-one projection is constructed. This makes the projection testable for idempotency.

Phase 2 passed all kernel normalization and projection-idempotency checks.

This phase validates the declared structural projection interface only. It does not establish cross-domain universality of $\sigma_0$ and does not establish physical fundamentality of $\sigma_0$.

---

### 7.4 Phase 3 — Global Projector

Phase 3 validates the global projector $\hat{H}$ in the finite workstation representation.

In the controlled 22-dimensional representation used here, $\hat{H}$ is constructed from the declared coordinate projection operators. This yields the identity projector in the finite validation space.

The phase checks:

- global projector construction;
- idempotency of $\hat{H}$;
- closure of projected states in the declared finite vector space;
- explicit operator-pair composition checks;
- absence of unvalidated global-commutativity assumptions.

The key point is that Phase 3 does **not** assume unrestricted global commutativity. It validates the declared finite projector interface and the explicitly tested composition structure.

Phase 3 passed the global-projector, closure, and composition checks.

This phase verifies that the operator-level structure can be assembled into a global projector interface under controlled finite-dimensional validation conditions.

---

### 7.5 Phase 4 — Fixed-Point Structure

Phase 4 validates the behavior of repeated projection.

The iteration rule is:

```text
H_after_pi_kappa
```

Conceptually:

$$\Psi_{n+1}=\hat{H}(\pi_\kappa(\Psi_n))$$

The phase starts from synthetic initial states and repeatedly applies the projection rule. It then measures whether the resulting sequence converges, remains neutral, oscillates, or diverges.

The key checks are:

- convergence under repeated projection;
- final residual size;
- classification of fixed-point behavior;
- norm tracking;
- repeatability;
- perturbation classification.

All tested states converged under the declared rule.

The norm behavior is classified as `contractive-projection`. This is important: norm contraction caused by projection is not interpreted as physical energy loss. It is a structural effect of projection in the validation apparatus.

Phase 4 demonstrates that the declared projection apparatus is iteratively controlled under the tested synthetic conditions.

---

### 7.6 Phase 5 — Drift and Stability

Phase 5 validates drift and stability behavior.

The drift metric is:

$$D(\Psi)=\|\hat{H}_{eff}(\Psi)-\Psi\|$$

Because $\hat{H}$ is the identity projector in the finite Phase 3 representation, Phase 5 uses an effective projector definition:

$$\hat{H}_{eff}(\Psi)=\hat{H}(\pi_\kappa(\Psi))$$

This avoids a trivial drift metric and aligns Phase 5 with the Phase 4 iteration rule.

Phase 5 tests:

- projector-invariant states;
- slightly violated states;
- strongly violated states;
- monotonic drift separation;
- perturbation recovery;
- raw drift scaling behavior;
- normalized drift invariance.

The phase checks whether drift increases in the expected order:

```text
projector-invariant
→ slightly violated
→ strongly violated
```

It also tests whether perturbed states return toward the projection structure.

Phase 5 passed the drift and stability checks. In the frozen run, all tested perturbation responses were classified as stable.

This phase is important because it moves beyond simple algebraic consistency. It tests whether the validation apparatus can distinguish structural deviation and recover from controlled perturbation.

---

### 7.7 Phase 6 — Workstation Scaling

Phase 6 validates that the apparatus can be executed on a local workstation and records runtime behavior.

It does not claim performance superiority.

The declared scaling matrix includes:

```text
grids = 128, 256, 512
threads = 1, 2, 4, 8, 16, 32
```

The executed profile is the safe workstation gate run:

```text
executed grid = 128
executed threads = 1, 2, 4, 8, 16, 32
```

Larger declared runs are recorded as intentionally skipped under the safe-gate policy. They remain available through the full execution mode but were not required for the frozen workstation gate.

Phase 6 records:

- runtime profiles;
- memory-profile information;
- raw execution logs;
- thread behavior;
- machine profile;
- safe-gate execution status.

This phase is not HPC validation, cloud validation, production validation, or SWORD execution. It is a controlled workstation execution record.

---

### 7.8 Phase 7 — Artifact Freeze

Phase 7 freezes the validation package.

It does not generate new scientific results. It performs artifact governance.

Phase 7 produces:

- pre-freeze audit report;
- artifact inventory;
- hash manifest;
- reproducibility manifest;
- freeze report;
- frozen ZIP package.

The frozen ZIP package is:

```text
SORT_Version_7_Workstation_Validation.zip
```

Its SHA-256 hash is:

```text
2fc5e68551f70ac25e8970e51204de80c03d593003aab8123091366cac8df505
```

The freeze report records the ZIP name, path, SHA-256 hash, package size, Phase 6 inclusion status, and non-claims.

Phase 7 makes the validation run citable and auditable.

---

## 8. How to inspect the package

The frozen package can be inspected by listing the ZIP contents.

From the Phase 7 output directory:

```bash
cd validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/outputs
python -m zipfile -l SORT_Version_7_Workstation_Validation.zip
```

On Windows PowerShell, the ZIP hash can be checked with:

```powershell
Get-FileHash SORT_Version_7_Workstation_Validation.zip -Algorithm SHA256
```

Expected SHA-256:

```text
2fc5e68551f70ac25e8970e51204de80c03d593003aab8123091366cac8df505
```

---

## 9. Main reproducibility artifacts

The most important files are:

| Artifact | Path |
|---|---|
| Frozen ZIP package | `validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/outputs/SORT_Version_7_Workstation_Validation.zip` |
| Freeze report | `validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/outputs/freeze_report.json` |
| Reproducibility manifest | `validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/outputs/repro_manifest.json` |
| Artifact inventory | `validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/outputs/artifact_inventory.csv` |
| Hash manifest | `validation_runs/sort_version_7_workstation_validation/phase_7_artifact_freeze/outputs/hashes.txt` |
| Phase 0 run manifest | `validation_runs/sort_version_7_workstation_validation/phase_0_setup/run_manifest.json` |
| Phase 1 operator report | `validation_runs/sort_version_7_workstation_validation/phase_1_operator_integrity/outputs/operator_integrity_report.json` |
| Phase 2 kernel report | `validation_runs/sort_version_7_workstation_validation/phase_2_projection_kernel/outputs/kernel_norm_report.json` |
| Phase 3 projector report | `validation_runs/sort_version_7_workstation_validation/phase_3_global_projector/outputs/projector_validation.json` |
| Phase 4 fixed-point report | `validation_runs/sort_version_7_workstation_validation/phase_4_fixed_point/outputs/fixed_point_metrics.json` |
| Phase 5 drift definition | `validation_runs/sort_version_7_workstation_validation/phase_5_drift_stability/outputs/drift_metric_definition.json` |
| Phase 5 stability response | `validation_runs/sort_version_7_workstation_validation/phase_5_drift_stability/outputs/stability_response.json` |
| Phase 6 runtime profile | `validation_runs/sort_version_7_workstation_validation/phase_6_workstation_scaling/outputs/runtime_profiles.json` |

---

## 10. What the validation run establishes

The validation run establishes that, under the declared finite workstation representation:

1. The 22 declared operators pass the implemented integrity checks.
2. The declared projection kernel can be normalized and used as an idempotent projection interface.
3. The global projector interface can be constructed and validated.
4. The projection apparatus exhibits controlled fixed-point behavior.
5. The drift metric distinguishes controlled structural deviations.
6. Perturbed states return under the tested stability protocol.
7. The validation apparatus executes on the declared workstation profile.
8. The resulting artifacts can be inventoried, hashed, packaged, frozen, cited, and archived.

The validation run therefore supports the statement:

> SORT Version 7 has a reproducible Level-0 workstation validation artifact package for its declared operator, kernel, projector, fixed-point, drift, stability, execution, and freeze protocol.

---

## 11. What the validation run does not establish

The validation run does not establish:

- that SORT is empirically correct;
- that SORT is a physical theory;
- that SORT replaces GR, QFT, $\Lambda$CDM, AI systems theory, control theory, or any other Level-1 theory;
- that the 22-operator basis is minimal;
- that the 22-operator basis is necessary;
- that $\sigma_0$ is a universal constant;
- that the projection kernel has been fitted to empirical data;
- that the apparatus works in production;
- that the apparatus improves runtime systems;
- that SWORD has been implemented;
- that ASDV has been executed.

These questions belong to later research stages.

---

## 12. Interpretation for SORT Version 7

The validation run supports Version 7 as a structural consolidation stage.

It shows that the declared Level-0 structures are not only described conceptually, but can also be translated into a deterministic validation apparatus.

This is a limited but important result.

The correct interpretation is:

```text
constructive structural validation
```

not:

```text
empirical proof
```

not:

```text
production validation
```

not:

```text
minimality proof
```

not:

```text
structural necessity proof
```

This distinction is central to the scientific positioning of SORT Version 7.

---

## 13. Citation

If you use or reference this validation artifact package, please cite:

Wegener, G. H. (2026). *gregorwegener/SORT: SORT Version 7 Workstation Validation Run — Frozen Artifact Package (sort-v7-workstation-validation-v1.0.0).* Zenodo. https://doi.org/10.5281/zenodo.20634212

```bibtex
@software{wegener_2026_sort_v7_workstation_validation,
  author       = {Wegener, Gregor H.},
  title        = {{gregorwegener/SORT: SORT Version 7 Workstation Validation Run — Frozen Artifact Package}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {sort-v7-workstation-validation-v1.0.0},
  doi          = {10.5281/zenodo.20634212},
  url          = {https://doi.org/10.5281/zenodo.20634212}
}
```

---

## 14. Summary

The SORT Version 7 Workstation Validation Run is best understood as a reproducible structural validation artifact.

It takes the declared SORT Version 7 Level-0 objects and tests them through a controlled sequence:

```text
environment
→ operators
→ kernel
→ global projector
→ fixed-point structure
→ drift and stability
→ workstation execution
→ artifact freeze
```

The result is a frozen, hashed, archived package that documents the computational validation status of SORT Version 7 under single-workstation conditions.

This package does not complete the SORT research program. It provides a stable validation foundation for the next stage of the research program.
