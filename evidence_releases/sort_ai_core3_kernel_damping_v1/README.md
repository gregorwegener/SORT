# SORT-AI Core-3 Kernel-Damping Evidence Release v1

**Release name:** `sort_ai_core3_kernel_damping_v1`  
**Status:** Scaffold prepared for extracted evidence artefacts  
**Scope:** AI.01, AI.04, AI.13  
**Reference architecture:** MOCK v4 frozen structural reference  
**Canonical kernel parameter:** `sigma0 = 0.00190643`

---

## Purpose

This directory is reserved for the reproducible SORT-AI Core-3 kernel-damping evidence release.

The release is intended to host the extracted contents of the corresponding evidence bundle ZIP. It supports a reproducible analysis-layer protocol for the Core-3 SORT-AI applications:

| Application | Coupling axis | Scenario scope |
|-------------|---------------|----------------|
| `AI.01` | physical / interconnect coupling | core, boundary, overlap scenarios |
| `AI.04` | logical / runtime-control coupling | core, boundary, overlap scenarios |
| `AI.13` | semantic / agentic coupling | core, boundary, overlap scenarios |

---

## Boundary Statement

This evidence release operates on top of the frozen MOCK v4 reference architecture. It does not modify MOCK v4, does not define a new MOCK version, and does not claim production deployment, empirical benchmarking, vendor-specific measurement, or execution by MOCK v4.

The intended claim is structural reproducibility of declared risk-transition calculations under the canonical SORT kernel scale parameter.

---

## Expected Directory Layout

After extraction, this directory should contain:

```text
evidence_releases/sort_ai_core3_kernel_damping_v1/
  README.md
  manifest.json
  CITATION.cff
  requirements.txt

  data/
    core3_metrics.csv
    scenarios.json
    ai01/
    ai04/
    ai13/

  docs/
    methodology.md
    mock_v4_reference_boundary.md
    non_claims.md
    source_notes.md

  scripts/
    kernel_damping.py
    run_all.py
    validate_manifest.py
    build_core3_evidence.py

  outputs_expected/
    core3_summary.json
    ai01_results.csv
    ai04_results.csv
    ai13_results.csv
    reproducibility_report.md

  source_material/
    AI.01 Kernel-Damping Evidence Set v1.md
    AI.04 Kernel-Damping Evidence Set v3.md
    AI.13 Kernel-Damping Evidence Set v1.md
```

---

## Reproduction Command

After the extracted bundle contents are committed, run:

```bash
cd evidence_releases/sort_ai_core3_kernel_damping_v1
python scripts/run_all.py
```

The generated outputs should match the files under `outputs_expected/` within declared rounding tolerances.
