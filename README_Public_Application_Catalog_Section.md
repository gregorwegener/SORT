# Public Application Catalog

**Version:** 6.2 | **Date:** February 2026 | **Status:** Public Reference  
**Source of Record:** [`catalog.public.json`](catalog.public.json)

---

## Overview

The SORT Public Application Catalog defines 107 applications across five domains, providing a complete reference for the analytical and diagnostic capabilities of the SORT framework.

| Category | Applications |
|----------|-------------|
| Technical Domains (AI, CX, QS) | 91 |
| Meta-Domain (Sovereign) | 5 |
| Non-IP (Cosmology) | 11 |
| **Total** | **107** |

---

## Architecture: Three Orthogonal Axes

| Axis | Meaning | License-Relevant |
|------|---------|------------------|
| **Domain** (vertical) | Market, buyer group, regulatory context | Yes |
| **Cluster A–E** (horizontal) | License and patent levels | Yes |
| **Structural Dimensions V1–V4** (internal) | Ordering and explanation layer | No |

**Key Principle:**
- **Domains are the What** — market and buyer context
- **Clusters are the How** — license levels and patent bundles
- **Applications are the Example** — specific use cases within a cluster
- **Structural Dimensions are the Why** — ordering and explanation layer

---

## Domains

### Domain-ID Mapping

The JSON file uses machine-readable `domain_id` values:

| Label | domain_id | Type | Cluster Scope |
|-------|-----------|------|---------------|
| **SOV** | `sovereign` | Meta-Domain | A, C, E only |
| **AI** | `ai-systems` | Technical Domain | A, B, C, D, E |
| **CX** | `complex-systems` | Technical Domain | A, B, C, D, E |
| **QS** | `quantum-systems` | Technical Domain | A, B, C, D, E |
| **COSMO** | `cosmology` | Non-IP | — |

### Domain Descriptions

- **SORT-AI (52 applications):** Infrastructure stability, runtime control, training adaptation, agentic systems, and evidence for AI/HPC systems.
- **SORT-CX (28 applications):** Pipeline stability, service mesh analysis, cascading failures, financial system regimes, and governance diagnostics.
- **SORT-QS (11 applications):** Noise filtering, error correction, calibration drift, and quantum-classical hybrid workflows.
- **SORT-Sovereign (5 applications):** Strategic infrastructure assessment, vendor lock-in analysis, and regulatory decision support as meta-layer.
- **SORT-COSMO (11 applications):** Scientific publications addressing cosmological anomalies and tensions (non-licensed).

---

## Clusters

| Cluster | Label | Structural Focus |
|---------|-------|------------------|
| **A** | Coupling | Physical and logical coupling |
| **B** | Learning | Temporal adaptation and learning |
| **C** | Control | Operative control and coherence |
| **D** | Emergence | Emergent, non-linear behavior |
| **E** | Evidence | Traceability, auditability, justification |

### Cluster Distribution

| Domain | A | B | C | D | E | Total |
|--------|---|---|---|---|---|-------|
| AI | 26 | 6 | 9 | 10 | 1 | 52 |
| CX | 10 | 4 | 6 | 7 | 1 | 28 |
| QS | 4 | 1 | 3 | 2 | 1 | 11 |
| **Technical Sum** | **40** | **11** | **18** | **19** | **3** | **91** |
| SOV* | 2 | — | 1 | — | 2 | 5 |

*Sovereign uses only clusters A, C, E (Meta-Domain)

---

## Core-3 Entry Points

The Core-3 applications serve as primary entry points for AI infrastructure licensing:

| ID | Title | Cluster | Related Whitepaper |
|----|-------|---------|-------------------|
| `ai.01` | Interconnect Stability Control | A | *SORT-AI: Interconnect Stability and Cost per Performance* |
| `ai.04` | Runtime Control Coherence | C | *SORT-AI: Runtime Control Coherence* |
| `ai.13` | Agentic System Stability | D | *SORT-AI: Agentic System Stability* |

**Core-3 = Three cluster licenses (A + C + D). Clusters B and E not included.**

---

## Structural Dimensions (V1–V4)

Structural Dimensions organize applications within clusters. They are **not** a license level.

| Dimension | Label | Focus |
|-----------|-------|-------|
| **V1** | Observed Structural Phenomenon | What was observed? |
| **V2** | Structural Cause / Coupling | Why does this occur? |
| **V3** | Structural Effect Space (SORT) | How does SORT work here? |
| **V4** | Decision and Utilization Space | What decisions become possible? |

**Central Rule:** Application ID and Title are primary. V1–V4 are supplementary.

---

## File Reference

The authoritative machine-readable source is:

```
catalog.public.json
```

### JSON Structure

```json
{
  "version": "6.2",
  "domains": { ... },
  "applications": {
    "ai-systems": [ { "id": "ai.01", "title": "...", "cluster": "A", ... } ],
    "complex-systems": [ ... ],
    "quantum-systems": [ ... ],
    "sovereign": [ ... ],
    "cosmology": [ ... ]
  },
  "statistics": { ... }
}
```

### Usage Examples

**Query all AI-Domain Cluster A applications:**
```javascript
const clusterA = catalog.applications["ai-systems"].filter(app => app.cluster === "A");
```

**Find Core-3 applications:**
```javascript
const core3 = catalog.core_3.applications; // ["ai.01", "ai.04", "ai.13"]
```

**Get statistics:**
```javascript
catalog.statistics.technical_domains.sum // { total: 91, A: 40, B: 11, ... }
```

---

## Licensing Information

### Technical Domains (AI, CX, QS)

| Option | Clusters | Scope |
|--------|----------|-------|
| Infrastructure Only | A | No training, agentic, or evidence use |
| Core-3 Coverage | A + C + D | No training or evidence scope |
| Full Technical | A + B + C + D | No evidence use |
| Full Domain | A + B + C + D + E | Complete coverage |

### Sovereign Meta-License

The Sovereign license is a **meta-license** that supplements technical cluster licenses:
- Addresses decision capability, traceability, and strategic risks
- Projects results from AI, CX, QS onto governance level
- Target groups: Government, regulatory, sovereign cloud, procurement, legal

---

## Changelog

### v6.2 (February 2026)

**New applications added (47 total):**
- AI Domain: ai.31–ai.52 (22 new applications)
- CX Domain: cx.12–cx.28 (17 new applications)
- QS Domain: qs.09–qs.11 (3 new applications)
- COSMO Domain: cosmo.07–cosmo.11 (5 new applications)

**Statistics updated:**
- Total applications: 60 → 107
- Technical domains: 49 → 91
- Non-IP (COSMO): 6 → 11

### v6.1 (January 2026)

- JSON reference filename corrected to `catalog.public.json`
- Domain-ID mapping table added
- Sovereign formally clarified as Meta-Domain with selective cluster scope (A, C, E)
- Statistics tables separated by technical domains, meta-domain, and non-IP

---

## Related Documents

- [SORT Konsolidierte Lizenzmatrix v6.2](SORT_Konsolidierte_Lizenzmatrix_v6_2.md) — Full license matrix with V1–V4 dimensions
- [Public Application Catalog Overview](Public_Application_Catalog_Overview.md) — Human-readable application descriptions
- [SORT Whitepaper v6](SORT_Whitepaper_v6.md) — Framework reference
