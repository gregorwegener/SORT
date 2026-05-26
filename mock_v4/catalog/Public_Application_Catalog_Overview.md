# Public Application Catalog Overview

**Version:** 6.2  
**Date:** February 2026  
**Status:** Public Reference  
**Source of Record:** `catalog.public.json`

---

## Purpose

This document provides an executive-readable overview of the SORT Public Application Catalog. Each application represents a distinct analytical or diagnostic perspective within the SORT framework.

The authoritative machine-readable source is `catalog.public.json`.

---

## Architecture: Three Orthogonal Axes

| Axis | Meaning | License-Relevant |
|------|---------|------------------|
| **Domain** | Market, buyer group, regulatory context | Yes |
| **Cluster A–E** | License and patent levels | Yes |
| **Structural Dimensions V1–V4** | Ordering and explanation layer | No |

**Interpretation:** Domains are the What, Clusters are the How, Applications are the Example, and Structural Dimensions are the Why.

---

## Domain-ID Mapping

| Label | `domain_id` | Type | Cluster Scope |
|-------|-------------|------|---------------|
| **SOV** | `sovereign` | Meta-Domain | A, C, E only |
| **AI** | `ai-systems` | Technical Domain | A, B, C, D, E |
| **CX** | `complex-systems` | Technical Domain | A, B, C, D, E |
| **QS** | `quantum-systems` | Technical Domain | A, B, C, D, E |
| **COSMO** | `cosmology` | Non-IP | none |

---

## Cluster Structure

| Cluster | Label | Structural Focus |
|---------|-------|------------------|
| **A** | Coupling | Physical and logical coupling |
| **B** | Learning | Temporal adaptation and learning |
| **C** | Control | Operative control and coherence |
| **D** | Emergence | Emergent, non-linear behavior |
| **E** | Evidence | Traceability, auditability, justification |

---

## Structural Dimensions V1–V4

Structural Dimensions are not a license level and not a product description. They structure applications within clusters and across domains.

| Dimension | Label | Focus |
|-----------|-------|-------|
| **V1** | Observed Structural Phenomenon | What was observed? |
| **V2** | Structural Cause / Coupling | Why does this occur? |
| **V3** | Structural Effect Space (SORT) | How does SORT work here? |
| **V4** | Decision and Utilization Space | What decisions become possible? |

Application ID and Title are primary. V1–V4 are supplementary.

---

## Core-3 Entry Points

| Application ID | Title | Cluster | Related Whitepaper |
|----------------|-------|---------|-------------------|
| `ai.01` | Interconnect Stability Control | AI-A | *SORT-AI: Interconnect Stability and Cost per Performance* |
| `ai.04` | Runtime Control Coherence | AI-C | *SORT-AI: Runtime Control Coherence* |
| `ai.13` | Agentic System Stability | AI-D | *SORT-AI: Agentic System Stability* |

**Core-3 = three cluster licenses (A + C + D). Clusters B and E are not included.**

---

## SORT-Sovereign Domain

The Sovereign Domain is a Meta-Domain with 5 applications. It projects structural results from AI, CX, and QS onto strategic, regulatory, and state decision spaces. It uses only clusters A, C, and E.

| ID | Title | Cluster |
|----|-------|---------|
| `sov.01` | Sovereign Infrastructure Stability & Bottleneck Risk Control | A |
| `sov.02` | Structural Vendor Lock-In Stability and Exit Risk Assessment | E |
| `sov.03` | Sovereign Runtime Auditability and Control Transparency | E |
| `sov.04` | Multi-Layer Infrastructure Dependency and Bottleneck Analysis | A |
| `sov.05` | Strategic Decision Support for Regulatory and State Actors | C |

---

## SORT-AI Domain

SORT-AI contains 52 applications across five clusters.

| Cluster | Focus | Count |
|---------|-------|------:|
| A | Infrastructure & Interconnect Stability | 26 |
| B | Temporal Adaptation & Training | 6 |
| C | Runtime Control & Coherence | 9 |
| D | Emergence & Agentic Systems | 10 |
| E | Evidence & Assurance | 1 |

Important AI applications include the Core-3 entry points `ai.01`, `ai.04`, and `ai.13`, plus adjacent application families such as accelerator runtime control, structural network scalability, virtualization overhead stability, benchmark drift, evaluation context instability, internal mechanism audit, and deployment drift signal aggregation.

---

## SORT-CX Domain

SORT-CX contains 28 applications across five clusters.

| Cluster | Focus | Count |
|---------|-------|------:|
| A | Coupling & Dependencies | 10 |
| B | Temporal Adaptation & Drift | 4 |
| C | Control & Regulation | 6 |
| D | Emergence & System-of-Systems | 7 |
| E | Evidence & Assurance | 1 |

---

## SORT-QS Domain

SORT-QS contains 11 applications across five clusters.

| Cluster | Focus | Count |
|---------|-------|------:|
| A | State Space & Coupling | 4 |
| B | Temporal Adaptation | 1 |
| C | Control & Measurement | 3 |
| D | Emergence & Regime Shifts | 2 |
| E | Evidence & Assurance | 1 |

---

## SORT-COSMO Domain

SORT-COSMO contains 11 non-IP scientific applications. These applications serve scientific publications and are not licensed.

| App-ID | Title |
|--------|-------|
| `cosmo.01` | Early Galaxies |
| `cosmo.02` | Early SMBH Seeds |
| `cosmo.03` | Hubble Drift |
| `cosmo.04` | CMB Anomalies |
| `cosmo.05` | Dark Baryon Oscillator |
| `cosmo.06` | Intergalactic Bridges |
| `cosmo.07` | Dark Flow Drift Signature Analysis |
| `cosmo.08` | CMB Signal Separation Diagnostics |
| `cosmo.09` | Quantum-Classical Transition Projection |
| `cosmo.10` | Metric Consistency Analysis |
| `cosmo.11` | Reionization Dynamics Modeling |

---

## Summary Statistics

### Technical Domains

| Domain | Apps | A | B | C | D | E |
|--------|-----:|--:|--:|--:|--:|--:|
| AI | 52 | 26 | 6 | 9 | 10 | 1 |
| CX | 28 | 10 | 4 | 6 | 7 | 1 |
| QS | 11 | 4 | 1 | 3 | 2 | 1 |
| **Sum Tech** | **91** | **40** | **11** | **18** | **19** | **3** |

### Meta-Domain

| Domain | Apps | A | C | E |
|--------|-----:|--:|--:|--:|
| SOV | 5 | 2 | 1 | 2 |

### Non-IP Domain

| Domain | Apps |
|--------|-----:|
| COSMO | 11 |

### Total

| Category | Apps |
|----------|-----:|
| Technical Domains | 91 |
| Meta-Domain | 5 |
| Non-IP | 11 |
| **Total** | **107** |

---

## Changelog

### v6.2 (February 2026)

- Added 47 new applications.
- Updated total applications from 60 to 107.
- Updated technical domains from 49 to 91 applications.
- Updated COSMO from 6 to 11 applications.
- Added Core-3 entry points and formalized SOV as a meta-domain.

### v6.1 (January 2026)

- Corrected JSON reference filename to `catalog.public.json`.
- Added domain-ID mapping table.
- Clarified Sovereign as Meta-Domain with selective cluster scope.
- Made explicit that Application ID and Title are primary while V1–V4 are supplementary.
