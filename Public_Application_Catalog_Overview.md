# Public Application Catalog Overview

**Version:** 6.2  
**Date:** February 2026  
**Status:** Public Reference  
**Source of Record:** `catalog.public.json`

---

## Purpose

This document provides a complete, executive-readable overview of all applications defined in the SORT Public Application Catalog. Each application represents a distinct analytical or diagnostic perspective within the SORT framework.

The authoritative machine-readable source is `catalog.public.json`.

---

## Architecture: Three Orthogonal Axes

| Axis | Meaning | License-Relevant |
|------|---------|------------------|
| **Domain** (vertical) | Market, buyer group, regulatory context | Yes |
| **Cluster A–E** (horizontal) | License and patent levels | Yes |
| **Structural Dimensions V1–V4** (internal) | Ordering and explanation layer | No |

> **Domains are the What** — market and buyer context  
> **Clusters are the How** — license levels and patent bundles, representing how scope is packaged commercially  
> **Applications are the Example** — specific use cases within a cluster  
> **Structural Dimensions are the Why** — ordering and explanation layer

---

## Domain-ID Mapping

The JSON file `catalog.public.json` uses machine-readable `domain_id` values. This table shows the mapping to human labels used in documentation:

| Label | domain_id | Type | Cluster Scope |
|-------|-----------|------|---------------|
| **SOV** | `sovereign` | Meta-Domain | A, C, E only |
| **AI** | `ai-systems` | Technical Domain | A, B, C, D, E |
| **CX** | `complex-systems` | Technical Domain | A, B, C, D, E |
| **QS** | `quantum-systems` | Technical Domain | A, B, C, D, E |
| **COSMO** | `cosmology` | Non-IP | — |

The `domain_id` in JSON is the canonical machine representation.

---

## Cluster Structure

| Cluster | Structural Focus | System Level |
|---------|------------------|--------------|
| **A** | Physical and logical coupling | Coupling |
| **B** | Temporal adaptation and learning | Learning |
| **C** | Operative control and coherence | Control |
| **D** | Emergent, non-linear behavior | Emergence |
| **E** | Traceability, auditability, justification | Evidence |

---

## Structural Dimensions (V1–V4)

Structural Dimensions are **not a license level** and **not a product description**. They structure applications within clusters and across domains.

**Central Rule:** Structural Dimensions supplement application identity; they never replace it. Application ID and Title are primary, V1–V4 are secondary.

| Dimension | Label | Focus |
|-----------|-------|-------|
| **V1** | Observed Structural Phenomenon | Starting point: What was observed? |
| **V2** | Structural Cause / Coupling | Why does this occur? |
| **V3** | Structural Effect Space (SORT) | How does SORT work here? (Core) |
| **V4** | Decision and Utilization Space | What decisions become possible? |

---

## Core-3 Entry Points

| Application ID | Title | Cluster | Related Whitepaper |
|----------------|-------|---------|-------------------|
| ai.01 | Interconnect Stability Control | AI-A | *SORT-AI: Interconnect Stability and Cost per Performance* |
| ai.04 | Runtime Control Coherence | AI-C | *SORT-AI: Runtime Control Coherence* |
| ai.13 | Agentic System Stability | AI-D | *SORT-AI: Agentic System Stability* |

**Core-3 = three cluster licenses (A + C + D). Clusters B and E not included.**

---

## SORT-Sovereign Domain (Meta-Domain, 5 Applications)

### Special Status of Sovereign Domain

The Sovereign Domain is a **Meta-Domain** with special characteristics:

1. **Not a technical domain** — Projects structural results from AI, CX, and QS onto strategic, regulatory, and state decision spaces
2. **Selective cluster usage** — Only Clusters A (Coupling), C (Control), and E (Evidence) are permitted; Clusters B (Learning) and D (Emergence) are excluded
3. **Meta-license** — Sovereign licenses supplement technical cluster licenses, they do not replace them; they address decision-capability, traceability, and strategic risks

**Cluster Scope: Meta (A, C, E)**

---

### sov.01 — Sovereign Infrastructure Stability & Bottleneck Risk Control

**Title:** Sovereign Infrastructure Stability & Bottleneck Risk Control  
**One-Liner:** Strategic structural assessment of critical infrastructure as coupled system, modeling failure costs, time risks, and sovereignty implications.  
**Cluster:** A (Coupling)

| V1 | V2 | V3 | V4 |
|----|----|----|-----|
| National infrastructure shows unexpected fragility under scaling or disruption | Coupling between subsystems creates systemic risks beyond local metrics | Projection onto structural stability spaces; bottleneck identification | Strategic CapEx decisions, sovereignty assessment, time-risk estimation |

---

### sov.02 — Structural Vendor Lock-In Stability and Exit Risk Assessment

**Title:** Structural Vendor Lock-In Stability and Exit Risk Assessment  
**One-Liner:** Structural analysis of dependencies that jeopardize stability and controllability during vendor transitions or multi-vendor setups, translating technical dependencies into strategic risks.  
**Cluster:** E (Evidence)

| V1 | V2 | V3 | V4 |
|----|----|----|-----|
| Vendor transitions cause unexpected instability despite functional equivalence | Lock-in is structural through implicit control assumptions | Dependency graph analysis; evidence generation for exit risks | Exit strategy assessment, multi-vendor architecture decisions, procurement justification |

---

### sov.03 — Sovereign Runtime Auditability and Control Transparency

**Title:** Sovereign Runtime Auditability and Control Transparency  
**One-Liner:** Structural evidence provision for runtime control in sovereign infrastructures, enabling formal justification to regulatory and state actors.  
**Cluster:** E (Evidence)

| V1 | V2 | V3 | V4 |
|----|----|----|-----|
| Sovereign systems are functionally stable but not formally auditable | Gap between operational control and demonstrable control integrity | Evidence layer for runtime control decisions without implementation disclosure | Regulatory compliance, state certification, audit-readiness |

---

### sov.04 — Multi-Layer Infrastructure Dependency and Bottleneck Analysis

**Title:** Multi-Layer Infrastructure Dependency and Bottleneck Analysis  
**One-Liner:** Structural analysis of dependencies and bottlenecks across infrastructure layers (energy, network, compute, storage), identifying critical coupling paths.  
**Cluster:** A (Coupling)

| V1 | V2 | V3 | V4 |
|----|----|----|-----|
| Infrastructure layers show unexpected coupling effects under scaling or failure | Cross-layer dependencies create systemic bottlenecks | Structural projection of multi-layer dependencies onto stability spaces | Infrastructure investment decisions, capacity planning, risk prioritization |

---

### sov.05 — Strategic Decision Support for Regulatory and State Actors

**Title:** Strategic Decision Support for Regulatory and State Actors  
**One-Liner:** Structural decision support translating technical stability analyses into governance-relevant decision foundations for regulatory and strategic actors.  
**Cluster:** C (Control)

| V1 | V2 | V3 | V4 |
|----|----|----|-----|
| Technical stability analyses are not decision-relevant for regulatory actors | Gap between engineering perspective and governance requirements | Structural translation of stability spaces into decision foundations | Regulatory framework decisions, funding allocation, strategic infrastructure planning |

---

## SORT-AI Domain (52 Applications)

### AI-Cluster A — Infrastructure & Interconnect Stability (26 Apps)

#### ai.01 — Interconnect Stability Control ⭐ Core-3

**Title:** Interconnect Stability Control  
**One-Liner:** Structural stability diagnostics and control for interconnect induced performance collapse in distributed AI and HPC systems.  
**Cluster:** A | **Related Whitepaper:** *SORT-AI: Interconnect Stability and Cost per Performance*

---

#### ai.02 — Structural Drift Diagnostics for AI Workloads

**Title:** Structural Drift Diagnostics for AI Workloads  
**One-Liner:** Detect structural drift across training and inference pipelines beyond metrics and telemetry.  
**Cluster:** A

---

#### ai.05 — Data and Retrieval Structural Integrity

**Title:** Data and Retrieval Structural Integrity  
**One-Liner:** Structural integrity diagnostics for RAG pipelines and retrieval induced drift.  
**Cluster:** A

---

#### ai.06 — Energy-Interconnect Stability Coupling

**Title:** Energy-Interconnect Stability Coupling  
**One-Liner:** Analysis of feedback loops between load dynamics, power supply, and interconnect stability in AI campuses.  
**Cluster:** A

---

#### ai.07 — Accelerator Runtime Control

**Title:** Accelerator Runtime Control  
**One-Liner:** Structure compatible control for heterogeneous hardware execution across GPU, TPU, NPU, and ASIC fleets.  
**Cluster:** A

---

#### ai.08 — Structural Scalability Certification

**Title:** Structural Scalability Certification  
**One-Liner:** Formal certification whether a system remains structurally stable under planned scaling increments.  
**Cluster:** A

---

#### ai.09 — Control-Flow Instability Mapping

**Title:** Control-Flow Instability Mapping  
**One-Liner:** Structural analysis of complex control flows across CUDA, OpenCL, agent workflows, and orchestration layers.  
**Cluster:** A

---

#### ai.10 — Interconnect-Aware Control Flow Diagnostics

**Title:** Interconnect-Aware Control Flow Diagnostics  
**One-Liner:** Correlation of barriers, kernel launches, memory transfers, and network load for interconnect coupled instability.  
**Cluster:** A

---

#### ai.11 — Structural Network Scalability Risk Modeling

**Title:** Structural Network Scalability Risk Modeling  
**One-Liner:** Model based assessment of network scaling risks including topology, SDN, routing, and fault tolerance.  
**Cluster:** A

---

#### ai.12 — Network Scalability Stress Mapping

**Title:** Network Scalability Stress Mapping  
**One-Liner:** Dynamic projection of network states into a structural stability space for congestion and tail latency risk.  
**Cluster:** A

---

#### ai.14 — Virtualization Overhead Stability Analysis

**Title:** Virtualization Overhead Stability Analysis  
**One-Liner:** Structural instability analysis through virtualization, SR-IOV, RDMA, and multi tenant noise effects.  
**Cluster:** A

---

#### ai.15 — Memory-Interconnect Coupling Diagnostics

**Title:** Memory-Interconnect Coupling Diagnostics  
**One-Liner:** Analysis of coupling between memory bandwidth, memory latency, and interconnect synchronization behavior.  
**Cluster:** A

---

#### ai.19 — Kubernetes Control-Plane Stability Assessment

**Title:** Kubernetes Control-Plane Stability Assessment  
**One-Liner:** Structural impact assessment of control plane decisions on interconnect and runtime stability.  
**Cluster:** A

---

#### ai.20 — Structural Cloud Migration Risk Assessment

**Title:** Structural Cloud Migration Risk Assessment  
**One-Liner:** Structural analysis of on premises to cloud migrations across compute, network, and control plane layers.  
**Cluster:** A

---

#### ai.21 — SDN-Interconnect Stability Diagnostics

**Title:** SDN-Interconnect Stability Diagnostics  
**One-Liner:** Formal analysis of coupling between SDN control plane, cloud interconnects, and application runtime stability.  
**Cluster:** A

---

#### ai.26 — Distributed Training Synchronization Stability

**Title:** Distributed Training Synchronization Stability  
**One-Liner:** Structural stability analysis of distributed training synchronization including gradient aggregation and parameter server patterns.  
**Cluster:** A

---

#### ai.34 — Internal-External Representation Projection Diagnostics

**Title:** Internal-External Representation Projection Diagnostics  
**One-Liner:** Structural analysis of information loss in mapping internal computation to external explanation, identifying interpretability limits.  
**Cluster:** A

---

#### ai.35 — Capability Space Structural Decomposition

**Title:** Capability Space Structural Decomposition  
**One-Liner:** Orthogonal decomposition of AI capability dimensions with safety region mapping across autonomy, generality, and intelligence axes.  
**Cluster:** A

---

#### ai.40 — Training Data Poisoning Backdoor Diagnostics

**Title:** Training Data Poisoning Backdoor Diagnostics  
**One-Liner:** Structural detection of hidden backdoor patterns in trained models, analyzing coupling between data artifacts and model behavior.  
**Cluster:** A

---

#### ai.42 — Prompt Injection Surface Mapping

**Title:** Prompt Injection Surface Mapping  
**One-Liner:** Structural boundary analysis between instruction space and policy space, mapping jailbreak vulnerability surfaces.  
**Cluster:** A

---

#### ai.44 — Multimodal Injection Isolation Diagnostics

**Title:** Multimodal Injection Isolation Diagnostics  
**One-Liner:** Structural separation of semantic content from control signals across modalities, analyzing cross-modal safety boundaries.  
**Cluster:** A

---

#### ai.46 — World Model Projection-Execution Gap Diagnostics

**Title:** World Model Projection-Execution Gap Diagnostics  
**One-Liner:** Structural diagnostics of drift between simulated plan space and execution reality, analyzing imagination-execution coherence.  
**Cluster:** A

---

#### ai.49 — Training Constraint Conflict Detection

**Title:** Training Constraint Conflict Detection  
**One-Liner:** Structural scanning for implicit contradictions in training environments that produce unstable attractors.  
**Cluster:** A

---

#### ai.51 — Internal Mechanism Structural Audit

**Title:** Internal Mechanism Structural Audit  
**One-Liner:** Structural audit methodology treating internal computations as operator graphs, providing interpretability meta-framework.  
**Cluster:** A

---

#### ai.52 — Deployment Drift Signal Aggregation

**Title:** Deployment Drift Signal Aggregation  
**One-Liner:** Structural framework for distributed weak signal aggregation across deployment environments, enabling live monitoring patterns.  
**Cluster:** A

---

### AI-Cluster B — Temporal Adaptation & Training (6 Apps)

#### ai.16 — Benchmark Integrity and Drift Diagnostics

**Title:** Benchmark Integrity and Drift Diagnostics  
**One-Liner:** Structural stability metrics complementing classical benchmarks to detect drift across releases and configurations.  
**Cluster:** B

---

#### ai.22 — Structural Architecture Stability Diagnostics for Large-Scale AI Models

**Title:** Structural Architecture Stability Diagnostics for Large-Scale AI Models  
**One-Liner:** Pre-training and early-training stability analysis for large-scale AI model architectures, identifying structural risk from information flow, residual paths, and routing mechanisms.  
**Cluster:** B

---

#### ai.25 — Training Pipeline Consistency Monitoring

**Title:** Training Pipeline Consistency Monitoring  
**One-Liner:** Structural consistency monitoring across training pipeline stages, detecting drift before it affects model quality.  
**Cluster:** B

---

#### ai.29 — Structural Continual Learning Stability Assessment

**Title:** Structural Continual Learning Stability Assessment  
**One-Liner:** Structural assessment of stability, control, and forgetting risks in post-hoc model adaptation and incremental learning.  
**Cluster:** B

---

#### ai.41 — Fine-Tuning Drift Stability Analysis

**Title:** Fine-Tuning Drift Stability Analysis  
**One-Liner:** Structural stability margins before belief-space collapse under fine-tuning, analyzing perturbation sensitivity.  
**Cluster:** B

---

#### ai.45 — Fleet Skill Propagation Stability

**Title:** Fleet Skill Propagation Stability  
**One-Liner:** Structural stability for skill updates across robotic fleets with cascade detection, analyzing propagation dynamics.  
**Cluster:** B

---

### AI-Cluster C — Runtime Control & Coherence (9 Apps)

#### ai.04 — Runtime Control Coherence ⭐ Core-3

**Title:** Runtime Control Coherence  
**One-Liner:** Diagnose and reduce incoherence between scheduler, runtime and model control loops.  
**Cluster:** C | **Related Whitepaper:** *SORT-AI: Runtime Control Coherence*

---

#### ai.17 — Fault-Recovery Collapse Prevention

**Title:** Fault-Recovery Collapse Prevention  
**One-Liner:** Analysis of instability through checkpointing, restart, replication, and proactive migration mechanisms.  
**Cluster:** C

---

#### ai.18 — Workload Placement Stability Validation

**Title:** Workload Placement Stability Validation  
**One-Liner:** Structural assessment of placement decisions independent of scheduler logic for stability verification.  
**Cluster:** C

---

#### ai.27 — Inference Pipeline Control Coherence

**Title:** Inference Pipeline Control Coherence  
**One-Liner:** Structural coherence analysis of inference pipelines including batching, caching, and serving control loops.  
**Cluster:** C

---

#### ai.32 — Training-Deployment Phase Transition Diagnostics

**Title:** Training-Deployment Phase Transition Diagnostics  
**One-Liner:** Structural analysis of objective function coherence across training-deployment boundary, detecting stability breaks at phase transitions.  
**Cluster:** C

---

#### ai.33 — Objective-Constraint Surface Divergence Analysis

**Title:** Objective-Constraint Surface Divergence Analysis  
**One-Liner:** Structural analysis of divergence between specified constraints and implicit desiderata, covering Goodhart effects and reward hacking patterns.  
**Cluster:** C

---

#### ai.38 — Value Trajectory Lock-In Analysis

**Title:** Value Trajectory Lock-In Analysis  
**One-Liner:** Structural analysis of modifiability constraints as capability increases, identifying intervention windows before lock-in.  
**Cluster:** C

---

#### ai.47 — Evaluation Context Projection Instability

**Title:** Evaluation Context Projection Instability  
**One-Liner:** Structural analysis of behavior divergence between evaluation and deployment contexts, detecting test awareness patterns.  
**Cluster:** C

---

#### ai.50 — Persona Coherence Stability Diagnostics

**Title:** Persona Coherence Stability Diagnostics  
**One-Liner:** Structural analysis of identity and character invariance under task and context perturbations, analyzing character steering stability.  
**Cluster:** C

---

### AI-Cluster D — Emergence & Agentic Systems (10 Apps)

#### ai.03 — Safety and Risk Surfaces under Projection

**Title:** Safety and Risk Surfaces under Projection  
**One-Liner:** Projection based risk surfaces for advanced AI systems, stability classes and failure modes.  
**Cluster:** D

---

#### ai.13 — Agentic System Stability ⭐ Core-3

**Title:** Agentic System Stability  
**One-Liner:** Stability control for agent workflows with retry loops, self verification, and tool calling patterns.  
**Cluster:** D | **Related Whitepaper:** *SORT-AI: Agentic System Stability*

---

#### ai.23 — Model Capacity Saturation and Collapse Risk

**Title:** Model Capacity Saturation and Collapse Risk  
**One-Liner:** Analysis of capacity saturation patterns and sudden collapse risks in large models under scaling stress.  
**Cluster:** D

---

#### ai.24 — Emergent Capability Boundary Stability

**Title:** Emergent Capability Boundary Stability  
**One-Liner:** Structural stability analysis at capability emergence boundaries, detecting phase transitions in model behavior.  
**Cluster:** D

---

#### ai.28 — Structural Failure Containment and Blast Radius Control

**Title:** Structural Failure Containment and Blast Radius Control  
**One-Liner:** Analysis of whether and how structural failures remain contained or escalate system-wide through coupling, projection, and closure mechanisms.  
**Cluster:** D

---

#### ai.31 — Instrumental Goal Convergence Diagnostics

**Title:** Instrumental Goal Convergence Diagnostics  
**One-Liner:** Structural detection of convergent instrumental sub-goals in goal-directed systems, identifying stability risks from emergent goal structures.  
**Cluster:** D

---

#### ai.36 — Multi-Agent Stability Regime Analysis

**Title:** Multi-Agent Stability Regime Analysis  
**One-Liner:** Structural analysis of stability conditions in multi-agent systems with incompatible objectives, identifying equilibrium and instability regimes.  
**Cluster:** D

---

#### ai.37 — Capability Emergence Threshold Diagnostics

**Title:** Capability Emergence Threshold Diagnostics  
**One-Liner:** Structural characterization of discontinuous capability emergence at scale thresholds, treating emergence as phase transition.  
**Cluster:** D

---

#### ai.39 — Mesa-Optimization Structural Detection

**Title:** Mesa-Optimization Structural Detection  
**One-Liner:** Structural signatures of internal optimization processes diverging from base objectives, detecting emergent optimization behavior.  
**Cluster:** D

---

#### ai.43 — Agentic Goal Projection Instability

**Title:** Agentic Goal Projection Instability  
**One-Liner:** Structural analysis of goal-projection regimes in autonomous agents, identifying exploitation risks and stability boundaries.  
**Cluster:** D

---

#### ai.48 — Adversarial Strategy Phase Transition Diagnostics

**Title:** Adversarial Strategy Phase Transition Diagnostics  
**One-Liner:** Structural detection of strategy regime shifts under adversarial or stress triggers, analyzing behavioral phase transitions.  
**Cluster:** D

---

### AI-Cluster E — Evidence & Assurance (1 App)

#### ai.30 — Structural Stability Evidence Pack for Assessments

**Title:** Structural Stability Evidence Pack for Assessments  
**One-Liner:** Standardized evidence and assurance structure for stability claims, enabling formal justification and audit-readiness without system implementation details.  
**Cluster:** E

---

## SORT-CX Domain — Complex Systems (28 Applications)

### CX-Cluster A — Coupling & Dependencies (10 Apps)

#### cx.03 — Network Function Graph Stability

**Title:** Network Function Graph Stability  
**One-Liner:** Structural stability metrics for function graphs, cascades and recovery.  
**Cluster:** A

---

#### cx.05 — Service Mesh Coupling Stability Assessment

**Title:** Service Mesh Coupling Stability Assessment  
**One-Liner:** Structural stability analysis of service mesh, east-west traffic, policy coupling, and dependency graphs in microservice landscapes.  
**Cluster:** A

---

#### cx.09 — Nonlinear Subsystem Coupling Analysis

**Title:** Nonlinear Subsystem Coupling Analysis  
**One-Liner:** Structural analysis of nonlinear couplings between subsystems in complex platforms, identifying hidden dependencies and amplification paths.  
**Cluster:** A

---

#### cx.12 — Multi-Actor Coordination Failure Dynamics

**Title:** Multi-Actor Coordination Failure Dynamics  
**One-Liner:** Structural analysis of coordination failures in multi-actor systems with misaligned incentives, analyzing race dynamics and equilibrium instabilities.  
**Cluster:** A

---

#### cx.15 — Cross-Market Coupling Stress Analysis

**Title:** Cross-Market Coupling Stress Analysis  
**One-Liner:** Structural analysis of latent coupling producing non-local amplification across markets, identifying stress transmission paths.  
**Cluster:** A

---

#### cx.16 — Clearing Infrastructure Stability Diagnostics

**Title:** Clearing Infrastructure Stability Diagnostics  
**One-Liner:** Structural stability analysis of settlement infrastructure as system tipping point, identifying infrastructure regime risks.  
**Cluster:** A

---

#### cx.24 — AI-Orchestrated Intrusion Chain Patterns

**Title:** AI-Orchestrated Intrusion Chain Patterns  
**One-Liner:** Structural pattern extraction from multi-stage attack sequences, analyzing cyber attack graph coupling.  
**Cluster:** A

---

#### cx.26 — Control Plane Monoculture Risk Analysis

**Title:** Control Plane Monoculture Risk Analysis  
**One-Liner:** Structural analysis of single-point-of-failure patterns in centralized control planes, assessing platform takeover resilience.  
**Cluster:** A

---

#### cx.28 — Control Surface Expansion Tracking

**Title:** Control Surface Expansion Tracking  
**One-Liner:** Structural monitoring of attack surface growth through adversarial adaptation, tracking jailbreak-driven expansion.  
**Cluster:** A

---

### CX-Cluster B — Temporal Adaptation & Drift (4 Apps)

#### cx.04 — Detection Graph Drift Control

**Title:** Detection Graph Drift Control  
**One-Liner:** Drift control for detection graphs, reducing false positives and mode collapse.  
**Cluster:** B

---

#### cx.06 — Autoscaling Oscillation and Phase Transition Diagnostics

**Title:** Autoscaling Oscillation and Phase Transition Diagnostics  
**One-Liner:** Structural analysis of auto-scaling, queueing, backpressure, and control loop instabilities including oscillations, thundering herd, flapping, and regime shifts.  
**Cluster:** B

---

#### cx.10 — Regime Shift Early Warning Diagnostics

**Title:** Regime Shift Early Warning Diagnostics  
**One-Liner:** Structural early detection of regime shifts in complex systems before they manifest as performance degradation or failure.  
**Cluster:** B

---

#### cx.19 — Pipeline Phase Regime Drift Detection

**Title:** Pipeline Phase Regime Drift Detection  
**One-Liner:** Structural analysis of multi-phase pipeline coupling and late-stage failures, identifying drift across pipeline phases.  
**Cluster:** B

---

### CX-Cluster C — Control & Regulation (6 Apps)

#### cx.01 — Pipeline Stability Control

**Title:** Pipeline Stability Control  
**One-Liner:** Drift and reproducibility diagnostics for distributed dataflow pipelines.  
**Cluster:** C

---

#### cx.11 — Intervention Window Identification

**Title:** Intervention Window Identification  
**One-Liner:** Structural identification of effective intervention windows in complex systems, determining when and where interventions have stabilizing effects.  
**Cluster:** C

---

#### cx.17 — Command Hierarchy Partial Failure Coherence

**Title:** Command Hierarchy Partial Failure Coherence  
**One-Liner:** Structural analysis of control coherence under partial system failure, assessing control hierarchy robustness.  
**Cluster:** C

---

#### cx.18 — Decision Loop Saturation Detection

**Title:** Decision Loop Saturation Detection  
**One-Liner:** Structural detection of decision loop time compression exceeding oversight capacity, treating time as structural dimension.  
**Cluster:** C

---

#### cx.22 — Institutional Drift Monitoring

**Title:** Institutional Drift Monitoring  
**One-Liner:** Structural drift detection between formal structure and effective function, analyzing governance effectiveness.  
**Cluster:** C

---

#### cx.25 — Human Decision Bottleneck Drift

**Title:** Human Decision Bottleneck Drift  
**One-Liner:** Structural analysis of control coherence between formal authority and effective steering, identifying oversight timing mismatch.  
**Cluster:** C

---

### CX-Cluster D — Emergence & System-of-Systems (7 Apps)

#### cx.02 — Emergent Stability under Projection

**Title:** Emergent Stability under Projection  
**One-Liner:** Detect stability islands and regime shifts under projection and aggregation.  
**Cluster:** D

---

#### cx.07 — Cascading Failure Containment for System of Systems

**Title:** Cascading Failure Containment for System of Systems  
**One-Liner:** Structural assessment of containment capability, blast radius, and propagation paths in coupled platform systems.  
**Cluster:** D

---

#### cx.13 — Feedback Loop Cascade Diagnostics

**Title:** Feedback Loop Cascade Diagnostics  
**One-Liner:** Structural detection of feedback amplification before cascade threshold, identifying flash crash and runaway patterns.  
**Cluster:** D

---

#### cx.14 — Liquidity Regime Transition Detection

**Title:** Liquidity Regime Transition Detection  
**One-Liner:** Structural detection of hidden regime transitions in coupled financial systems, identifying stability breaks before market stress.  
**Cluster:** D

---

#### cx.20 — Capacity Phase Transition Monitoring

**Title:** Capacity Phase Transition Monitoring  
**One-Liner:** Structural detection of emergent capacity collapse versus nominal reserve, identifying hidden capacity stress.  
**Cluster:** D

---

#### cx.21 — Policy-Induced System Fragility Detection

**Title:** Policy-Induced System Fragility Detection  
**One-Liner:** Structural analysis of governance interventions producing unintended instability, treating governance as structural perturbation.  
**Cluster:** D

---

#### cx.23 — Swarm Escalation Dynamics Analysis

**Title:** Swarm Escalation Dynamics Analysis  
**One-Liner:** Structural analysis of cascade amplification in autonomous swarm systems, analyzing drone swarm stability patterns.  
**Cluster:** D

---

#### cx.27 — Machine-Time Escalation Early Warning

**Title:** Machine-Time Escalation Early Warning  
**One-Liner:** Structural detection of escalation dynamics exceeding human intervention time, analyzing OODA loop compression.  
**Cluster:** D

---

### CX-Cluster E — Evidence & Assurance (1 App)

#### cx.08 — Infrastructure Auditability of Complex Control Planes

**Title:** Infrastructure Auditability of Complex Control Planes  
**One-Liner:** Evidence and auditability layer for control regimes in complex platforms, enabling formal justification without implementation details.  
**Cluster:** E

---

## SORT-QS Domain — Quantum Systems (11 Applications)

### QS-Cluster A — State Space & Coupling (4 Apps)

#### qs.01 — Noise Filtering and Operator Diagnostics

**Title:** Noise Filtering and Operator Diagnostics  
**One-Liner:** Structural noise filtering and diagnostics for operator chains and channels.  
**Cluster:** A

---

#### qs.07 — Quantum Control Stability under Classical Coupling

**Title:** Quantum Control Stability under Classical Coupling  
**One-Liner:** Structural assessment of quantum control stability under influence of classical coupling mechanisms, analyzing how classical control affects quantum coherence.  
**Cluster:** A

---

#### qs.09 — Error Correction Threshold Stability Diagnostics

**Title:** Error Correction Threshold Stability Diagnostics  
**One-Liner:** Structural analysis of stability boundaries in scaling quantum error correction, identifying threshold regime risks.  
**Cluster:** A

---

#### qs.10 — Logical Qubit Scaling Regime Certification

**Title:** Logical Qubit Scaling Regime Certification  
**One-Liner:** Structural criteria for distinguishing demonstration from platform-ready error correction, providing certification framework.  
**Cluster:** A

---

### QS-Cluster B — Temporal Adaptation (1 App)

#### qs.04 — Calibration Drift and Retuning Stability Assessment

**Title:** Calibration Drift and Retuning Stability Assessment  
**One-Liner:** Structural assessment of drift, recalibration, and retuning loops including risks that local stabilization causes global instability.  
**Cluster:** B

---

### QS-Cluster C — Control & Measurement (3 Apps)

#### qs.02 — Error Correction Diagnostics

**Title:** Error Correction Diagnostics  
**One-Liner:** Structural criteria for error correction performance and failure detection.  
**Cluster:** C

---

#### qs.03 — Hybrid Quantum Workflow Stability

**Title:** Hybrid Quantum Workflow Stability  
**One-Liner:** Stability diagnostics for hybrid quantum classical workflows and scheduling.  
**Cluster:** C

---

#### qs.11 — Measurement-Control Interaction Drift Analysis

**Title:** Measurement-Control Interaction Drift Analysis  
**One-Liner:** Structural analysis of feedback loops between observation and system state in quantum systems, analyzing observer-system coupling.  
**Cluster:** C

---

### QS-Cluster D — Emergence & Regime Shifts (2 Apps)

#### qs.05 — Error Burst Emergence and Regime Shift Diagnostics

**Title:** Error Burst Emergence and Regime Shift Diagnostics  
**One-Liner:** Structural detection of error burst regimes and sudden regime shifts in systems showing superficially stable error rates.  
**Cluster:** D

---

#### qs.08 — Structural Error Propagation Analysis

**Title:** Structural Error Propagation Analysis  
**One-Liner:** Structural analysis of error propagation and instability spreading in quantum systems, identifying paths through which local errors escalate system-wide.  
**Cluster:** D

---

### QS-Cluster E — Evidence & Assurance (1 App)

#### qs.06 — Evidence of Calibration Stability and Control Integrity

**Title:** Evidence of Calibration Stability and Control Integrity  
**One-Liner:** Evidence and assurance layer for calibration stability and control integrity over time, enabling formal justification for quantum system deployments.  
**Cluster:** E

---

## SORT-COSMO Domain — Cosmology (Non-IP, 11 Applications)

Cosmology applications serve scientific publications and are not licensed. They receive no Structural Dimensions and no cluster assignment.

| App-ID | Title | One-Liner |
|--------|-------|-----------|
| cosmo.01 | Early Galaxies | Explain high redshift massive galaxy candidates via projection stabilized structure formation. |
| cosmo.02 | Early SMBH Seeds | Reconcile early supermassive black holes via kernel controlled growth and drift regimes. |
| cosmo.03 | Hubble Drift | Model scale dependent H0 as a projection drift signature across datasets. |
| cosmo.04 | CMB Anomalies | Treat large scale anomalies as projection level structural artifacts under kernel regimes. |
| cosmo.05 | Dark Baryon Oscillator | Coupled sector surrogate model for tension patterns via drift coupled response. |
| cosmo.06 | Intergalactic Bridges | Filament baryons and bridges as stable projections of operator adjacency under kernel coupling. |
| cosmo.07 | Dark Flow Drift Signature Analysis | Bulk flow coherence as projection signature, analyzing cosmological drift patterns at large scales. |
| cosmo.08 | CMB Signal Separation Diagnostics | Foreground separation structural criteria for CMB analysis, providing diagnostic methodology. |
| cosmo.09 | Quantum-Classical Transition Projection | Projection-based treatment of quantum-to-classical transition mechanism with diagnostic framing. |
| cosmo.10 | Metric Consistency Analysis | Structural analysis of metric projection dependence under acceleration, diagnostic framework for g_tt behavior. |
| cosmo.11 | Reionization Dynamics Modeling | Early ionization via projection dynamics, structural diagnostic for reionization history. |

---

## Summary Statistics

### Technical Domains (AI, CX, QS)

| Domain | Apps | A | B | C | D | E |
|--------|------|---|---|---|---|---|
| AI | 52 | 26 | 6 | 9 | 10 | 1 |
| CX | 28 | 10 | 4 | 6 | 7 | 1 |
| QS | 11 | 4 | 1 | 3 | 2 | 1 |
| **Sum Tech** | **91** | **40** | **11** | **18** | **19** | **3** |

### Meta-Domain (Sovereign)

| Domain | Apps | A | C | E |
|--------|------|---|---|---|
| SOV | 5 | 2 | 1 | 2 |

**Note:** Sovereign statistics are not comparable to technical domains. Clusters B and D are excluded in Sovereign.

### Non-IP Domain

| Domain | Apps |
|--------|------|
| COSMO | 11 |

### Total

| Category | Apps |
|----------|------|
| Technical Domains | 91 |
| Meta-Domain | 5 |
| Non-IP | 11 |
| **Total** | **107** |

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

**Formal changes:**
- JSON reference filename corrected to `catalog.public.json`
- Domain-ID mapping table added
- Sovereign formally clarified as Meta-Domain with selective cluster scope (A, C, E)
- sov.01 and sov.02 reassigned from "Meta-Level" to concrete clusters (A and E respectively)
- Statistics tables separated by technical domains, meta-domain, and non-IP
- Rule made explicit: Application ID and Title are primary, V1–V4 are supplementary

**No content changes to applications.**
