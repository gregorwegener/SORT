# SORT Public Analysis Layer

This directory documents the public analysis-layer methodology used by SORT and SORT-AI.

The Public Analysis Layer explains how a structural observation becomes an assessment case before it is connected to a reproducible evidence protocol or future execution layer.

It defines the public structural assessment grammar:

\[
S_D
\rightarrow
(V_1,V_2,V_3,V_4)
\rightarrow
A_j
\rightarrow
C_{j\ell}
\rightarrow
M_{j\ell}
\rightarrow
\rho_{j\ell}
\rightarrow
E_{j\ell}.
\]

where:

| Symbol | Meaning |
|---|---|
| \(S_D\) | structured system state in domain \(D\) |
| \(V_1,V_2,V_3,V_4\) | public diagnostic dimensions |
| \(A_j\) | application identity |
| \(C_{j\ell}\) | scenario class within application \(A_j\) |
| \(M_{j\ell}\) | metric set associated with the scenario class |
| \(\rho_{j\ell}\) | regime classification |
| \(E_{j\ell}\) | evidence interface |

## Scope

This layer is methodological and analysis-oriented. It is not an execution engine, customer assessment engine, benchmark harness, vendor observability stack, runtime optimizer, or production integration layer.

It documents the public grammar required to move from observed structural phenomena to evidence-compatible assessment cases.

## Included Documents

| File | Purpose |
|---|---|
| `structural_assessment_protocol.md` | Public assessment-case protocol and scope |
| `v1_v4_diagnostic_grammar.md` | V1-V4 diagnostic dimensions |
| `assessment_case_tuple.md` | Formal assessment-case tuple |
| `scenario_metric_regime_hierarchy.md` | Application, scenario, metric, and regime hierarchy |
| `public_private_boundary.md` | Public/proprietary disclosure boundary |

## Public Boundary

This directory publicly documents:

- the V1-V4 diagnostic grammar;
- the assessment-case tuple;
- the application-to-scenario-to-metric hierarchy;
- core, boundary, and overlap regime classes;
- the evidence-interface position of kernel-damping protocols;
- public non-claims and disclosure boundaries.

This directory does not disclose:

- customer-specific operator chains;
- vendor telemetry mappings;
- metric weighting functions;
- scoring functions;
- production thresholds;
- intervention playbooks;
- runtime integration logic;
- SWORD execution logic.

## Relation to MOCK v4

MOCK v4 remains the frozen structural reference architecture. The Public Analysis Layer operates above MOCK v4 and does not modify it.

\[
\text{MOCK v4}
\rightarrow
\text{Public Analysis Layer}
\rightarrow
\text{Evidence Protocols}
\rightarrow
\text{Future Execution Layers}
\]

## Relation to Evidence Releases

Evidence releases, such as the SORT-AI Core-3 Kernel-Damping Evidence Release, attach to public assessment cases through the evidence interface \(E_{j\ell}\).

The Public Analysis Layer explains how a case becomes structurally assessable. Evidence releases show how selected declared risk-transition cases can be reproduced mathematically.