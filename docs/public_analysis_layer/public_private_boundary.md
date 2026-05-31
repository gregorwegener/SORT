# Public / Proprietary Boundary

This document defines the disclosure boundary for the SORT Public Analysis Layer.

## Public Layer

The public repository may document:

| Public Component | Description |
|---|---|
| Level-0 positioning | SORT as structural assessment framework |
| MOCK v4 | frozen structural reference architecture |
| Application catalog | public domain, cluster, and application identities |
| V1-V4 grammar | public diagnostic dimensions |
| Assessment-case tuple | public analysis-layer case structure |
| Scenario classes | core, boundary, and overlap regimes |
| Metric sets | declared public or synthetic analysis-layer indicators |
| Evidence interfaces | reproducible evidence protocol attachment points |
| Kernel-damping protocol | reproducible analysis-layer calculations where explicitly released |

## Proprietary or Gated Layer

The public repository does not disclose:

| Non-Public Component | Reason |
|---|---|
| customer-specific operator chains | execution-layer assessment logic |
| vendor telemetry mappings | customer-specific integration value |
| metric weighting functions | assessment-engine calibration |
| scoring functions | operational decision logic |
| production thresholds | deployment-specific calibration |
| intervention playbooks | consulting and implementation layer |
| runtime integration logic | product or customer-specific engineering |
| SWORD execution logic | future operator-resolved execution layer |

## Public Claim Boundary

The repository supports this claim:

```text
SORT provides a public structural assessment grammar and frozen reference architecture through which analysis-layer evidence protocols can be formed and reproduced.
```

The repository does not support these claims:

```text
SORT has implemented a complete production assessment engine.
SORT has optimized a vendor runtime.
SORT has executed customer telemetry mapping.
SORT has released SWORD execution logic.
```

## Governance Principle

The public analysis layer is intended to make SORT scientifically readable and methodologically auditable without disclosing the customer-specific or execution-layer machinery required for operational deployment.

\[
\text{public grammar}
\neq
\text{complete assessment engine}.
\]

## Future Disclosure

Future execution-layer work may be disclosed through controlled version gates, validation gates, peer-review gates, or risk gates. Such future disclosure does not modify the frozen MOCK v4 architecture unless the structural contracts themselves are changed.