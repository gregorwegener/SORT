# Scenario, Metric, and Regime Hierarchy

This document defines the public hierarchy used by the SORT Public Analysis Layer.

## Hierarchy

The public hierarchy is:

$$
\text{Domain}
\rightarrow
\text{Cluster}
\rightarrow
\text{Application}
\rightarrow
\text{Scenario Class}
\rightarrow
\text{Metric Set}
\rightarrow
\text{Regime Classification}
\rightarrow
\text{Evidence Interface}
$$

## Application

An Application is a recurrent structural problem form, not an isolated use case or product feature.

$$
A_j = \text{recurrent structural problem form}
$$

## Scenario Class

A Scenario Class is a typed manifestation inside an Application.

$$
C_{j\ell} \in \mathcal{S}(A_j)
$$

where $\mathcal{S}(A_j)$ denotes the scenario-class space of application $A_j$.

## Regime Classes

Scenario classes may be assigned to public regime classes:

$$
\rho(C_{j\ell}) \in \{\mathrm{core},\mathrm{boundary},\mathrm{overlap}\}
$$

| Regime | Meaning |
|---|---|
| core | central manifestation of the application |
| boundary | edge condition inside or near the application envelope |
| overlap | mixed regime between applications or structural problem forms |

## Application Regime Space

The public regime space of an application can be represented as:

$$
\mathcal{S}(A_j)
=
\mathcal{S}^{\mathrm{core}}_j
\cup
\mathcal{S}^{\mathrm{boundary}}_j
\cup
\mathcal{S}^{\mathrm{overlap}}_j
$$

This expression is classificatory. It does not imply that all regime spaces are strictly disjoint in the operational execution layer.

## Metric Set

A Metric Set is a declared family of public or derived indicators associated with a scenario class:

$$
M_{j\ell}
=
\{m_1,m_2,\ldots,m_n\}_{j\ell}
$$

The corresponding public metric vector is:

$$
\mathbf{x}_{j\ell}(S_D)
=
\left(
 m_1(S_D),
 m_2(S_D),
 \ldots,
 m_n(S_D)
\right)
$$

## Risk Transformation

For evidence-compatible cases, metric values may be transformed into risk variables:

$$
r_i=T_i(m_i(S_D))
$$

Public risk transformations may include direct risk, health-to-risk, or overhead-to-risk mappings. Customer-specific telemetry mappings and production calibrations are not included here.

## Evidence Interface

The evidence interface is the point where a scenario class and metric set become eligible for a reproducible evidence protocol.

$$
(C_{j\ell},M_{j\ell},\rho_{j\ell})
\rightarrow
E_{j\ell}
$$

For kernel-damping evidence releases, $E_{j\ell}$ contains declared risk-transition pairs and reproducible calculation rules.