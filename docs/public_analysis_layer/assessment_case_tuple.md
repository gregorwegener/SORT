# Assessment Case Tuple

This document defines the public SORT assessment-case tuple used by the Public Analysis Layer.

## Formal Object

$$
\mathcal{A}_{\mathrm{case}} =
\left(
S_D,
V_1,V_2,V_3,V_4,
A_j,
C_{j\ell},
M_{j\ell},
\rho_{j\ell},
E_{j\ell}
\right)
$$

The tuple is a public analysis-layer object. It is not an executable production assessment object.

## Components

| Component | Meaning |
|---|---|
| $S_D$ | structured system state in domain $D$ |
| $V_1$ | observed structural phenomenon |
| $V_2$ | structural cause or coupling surface |
| $V_3$ | effect, projection, risk, or drift space |
| $V_4$ | decision, evidence, or governance surface |
| $A_j$ | application identity |
| $C_{j\ell}$ | scenario class inside application $A_j$ |
| $M_{j\ell}$ | metric set for scenario class $C_{j\ell}$ |
| $\rho_{j\ell}$ | regime classification |
| $E_{j\ell}$ | evidence interface |

## Public Use

The tuple is used to make clear that SORT-AI Applications are not isolated use cases. They are recurrent structural problem forms that can be instantiated through scenario classes and metric sets.

## Non-Executable Status

The public tuple does not include the following execution-layer objects:

$$
\hat{J}^{\mathrm{exec}},\quad
\mathbf{t},\quad
\mathbf{w},\quad
\mathbf{s},\quad
\Theta,\quad
\mathcal{I}
$$

These objects would correspond to execution-layer or customer-specific assessment elements such as concrete operator chains, telemetry abstractions, weights, scores, thresholds, and intervention classes. They are not part of the public repository at this stage.

## Relation to Future Execution Layers

A future execution-layer object may extend the public case:

$$
\mathcal{A}^{\mathrm{exec}}_{\mathrm{case}}
=
\left(
\mathcal{A}_{\mathrm{case}},
\hat{J}^{\mathrm{exec}},
\mathbf{t},
\mathbf{w},
\mathbf{s},
\Theta,
\mathcal{I},
\mathcal{T}
\right)
$$

This future object is not implemented here. The present document only defines the public analysis-layer case.