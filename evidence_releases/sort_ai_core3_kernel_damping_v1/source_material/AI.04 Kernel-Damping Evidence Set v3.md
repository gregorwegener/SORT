# AI.04 Kernel-Damping Evidence Set v3

**Application:** AI.04 — Runtime Control Coherence  
**Domain:** SORT-AI  
**Cluster:** C — Control  
**Structural axis:** logical / runtime-control coupling  
**Evidence level:** analysis-layer structural consistency evidence  
**Canonical scale parameter:** \(\sigma_0=0.00190643\)

## 1. Claim

The narrow claim of this evidence set is:

\[
\text{AI.04 admits a kernel-damping representation for runtime-control risk modes, including boundary and overlap regimes.}
\]

This evidence set does not claim production deployment, empirical benchmarking, vendor-specific measurement, runtime optimization, or execution by MOCK v4. The scenario values are synthetic but structurally grounded analysis-layer inputs for a reproducible kernel-damping calculation.

## 2. Structural Scope

AI.04 describes runtime-control incoherence as a recurrent structural problem in large-scale AI systems. The relevant risk modes arise when locally correct schedulers, orchestrators, runtime engines, retry mechanisms, admission logic, and policy layers compose into globally incoherent behavior.

The application is vendor-agnostic and diagnostic. It does not define a scheduler, runtime product, control-plane implementation, or operational remediation mechanism.

## 3. Scenario Classes

| Scenario ID | Scenario class | Type | Structural purpose |
|---|---|---|---|
| AI.04.C1 | Cross-Layer Control Conflict | Core | Conflict between scheduler, orchestrator, runtime, and policy layer |
| AI.04.C2 | Retry Amplification | Core | Local retry logic generates global attempt and cost amplification |
| AI.04.C3 | Control Oscillation | Core | Control loops amplify one another over time |
| AI.04.B1 | SLA Boundary Occupation | Boundary | Runtime operates near SLA, capacity, or margin boundary |
| AI.04.O1 | Control plus Infrastructure Coupling | Overlap | Transition regime \(AI.04\cap AI.01\) |
| AI.04.O2 | Control plus Agentic Execution | Overlap | Transition regime \(AI.04\cap AI.13\) |

The AI.04 scenario family is:

\[
\mathcal{S}_{AI.04}=\{C1,C2,C3,B1,O1,O2\}.
\]

## 4. Kernel-Damping Model

Each raw metric is transformed into a risk variable \(r\), where lower values represent lower structural risk.

For direct risk metrics:

\[
r=x.
\]

For multiplicative overhead metrics:

\[
r=x-1.
\]

For positive health or accuracy metrics:

\[
r=1-x.
\]

For each metric \(i\), the risk transition is represented as:

\[
r_i^{(1)}=\kappa_i r_i^{(0)}.
\]

The empirical damping ratio is:

\[
\kappa_i=\frac{r_i^{(1)}}{r_i^{(0)}}.
\]

The Gaussian SORT kernel is:

\[
\kappa_{\sigma_0}(\xi_i)=\exp\left[-\frac{(\sigma_0\xi_i)^2}{2}\right].
\]

Solving for the implied structural mode gives:

\[
\xi_i=\frac{\sqrt{-2\ln(\kappa_i)}}{\sigma_0}.
\]

The scenario-level statistics are:

\[
\bar{\xi}_j=\frac{1}{n}\sum_{i=1}^{n}\xi_{j,i},
\]

\[
s_{\xi,j}=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(\xi_{j,i}-\bar{\xi}_j\right)^2},
\]

\[
CV_j=\frac{s_{\xi,j}}{\bar{\xi}_j}.
\]

## 5. Scenario Summary

| Scenario | \(\bar{\xi}\) | \(s_\xi\) | \(CV\) | Interpretation |
|---|---:|---:|---:|---|
| AI.04.C1 | 768.40 | 28.64 | 0.037 | coherent core |
| AI.04.C2 | 952.00 | 36.50 | 0.038 | coherent core |
| AI.04.C3 | 823.00 | 24.90 | 0.030 | coherent core |
| AI.04.B1 | 1082.00 | 92.64 | 0.086 | coherent boundary |
| AI.04.O1 | 973.33 | 158.32 | 0.163 | acceptable mixed / overlap |
| AI.04.O2 | 1024.17 | 195.05 | 0.190 | acceptable mixed / overlap |

## 6. Aggregated Results

The core scenario set is:

\[
\mathcal{C}_{AI.04}=\{C1,C2,C3\}.
\]

The core mean is:

\[
\bar{\xi}_{AI.04,\mathrm{core}}=\frac{768.40+952.00+823.00}{3}=847.80.
\]

The pooled core statistics are:

\[
\bar{\xi}_{AI.04,\mathrm{core,pooled}}=847.80,
\]

\[
s_{\xi,AI.04,\mathrm{core,pooled}}=84.51,
\]

\[
CV_{AI.04,\mathrm{core,pooled}}=0.100.
\]

The boundary and overlap set is:

\[
\mathcal{B}_{AI.04}\cup\mathcal{O}_{AI.04}=\{B1,O1,O2\}.
\]

Its aggregate statistics are:

\[
\bar{\xi}_{AI.04,\mathrm{boundary/overlap}}=1023.24,
\]

\[
s_{\xi,AI.04,\mathrm{boundary/overlap}}=154.53,
\]

\[
CV_{AI.04,\mathrm{boundary/overlap}}=0.151.
\]

The full AI.04 family has:

\[
\bar{\xi}_{AI.04,\mathrm{all}}=941.00,
\]

\[
s_{\xi,AI.04,\mathrm{all}}=153.17,
\]

\[
CV_{AI.04,\mathrm{all}}=0.163.
\]

## 7. Interpretation

AI.04.C1, AI.04.C2, and AI.04.C3 form the runtime-control core. Their low scenario-level coefficients of variation indicate that the metric sets cluster within coherent structural-mode zones.

AI.04.B1 has a higher mean structural mode but remains internally coherent. It is therefore interpreted as a boundary mode within AI.04 rather than as a misclassified scenario.

AI.04.O1 is the transition regime toward AI.01:

\[
AI.04.O1\approx AI.04_{\mathrm{control}}\oplus AI.01_{\mathrm{interconnect}}.
\]

AI.04.O2 is the transition regime toward AI.13:

\[
AI.04.O2\approx AI.04_{\mathrm{control}}\oplus AI.13_{\mathrm{agentic}}.
\]

The higher dispersion in O1 and O2 is expected because overlap scenarios combine structurally adjacent application modes. This is an overlap signal rather than an incoherence signal.

## 8. Evidence Statement

For all declared AI.04 metrics, the comparison risk satisfies:

\[
0<r_i^{(1)}<r_i^{(0)}.
\]

Therefore:

\[
0<\kappa_i=\frac{r_i^{(1)}}{r_i^{(0)}}<1.
\]

For every \(\kappa_i\), the Gaussian kernel admits a unique positive implied structural mode:

\[
\xi_i=\frac{\sqrt{-2\ln(\kappa_i)}}{\sigma_0}.
\]

Thus:

\[
r_i^{(1)}=\kappa_{\sigma_0}(\xi_i)r_i^{(0)}
\]

with

\[
\kappa_{\sigma_0}(\xi_i)=\exp\left[-\frac{(\sigma_0\xi_i)^2}{2}\right]
\]

and

\[
\sigma_0=0.00190643.
\]

The complete metric-level source data are provided in `data/ai04/scenario_metrics.json` and `data/core3_metrics.csv`.