# Reproducibility Report

The declared evidence bundle contains:

| Item | Count |
|---|---:|
| Applications | 3 |
| Scenarios | 20 |
| Metrics | 104 |
| Canonical `sigma0` | 0.00190643 |

All metric rows were checked by recomputing:

\[
\kappa_i=\frac{r_i^{(1)}}{r_i^{(0)}}
\]

and

\[
\xi_i=\frac{\sqrt{-2\ln(\kappa_i)}}{\sigma_0}.
\]

Maximum absolute deviation against reported rounded values:

| Quantity | Max absolute deviation |
|---|---:|
| `kappa` | 0.000189 |
| `xi` | 0.427778 |

The deviations are rounding-level differences induced by the source tables.
