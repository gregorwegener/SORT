# Methodology

This release implements a minimal kernel-damping consistency protocol for the SORT-AI Core-3 applications.

## Hierarchy

```text
Domain -> Cluster -> Application -> Scenario Class -> Metric Set -> Kernel-Damping Test
```

## Core-3 axes

| Application | Axis |
|---|---|
| AI.01 | physical / interconnect coupling |
| AI.04 | logical / runtime-control coupling |
| AI.13 | semantic / agentic coupling |

## Risk transformation

All variables are converted into risk variables, where lower is better:

| Transform | Rule |
|---|---|
| identity | `r = x` |
| risk | `r = x` |
| health | `r = 1 - x` |
| multiplier | `r = x - 1` |

## Kernel-damping equations

$$
\kappa_i=\frac{r_i^{(1)}}{r_i^{(0)}}
$$

$$
\kappa_{\sigma_0}(\xi_i)=\exp\left[-\frac{(\sigma_0\xi_i)^2}{2}\right]
$$

$$
\xi_i=\frac{\sqrt{-2\ln(\kappa_i)}}{\sigma_0}
$$

with

$$
\sigma_0=0.00190643.
$$

## Classification

| CV range | Classification |
|---:|---|
| `CV <= 0.15` | coherent |
| `0.15 < CV <= 0.25` | acceptable mixed / overlap |
| `CV > 0.25` | unstable / outlier-dominated |

The coefficient of variation is computed per scenario from reported metric-level `xi` values using the sample standard deviation.