from __future__ import annotations

import math
from dataclasses import dataclass

SIGMA0 = 0.00190643


@dataclass(frozen=True)
class MetricResult:
    kappa: float
    xi: float


def compute_kappa(risk_baseline: float, risk_comparison: float) -> float:
    if risk_baseline <= 0:
        raise ValueError("risk_baseline must be positive")
    kappa = risk_comparison / risk_baseline
    if not 0 < kappa < 1:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    return kappa


def compute_xi(kappa: float, sigma0: float = SIGMA0) -> float:
    if sigma0 <= 0:
        raise ValueError("sigma0 must be positive")
    if not 0 < kappa < 1:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    return math.sqrt(-2.0 * math.log(kappa)) / sigma0


def compute_metric(risk_baseline: float, risk_comparison: float, sigma0: float = SIGMA0) -> MetricResult:
    kappa = compute_kappa(risk_baseline, risk_comparison)
    xi = compute_xi(kappa, sigma0=sigma0)
    return MetricResult(kappa=kappa, xi=xi)
