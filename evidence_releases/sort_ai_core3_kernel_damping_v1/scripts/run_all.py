from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from kernel_damping import SIGMA0, compute_metric


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "data" / "core3_metrics.csv"


def classify_cv(cv: float) -> str:
    if cv <= 0.15:
        return "coherent"
    if cv <= 0.25:
        return "acceptable mixed / overlap"
    return "unstable / outlier-dominated"


def load_metrics() -> list[dict]:
    with METRICS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_metrics()
    for row in rows:
        risk_baseline = float(row["risk_baseline"])
        risk_comparison = float(row["risk_comparison"])
        result = compute_metric(risk_baseline, risk_comparison, sigma0=SIGMA0)
        row["kappa_calculated"] = result.kappa
        row["xi_calculated"] = result.xi
        row["kappa_abs_error_vs_reported"] = abs(result.kappa - float(row["kappa_reported"]))
        row["xi_abs_error_vs_reported"] = abs(result.xi - float(row["xi_reported"]))

    scenario_groups: dict[str, list[dict]] = {}
    for row in rows:
        scenario_groups.setdefault(row["scenario_id"], []).append(row)

    scenarios = []
    for scenario_id, group in sorted(scenario_groups.items()):
        xis = [float(r["xi_reported"]) for r in group]
        mean = statistics.mean(xis)
        std = statistics.stdev(xis) if len(xis) > 1 else 0.0
        cv = std / mean if mean else 0.0
        scenarios.append({
            "scenario_id": scenario_id,
            "application_id": group[0]["application_id"],
            "scenario_type": group[0]["scenario_type"],
            "metric_count": len(group),
            "xi_mean": round(mean, 2),
            "xi_std_sample": round(std, 2),
            "cv": round(cv, 3),
            "classification": classify_cv(cv),
        })

    all_xis = [float(r["xi_reported"]) for r in rows]
    summary = {
        "sigma0": SIGMA0,
        "metric_count": len(rows),
        "scenario_count": len(scenarios),
        "application_count": len({r["application_id"] for r in rows}),
        "overall_xi_mean": round(statistics.mean(all_xis), 2),
        "overall_xi_std_sample": round(statistics.stdev(all_xis), 2),
        "overall_cv": round(statistics.stdev(all_xis) / statistics.mean(all_xis), 3),
        "scenarios": scenarios,
        "max_kappa_abs_error_vs_reported": max(float(r["kappa_abs_error_vs_reported"]) for r in rows),
        "max_xi_abs_error_vs_reported": max(float(r["xi_abs_error_vs_reported"]) for r in rows),
    }

    out_dir = ROOT / "outputs_generated"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "core3_summary.generated.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
