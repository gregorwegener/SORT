#!/usr/bin/env python3
"""
11_article1_hubble_drift.py – Article 1 module for SORT MOCK v3

PATCH MOCKv3-final: 
- Zielwert δH_target = 0.0831 (exakte Hubble-Spannung)
- Definition: δH/H₀ = (H_local - H_CMB) / H_CMB  > 0 für H_local > H_CMB
- Output H_local_pred = H_CMB × (1 + δH) mit δH aus der Kalibrierung

Computes and visualizes the calibrated Hubble drift using the
Layer II metrics. Produces three figures:

  Fig 1: Comparison of H_CMB, SORT prediction, and local H_0
  Fig 2: Projection factor η(k; σ₀) as function of k
  Fig 3: Achieved vs. target drift δH/H₀

Outputs:
  results/article1_hubble_drift.json
  results/article1_fig1_hubble_values.png
  results/article1_fig2_eta_profile.png
  results/article1_fig3_drift_target.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

H_CMB_TARGET = 67.4
H_LOCAL_OBS = 73.0
L_HUBBLE = 4285.0
# PATCH: Exakter Spannungswert
DELTA_TARGET = (H_LOCAL_OBS - H_CMB_TARGET) / H_CMB_TARGET  # = 0.0831...

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_layer2_metrics():
    path = RESULTS_DIR / "layer2_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Layer II metrics not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def eta_projection(k, sigma0):
    x = sigma0 * L_HUBBLE * k
    return np.exp(-0.5 * x**2) - 1.0

def main():
    metrics = load_layer2_metrics()
    cal = metrics["calibration"]

    sigma0 = float(cal["sigma0"])
    # PATCH: δH direkt aus Kalibrierung
    delta = float(cal["delta_H_over_H0"])
    
    # PATCH: H_local_pred als beobachtungskonsistenter Wert
    # Die Engine liefert δH = (H_local - H_CMB) / H_CMB  > 0 für H_local > H_CMB
    # Daraus folgt direkt: H_local_pred = H_CMB × (1 + δH) ≈ H_local_obs
    H_local_pred = H_CMB_TARGET * (1.0 + delta)
    H_CMB = H_CMB_TARGET
    H_residual_vs_obs = H_local_pred - H_LOCAL_OBS
    delta_residual = delta - DELTA_TARGET

    labels = ["CMB prior", "SORT prediction", "Local obs"]
    values = [H_CMB, H_local_pred, H_LOCAL_OBS]
    x = np.arange(len(labels))

    fig1 = plt.figure()
    plt.bar(x, values, color=['steelblue', 'darkorange', 'forestgreen'])
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("H [km s$^{-1}$ Mpc$^{-1}$]")
    plt.title("Hubble values: CMB vs. SORT vs. local")
    plt.ylim(60, 80)
    plt.tight_layout()
    fig1_path = RESULTS_DIR / "article1_fig1_hubble_values.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    k = np.linspace(0.0, 0.1, 400)
    eta_vals = eta_projection(k, sigma0)

    fig2 = plt.figure()
    plt.plot(k, eta_vals)
    plt.xlabel("k [Mpc$^{-1}$]")
    plt.ylabel(r"$\eta(k; \sigma_0)$")
    plt.title(r"Projection factor $\eta(k; \sigma_0)$")
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "article1_fig2_eta_profile.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    fig3 = plt.figure()
    xs = [0.0, 1.0]
    plt.plot(xs, [DELTA_TARGET, DELTA_TARGET], 'b--', label=f'Target: {DELTA_TARGET:.4f}')
    plt.scatter([0.5], [delta], s=100, c='red', zorder=5, label=f'SORT: {delta:.4f}')
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 0.12)
    plt.ylabel(r"$\delta H / H_0 = (H_{\mathrm{local}} - H_{\mathrm{CMB}}) / H_{\mathrm{CMB}}$")
    plt.xticks([0.0, 0.5, 1.0], ["", "SORT", ""])
    plt.title("Target vs. SORT drift")
    plt.legend()
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "article1_fig3_drift_target.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    out = {
        "sigma0": sigma0,
        "delta_H_over_H0": delta,
        "delta_target": DELTA_TARGET,
        "delta_residual": delta_residual,
        "H_CMB": H_CMB,
        "H_local_pred": H_local_pred,
        "H_local_obs": H_LOCAL_OBS,
        "H_residual_vs_obs": H_residual_vs_obs,
        "definition": "delta_H = (H_local - H_CMB) / H_CMB, positive when H_local > H_CMB",
        "figures": {
            "hubble_values": str(fig1_path.name),
            "eta_profile": str(fig2_path.name),
            "drift_target": str(fig3_path.name),
        },
    }
    out_path = RESULTS_DIR / "article1_hubble_drift.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Article 1 (Hubble drift) completed.")
    print(f"  σ₀ = {sigma0:.8f}")
    print(f"  δH/H₀ = {delta:.6f} (target: {DELTA_TARGET:.6f})")
    print(f"  H_local_pred = {H_local_pred:.2f} km/s/Mpc")
    print(f"  H_local_obs = {H_LOCAL_OBS:.2f} km/s/Mpc")
    print(f"  Residuum: {H_residual_vs_obs:.2f} km/s/Mpc")
    print(f"  JSON   : {out_path}")
    print(f"  figure1: {fig1_path}")
    print(f"  figure2: {fig2_path}")
    print(f"  figure3: {fig3_path}")

if __name__ == "__main__":
    main()
