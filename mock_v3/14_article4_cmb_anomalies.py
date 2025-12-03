#!/usr/bin/env python3
"""
14_article4_cmb_anomalies.py – Article 4 module for SORT MOCK v3
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    yaml = None

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_config():
    cfg_path = BASE_DIR / "05_config.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required for 05_config.yaml (pip install pyyaml).")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_kernel():
    path = RESULTS_DIR / "M_layer2.npy"
    if not path.exists():
        raise FileNotFoundError(f"Structural matrix not found: {path}")
    return np.load(path)

def build_power_spectrum(kernel, dx, n_bins=60):
    N = kernel.shape[0]
    kappa_k = np.fft.fftn(kernel)
    power_k = np.abs(kappa_k) ** 2

    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k_norm = np.sqrt(kx**2 + ky**2 + kz**2)

    k_flat = k_norm.ravel()
    p_flat = power_k.ravel()

    k_max = k_flat.max()
    bins = np.linspace(0.0, k_max, n_bins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    P = np.zeros_like(k_centers)
    counts = np.zeros_like(k_centers)

    inds = np.digitize(k_flat, bins) - 1
    for idx, val in zip(inds, p_flat):
        if 0 <= idx < len(P):
            P[idx] += val
            counts[idx] += 1

    counts[counts == 0] = 1.0
    P /= counts

    return k_centers, P, power_k, kx, ky, kz

def main():
    cfg = load_config()
    kernel = load_kernel()

    dx = float(cfg["lattice"]["dx"])

    k_centers, P_k, power_k, kx, ky, kz = build_power_spectrum(kernel, dx)

    fig1 = plt.figure()
    plt.loglog(k_centers[1:], P_k[1:])
    plt.xlabel("k [grid units]")
    plt.ylabel("P(k) [arb.]")
    plt.title("Kernel-based power spectrum")
    plt.tight_layout()
    fig1_path = RESULTS_DIR / "article4_fig1_power_spectrum.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    k_cut = k_centers.max() * 0.2
    mask = (k_centers > 0.0) & (k_centers < k_cut)

    fig2 = plt.figure()
    plt.plot(k_centers[mask], P_k[mask])
    plt.xlabel("k [grid units]")
    plt.ylabel("P(k) [arb.]")
    plt.title("Low-k regime of the spectrum")
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "article4_fig2_lowk_zoom.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    mask_plus = kx > 0.0
    mask_minus = kx < 0.0
    P_plus = float(power_k[mask_plus].mean())
    P_minus = float(power_k[mask_minus].mean())
    asym = (P_plus - P_minus) / (P_plus + P_minus)

    fig3 = plt.figure()
    plt.bar([0, 1], [P_plus, P_minus])
    plt.xticks([0, 1], ["kx>0", "kx<0"])
    plt.ylabel("mean power [arb.]")
    plt.title("Hemispherical power asymmetry")
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "article4_fig3_hemisphere_power.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    out = {
        "k_centers": k_centers.tolist(),
        "P_k": P_k.tolist(),
        "P_plus": P_plus,
        "P_minus": P_minus,
        "hemispherical_asymmetry": asym,
        "figures": {
            "power_spectrum": str(fig1_path.name),
            "lowk_zoom": str(fig2_path.name),
            "hemisphere_power": str(fig3_path.name),
        },
    }
    out_path = RESULTS_DIR / "article4_cmb_anomalies.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Article 4 (CMB anomalies) completed.")
    print(f"  JSON   : {out_path}")
    print(f"  figure1: {fig1_path}")
    print(f"  figure2: {fig2_path}")
    print(f"  figure3: {fig3_path}")

if __name__ == "__main__":
    main()
