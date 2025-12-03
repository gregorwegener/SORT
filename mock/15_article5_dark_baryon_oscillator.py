#!/usr/bin/env python3
"""
15_article5_dark_baryon_oscillator.py – Article 5 module for SORT MOCK v3
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

def build_power_spectrum(kernel, dx, n_bins=80):
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

    return k_centers, P

def main():
    cfg = load_config()
    kernel = load_kernel()
    dx = float(cfg["lattice"]["dx"])

    k_centers, P_k = build_power_spectrum(kernel, dx)

    mask = k_centers > 0.0
    k_fit = k_centers[mask]
    P_fit = P_k[mask]

    logk = np.log(k_fit)
    logP = np.log(P_fit + 1e-40)

    coeffs = np.polyfit(logk, logP, deg=3)
    logP_smooth = np.polyval(coeffs, logk)
    P_smooth = np.exp(logP_smooth)
    wiggles = P_fit / P_smooth - 1.0

    fig1 = plt.figure()
    plt.loglog(k_centers[1:], P_k[1:])
    plt.xlabel("k [grid units]")
    plt.ylabel("P(k) [arb.]")
    plt.title("Kernel-based power spectrum")
    plt.tight_layout()
    fig1_path = RESULTS_DIR / "article5_fig1_power_spectrum.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.loglog(k_fit, P_fit)
    plt.loglog(k_fit, P_smooth)
    plt.xlabel("k [grid units]")
    plt.ylabel("P(k) [arb.]")
    plt.title("Smooth background vs spectrum")
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "article5_fig2_smooth_vs_data.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(k_fit, wiggles)
    plt.xlabel("k [grid units]")
    plt.ylabel("relative wiggle")
    plt.title("Dark baryon oscillator pattern")
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "article5_fig3_wiggle_residuals.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    out = {
        "k_centers": k_centers.tolist(),
        "P_k": P_k.tolist(),
        "k_fit": k_fit.tolist(),
        "P_fit": P_fit.tolist(),
        "P_smooth": P_smooth.tolist(),
        "wiggles": wiggles.tolist(),
        "figures": {
            "power_spectrum": str(fig1_path.name),
            "smooth_vs_data": str(fig2_path.name),
            "wiggle_residuals": str(fig3_path.name),
        },
    }
    out_path = RESULTS_DIR / "article5_dark_baryon_oscillator.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Article 5 (dark baryon oscillator) completed.")
    print(f"  JSON   : {out_path}")
    print(f"  figure1: {fig1_path}")
    print(f"  figure2: {fig2_path}")
    print(f"  figure3: {fig3_path}")

if __name__ == "__main__":
    main()
