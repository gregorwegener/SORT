#!/usr/bin/env python3
"""
04_layer2_plot.py – visualize Layer II structural kernel for SORT MOCK v3

Generates three figures from M_layer2.npy:
  1) Central slice heatmap of the real-space kernel
  2) 1D radial profile of the kernel
  3) Histogram of kernel values

All outputs are written into results/.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_kernel():
    path = RESULTS_DIR / "M_layer2.npy"
    if not path.exists():
        raise FileNotFoundError(f"Structural matrix not found: {path}")
    return np.load(path)

def main():
    kappa = load_kernel()
    N = kappa.shape[0]

    # --- Figure 1: central slice heatmap ---
    mid = N // 2
    slice_xy = kappa[:, :, mid]

    fig1 = plt.figure()
    plt.imshow(slice_xy, origin="lower")
    plt.colorbar(label="kernel value")
    plt.title("Layer II kernel – central slice")
    plt.tight_layout()
    fig1_path = RESULTS_DIR / "article0_fig1_kernel_slice.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # --- Figure 2: radial profile ---
    x = np.arange(N) - mid
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    r_flat = r.ravel()
    v_flat = slice_xy.ravel()

    r_max = r_flat.max()
    n_bins = 50
    bins = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    prof = np.zeros_like(r_centers)
    counts = np.zeros_like(r_centers)

    inds = np.digitize(r_flat, bins) - 1
    for idx, val in zip(inds, v_flat):
        if 0 <= idx < len(prof):
            prof[idx] += val
            counts[idx] += 1

    counts[counts == 0] = 1.0
    prof /= counts

    fig2 = plt.figure()
    plt.plot(r_centers, prof)
    plt.xlabel("radius (grid units)")
    plt.ylabel("mean kernel value")
    plt.title("Layer II kernel – radial profile")
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "article0_fig2_kernel_radial_profile.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # --- Figure 3: histogram ---
    fig3 = plt.figure()
    plt.hist(kappa.ravel(), bins=80)
    plt.xlabel("kernel value")
    plt.ylabel("count")
    plt.title("Layer II kernel – value distribution")
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "article0_fig3_kernel_histogram.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    summary = {
        "shape": list(kappa.shape),
        "min": float(kappa.min()),
        "max": float(kappa.max()),
        "mean": float(kappa.mean()),
        "std": float(kappa.std()),
        "figures": {
            "slice": str(fig1_path.name),
            "radial_profile": str(fig2_path.name),
            "histogram": str(fig3_path.name),
        },
    }
    with (RESULTS_DIR / "article0_kernel_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Layer II kernel plots completed.")
    print(f"  slice figure   : {fig1_path}")
    print(f"  radial profile : {fig2_path}")
    print(f"  histogram      : {fig3_path}")

if __name__ == "__main__":
    main()
