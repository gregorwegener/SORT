#!/usr/bin/env python3
"""
16_article6_intergalactic_bridges.py – Article 6 module for SORT MOCK v3
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

def build_correlation(kernel):
    Kk = np.fft.fftn(kernel)
    Pk = np.abs(Kk) ** 2
    xi = np.fft.ifftn(Pk).real
    return xi

def radial_profile(field, L):
    N = field.shape[0]
    mid = N // 2

    x = (np.arange(N) - mid) * (L / N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)

    r_flat = r.ravel()
    f_flat = field.ravel()

    r_max = r_flat.max()
    n_bins = 60
    bins = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    prof = np.zeros_like(r_centers)
    counts = np.zeros_like(r_centers)

    inds = np.digitize(r_flat, bins) - 1
    for idx, val in zip(inds, f_flat):
        if 0 <= idx < len(prof):
            prof[idx] += val
            counts[idx] += 1

    counts[counts == 0] = 1.0
    prof /= counts

    return r_centers, prof

def main():
    cfg = load_config()
    kernel = load_kernel()

    L = float(cfg["lattice"]["L"])

    xi = build_correlation(kernel)
    r_centers, xi_r = radial_profile(xi, L)

    norm = max(abs(xi_r.max()), abs(xi_r.min()), 1e-30)
    xi_r_norm = xi_r / norm

    fig1 = plt.figure()
    plt.plot(r_centers, xi_r_norm)
    plt.xlabel("r [box units]")
    plt.ylabel(r"$\xi(r)$ (normalised)")
    plt.title("Intergalactic bridge correlation function")
    plt.tight_layout()
    fig1_path = RESULTS_DIR / "article6_fig1_correlation_radial.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    r_cut = r_centers.max() * 0.7
    mask = r_centers > r_cut

    fig2 = plt.figure()
    plt.plot(r_centers[mask], xi_r_norm[mask])
    plt.xlabel("r [box units]")
    plt.ylabel(r"$\xi(r)$ (normalised)")
    plt.title("Large-r tail of ξ(r)")
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "article6_fig2_tail_zoom.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    N = xi.shape[0]
    mid = N // 2
    slice_xy = xi[:, :, mid]

    fig3 = plt.figure()
    plt.imshow(slice_xy, origin="lower")
    plt.colorbar(label=r"$\xi(x,y,z_\mathrm{mid})$")
    plt.title("Correlation slice through the box")
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "article6_fig3_correlation_slice.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    out = {
        "r_centers": r_centers.tolist(),
        "xi_r": xi_r.tolist(),
        "figures": {
            "correlation_radial": str(fig1_path.name),
            "tail_zoom": str(fig2_path.name),
            "correlation_slice": str(fig3_path.name),
        },
    }
    out_path = RESULTS_DIR / "article6_intergalactic_bridges.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Article 6 (intergalactic bridges) completed.")
    print(f"  JSON   : {out_path}")
    print(f"  figure1: {fig1_path}")
    print(f"  figure2: {fig2_path}")
    print(f"  figure3: {fig3_path}")

if __name__ == "__main__":
    main()
