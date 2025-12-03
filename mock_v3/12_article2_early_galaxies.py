#!/usr/bin/env python3
"""
12_article2_early_galaxies.py – Article 2 module for SORT MOCK v3

PATCH MOCKv3-final: Enhancement strukturell an σ₀ und Kernel gebunden
- Nicht mehr ad-hoc: enhancement = 1 + 2σ₀(1+z)
- Jetzt kernelgebunden: enhancement = 1 + |η(k_eff(z), σ₀)|
  wobei η = exp(-(σ₀ L_H k)²/2) - 1

Uses the calibrated sigma0 to compute a structural enhancement of the
high-z galaxy abundance and produces three figures.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Physikalische Konstanten
L_HUBBLE = 4285.0  # Mpc

def load_layer2_metrics():
    path = RESULTS_DIR / "layer2_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Layer II metrics not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def k_effective(z):
    """
    PATCH MOCKv3-final: Effektive Wellenzahl als Funktion von z
    
    Auf galaktischen Skalen skaliert k_eff mit (1+z):
    - Bei z=10 entspricht k_eff ~ 0.1 Mpc⁻¹ typischen Galaxienskalen
    - k_eff wächst mit z (kleinere physikalische Skalen bei höherem z)
    """
    return 0.1 * (1.0 + z) / 10.0  # Mpc⁻¹

def eta_projection(k, sigma0):
    """
    Projektionskorrektur η(k; σ₀)
    
    η = exp(-(σ₀ L_H k)²/2) - 1
    
    Negativ für alle k > 0 (Dämpfung).
    """
    x = sigma0 * L_HUBBLE * k
    return np.exp(-0.5 * x**2) - 1.0

def enhancement_kernel(z, sigma0):
    """
    PATCH MOCKv3-final: Kernelgebundenes Enhancement
    
    Die Projektion modifiziert die effektive Varianz des Dichtefeldes.
    Für kleine |η| ist das Enhancement der Halo-Anzahl ~ 1 + |η|.
    
    Args:
        z: Rotverschiebung
        sigma0: Kalibrierter dimensionsloser Parameter
        
    Returns:
        Enhancement-Faktor ≥ 1
    """
    k_eff = k_effective(z)
    eta = eta_projection(k_eff, sigma0)
    # η ist negativ, daher |η| für positives Enhancement
    return 1.0 + abs(eta)

def compute_number_density(sigma0, z_array=None, M_array=None):
    """
    Berechnet Galaxien-Anzahldichte mit kernelgebundenem Enhancement.
    """
    if z_array is None:
        z_array = np.array([6, 8, 10, 12, 14], dtype=float)
    if M_array is None:
        M_array = np.logspace(8, 12, 40)

    n_M_z = np.zeros((len(M_array), len(z_array)))
    enhancement = np.zeros_like(z_array)
    k_eff_arr = np.zeros_like(z_array)
    eta_arr = np.zeros_like(z_array)

    for j, z in enumerate(z_array):
        # PATCH: Kernelgebundenes Enhancement
        k_eff_arr[j] = k_effective(z)
        eta_arr[j] = eta_projection(k_eff_arr[j], sigma0)
        enhancement[j] = enhancement_kernel(z, sigma0)
        
        for i, M in enumerate(M_array):
            # Einfache Press-Schechter-artige Massenfunktion (Platzhalter)
            n_LCDM = 1e-8 * (M / 1e10) ** -1.5 * np.exp(-z / 10.0)
            n_M_z[i, j] = n_LCDM * enhancement[j]

    return M_array, z_array, n_M_z, enhancement, k_eff_arr, eta_arr

def main():
    metrics = load_layer2_metrics()
    sigma0 = float(metrics["calibration"]["sigma0"])

    M_array, z_array, n_M_z, enhancement, k_eff_arr, eta_arr = compute_number_density(sigma0)

    # Figure 1: Mass functions
    fig1 = plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_array)))
    for j, z in enumerate(z_array):
        plt.loglog(M_array, n_M_z[:, j], color=colors[j], label=f"z={z:.0f}")
    plt.xlabel(r"$M\,[M_\odot]$")
    plt.ylabel(r"$n(M,z)$ [arb.]")
    plt.title("Early galaxy mass functions (SORT enhanced)")
    plt.legend()
    plt.tight_layout()
    fig1_path = RESULTS_DIR / "article2_fig1_mass_functions.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # Figure 2: Enhancement vs z
    fig2, ax1 = plt.subplots()
    ax1.plot(z_array, enhancement, 'b-o', label='Enhancement')
    ax1.set_xlabel("z")
    ax1.set_ylabel("Enhancement factor", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(z_array, -eta_arr, 'r--s', label=r'$|\eta(k_{\mathrm{eff}})|$')
    ax2.set_ylabel(r"$|\eta|$", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title("SORT enhancement of high-z abundance")
    fig2.tight_layout()
    fig2_path = RESULTS_DIR / "article2_fig2_enhancement_vs_z.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # Figure 3: Number density map
    fig3 = plt.figure()
    data = np.log10(n_M_z.T + 1e-40)
    im = plt.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=(np.log10(M_array[0]), np.log10(M_array[-1]), z_array[0], z_array[-1]),
        cmap='viridis'
    )
    plt.colorbar(im, label=r"log$_{10}$ n(M,z)")
    plt.xlabel(r"log$_{10} M\,[M_\odot]$")
    plt.ylabel("z")
    plt.title("Number density map (SORT enhanced)")
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "article2_fig3_number_density_map.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    out = {
        "sigma0": sigma0,
        "M_array": M_array.tolist(),
        "z_array": z_array.tolist(),
        "n_M_z": n_M_z.tolist(),
        "enhancement": enhancement.tolist(),
        "k_effective": k_eff_arr.tolist(),
        "eta_values": eta_arr.tolist(),
        "method": {
            "formula": "enhancement = 1 + |η(k_eff(z), σ₀)|",
            "k_eff": "k_eff(z) = 0.1 * (1+z) / 10 Mpc⁻¹",
            "eta": "η = exp(-(σ₀ L_H k)²/2) - 1",
            "note": "PATCH MOCKv3-final: Enhancement strukturell an Kernel gebunden"
        },
        "figures": {
            "mass_functions": str(fig1_path.name),
            "enhancement_vs_z": str(fig2_path.name),
            "density_map": str(fig3_path.name),
        },
    }
    out_path = RESULTS_DIR / "article2_early_galaxies.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Article 2 (early galaxies) completed.")
    print(f"  σ₀ = {sigma0:.8f}")
    print(f"  Enhancement-Methode: kernelgebunden (PATCH)")
    print("  Enhancement-Faktoren:")
    for j, z in enumerate(z_array):
        print(f"    z={z:.0f}: {enhancement[j]:.6f} (k_eff={k_eff_arr[j]:.4f}, η={eta_arr[j]:.6f})")
    print(f"  JSON   : {out_path}")
    print(f"  figure1: {fig1_path}")
    print(f"  figure2: {fig2_path}")
    print(f"  figure3: {fig3_path}")

if __name__ == "__main__":
    main()
