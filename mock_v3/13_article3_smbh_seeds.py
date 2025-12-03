#!/usr/bin/env python3
"""
13_article3_smbh_seeds.py – Article 3 module for SORT MOCK v3

PATCH MOCKv3-final: Physikalische Parameterwahl für realistische Saatmassen
- eta_0 = 1e-6 (statt 0.01) - seltene Seed-Kanäle
- delta_proj = 0.3 (statt 10.0) - typische Überdichte in Projektionsregionen

Ergibt M_seed ~ 10^7–10^8 M☉ statt 10^12–10^13 M☉
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

G_CONST = 4.302e-9
L_HUBBLE = 4285.0
RHO_CRIT_0 = 2.775e11
OMEGA_M = 0.315
RHO_M_0 = OMEGA_M * RHO_CRIT_0

def load_layer2_metrics():
    path = RESULTS_DIR / "layer2_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Layer II metrics not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def efficiency_factor(z, M=None):
    """
    PATCH MOCKv3-final: η₀ = 1e-6 für realistische Seed-Effizienz
    
    Physikalische Motivation: 
    - Nur ein kleiner Bruchteil der Halos bildet Direct-Collapse-Black-Holes
    - η ~ 10^-6 entspricht ~1 DCBH pro 10^6 geeigneten Halos
    """
    eta_0 = 1e-6  # PATCH: von 0.01 auf 1e-6
    f_collapse = min(1.0, (1.0 + float(z)) / 10.0)
    if M is not None and M > 1e6:
        f_feedback = (1e6 / M) ** 0.5
    else:
        f_feedback = 1.0
    return eta_0 * f_collapse * f_feedback

def sigma_comoving(sigma0, z):
    return sigma0 * L_HUBBLE / (1.0 + float(z))

def critical_potential():
    """Kritisches Potential für atomare Kühlung: ~10 (km/s)²"""
    return 10.0

def typical_potential_depth(z, sigma0, delta_proj=0.3):
    """
    PATCH MOCKv3-final: δ_proj = 0.3 für typische Überdichte
    
    Physikalische Motivation:
    - In Projektionsregionen ist δρ/ρ ~ 0.3 eine realistische Überdichte
    - Statt δ_proj = 10 (extreme Peaks) nutzen wir typische Werte
    """
    rho_m_z = RHO_M_0 * (1.0 + float(z)) ** 3
    sigma = sigma_comoving(sigma0, z)
    return G_CONST * rho_m_z * delta_proj * sigma ** 2

def seed_mass_from_potential(phi_abs, sigma, eta_BH):
    prefactor = eta_BH * (4.0 * np.pi / 3.0) / G_CONST
    return prefactor * phi_abs * sigma

def compute_seed_curves(sigma0, z_array=None):
    if z_array is None:
        z_array = np.array([7.0, 10.0, 15.0, 20.0])

    M_min = np.zeros_like(z_array)
    M_typ = np.zeros_like(z_array)
    M_max = np.zeros_like(z_array)
    eta_arr = np.zeros_like(z_array)
    sigma_arr = np.zeros_like(z_array)

    for i, z in enumerate(z_array):
        sigma = sigma_comoving(sigma0, z)
        sigma_arr[i] = sigma
        eta = efficiency_factor(z)
        eta_arr[i] = eta

        phi_crit = critical_potential()
        phi_typ = typical_potential_depth(z, sigma0)
        phi_max = 3.0 * phi_typ  # Seltene 3σ-Peaks

        M_min[i] = seed_mass_from_potential(phi_crit, sigma, eta)
        M_typ[i] = seed_mass_from_potential(phi_typ, sigma, eta)
        M_max[i] = seed_mass_from_potential(phi_max, sigma, eta)

    return z_array, sigma_arr, eta_arr, M_min, M_typ, M_max

def main():
    metrics = load_layer2_metrics()
    sigma0 = float(metrics["calibration"]["sigma0"])

    z_array, sigma_arr, eta_arr, M_min, M_typ, M_max = compute_seed_curves(sigma0)

    # Figure 1: Seed masses vs redshift
    fig1 = plt.figure()
    plt.semilogy(z_array, M_min, marker="o", label=r"$M_{\mathrm{min}}$")
    plt.semilogy(z_array, M_typ, marker="s", label=r"$M_{\mathrm{typ}}$")
    plt.semilogy(z_array, M_max, marker="^", label=r"$M_{\mathrm{max}}$")
    plt.xlabel("z")
    plt.ylabel(r"$M_{\mathrm{seed}}\,[M_\odot]$")
    plt.title("SMBH seed masses vs redshift")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1_path = RESULTS_DIR / "article3_fig1_mseed_vs_z.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # Figure 2: Efficiency vs redshift
    fig2 = plt.figure()
    plt.semilogy(z_array, eta_arr, marker="o")
    plt.xlabel("z")
    plt.ylabel(r"$\eta_{\mathrm{BH}}$")
    plt.title("Formation efficiency vs redshift")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "article3_fig2_efficiency_vs_z.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # Figure 3: Mass histogram at z≈10
    z_ref = 10.0
    idx = np.argmin(np.abs(z_array - z_ref))
    M0 = M_typ[idx]

    rng = np.random.default_rng(117666)
    logM0 = np.log10(M0 + 1e-30)
    logM_samples = rng.normal(loc=logM0, scale=0.4, size=2000)
    M_samples = 10.0 ** logM_samples

    fig3 = plt.figure()
    plt.hist(M_samples, bins=60, edgecolor='black', alpha=0.7)
    plt.xscale("log")
    plt.xlabel(r"$M_{\mathrm{seed}}\,[M_\odot]$")
    plt.ylabel("count")
    plt.title(f"Mock SMBH seed distribution at z≈{z_ref:.0f}")
    plt.axvline(M0, color='red', linestyle='--', label=f'$M_{{typ}}={M0:.2e}$')
    plt.legend()
    plt.tight_layout()
    fig3_path = RESULTS_DIR / "article3_fig3_mass_histogram_z10.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    out = {
        "sigma0": sigma0,
        "z_array": z_array.tolist(),
        "sigma_comoving": sigma_arr.tolist(),
        "eta_BH": eta_arr.tolist(),
        "M_seed_min": M_min.tolist(),
        "M_seed_typical": M_typ.tolist(),
        "M_seed_max": M_max.tolist(),
        "parameters": {
            "eta_0": 1e-6,
            "delta_proj": 0.3,
            "note": "PATCH MOCKv3-final: Realistic parameter values for DCBH seeds"
        },
        "figures": {
            "mseed_vs_z": str(fig1_path.name),
            "efficiency_vs_z": str(fig2_path.name),
            "mass_histogram_z10": str(fig3_path.name),
        },
    }
    out_path = RESULTS_DIR / "article3_smbh_seeds.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Article 3 (SMBH seeds) completed.")
    print(f"  σ₀ = {sigma0:.8f}")
    print(f"  η₀ = 1e-6 (PATCH)")
    print(f"  δ_proj = 0.3 (PATCH)")
    print(f"  M_typ(z=10) = {M_typ[idx]:.2e} M☉")
    print(f"  JSON   : {out_path}")
    print(f"  figure1: {fig1_path}")
    print(f"  figure2: {fig2_path}")
    print(f"  figure3: {fig3_path}")

if __name__ == "__main__":
    main()
