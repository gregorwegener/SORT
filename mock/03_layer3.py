#!/usr/bin/env python3
"""
03_layer3.py – Layer III semi-spectral evolution skeleton for SORT mock_v3

This script reads the calibrated configuration and Layer II kernel, constructs
a random test field, projects it, evolves a simple toy "time" series and
writes aggregate diagnostics suitable for Appendix J-style plots.
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np

from mockv3_engine import MockV3Config, MOCKv3NumericalEngine

BASE_DIR = Path(__file__).resolve().parent

def load_config():
    cfg_path = BASE_DIR / "05_config.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required to read 05_config.yaml (pip install pyyaml).")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg_raw = load_config()

    results_dir = BASE_DIR / cfg_raw["outputs"]["results_dir"]
    results_dir.mkdir(exist_ok=True)

    # Load Layer II kernel
    M_path = BASE_DIR / cfg_raw["outputs"]["M_layer2"]
    if not M_path.exists():
        raise RuntimeError("M_layer2.npy not found. Run 02_layer2.py first.")
    kappa_real = np.load(M_path)

    # Rebuild config and engine
    conf = MockV3Config(
        N=cfg_raw["lattice"]["N"],
        L=cfg_raw["lattice"]["L"],
        dx=cfg_raw["lattice"]["dx"],
        sigma0=None,
        beta=cfg_raw["projection"]["beta"],
        c_weights=cfg_raw["operators"]["c_weights"]
    )
    engine = MOCKv3NumericalEngine(conf)

    # Simple toy "evolution": apply projection repeatedly and track energy-like quantity
    np.random.seed(cfg_raw["seed"])
    psi = (np.random.normal(0, 1, (conf.N,)*3) +
           1j * np.random.normal(0, 1, (conf.N,)*3))

    # Embed kappa_real as convolution kernel via FFT from engine
    # (we reuse engine.construct_kernel just to get the FFT plans and grids)
    # Here we only need the FFT wrappers, not the calibrated sigma0
    # Embed kappa_real as convolution kernel via FFT from engine
    kernel_fft = engine.fft_forward(kappa_real)

    # Numerische Stabilisierung: Kernel im k-Raum auf max |K| = 1 normieren
    k_max = np.max(np.abs(kernel_fft))
    if not np.isfinite(k_max) or k_max == 0:
        k_max = 1.0
    kernel_fft = kernel_fft / k_max

    def project(field):
        f_fft = engine.fft_forward(field)
        proj_fft = f_fft * kernel_fft
        return engine.fft_backward(proj_fft)

    n_steps = 16
    energies = []


    for t in range(n_steps):
        psi = project(psi)

        # numerisch stabile Dichteberechnung
        psi = np.clip(psi, -1e150, 1e150)                 # verhindert Overflow in psi
        rho = np.abs(psi)**2
        rho = np.clip(rho, 0.0, 1e300)                   # verhindert Overflow in rho

        E = float(rho.mean())
        energies.append({"step": t, "E": E})


    # Write diagnostics
    energy_path = BASE_DIR / cfg_raw["outputs"]["layer3_energy_series"]
    with energy_path.open("w", encoding="utf-8") as f:
        f.write("step,E\n")
        for row in energies:
            f.write(f"{row['step']},{row['E']:.10e}\n")

    metrics_path = BASE_DIR / cfg_raw["outputs"]["layer3_metrics"]
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "n_steps": n_steps,
            "E_min": min(e["E"] for e in energies),
            "E_max": max(e["E"] for e in energies),
            "E_final": energies[-1]["E"]
        }, f, indent=2)

    print("Layer III completed.")
    print(f"  energy : {energy_path}")
    print(f"  metrics: {metrics_path}")

if __name__ == "__main__":
    sys.exit(main())
