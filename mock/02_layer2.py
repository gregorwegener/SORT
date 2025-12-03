#!/usr/bin/env python3
"""
02_layer2.py – Layer II structural matrix construction for SORT mock_v3

Uses the corrected mockv3_engine to run the full pipeline up to kernel
construction and validation; writes layer2_metrics.json and M_layer2.npy.
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np
import datetime 

def json_serializer(obj):
    # NumPy-Arrays und Skalar-Typen
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    # Zeitstempel: numpy.datetime64 oder datetime-Objekte
    if isinstance(obj, (np.datetime64, datetime.datetime, datetime.date)):
        return obj.isoformat()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


from mockv3_engine import MockV3Config, MOCKv3NumericalEngine, ValidationSuite

BASE_DIR = Path(__file__).resolve().parent

def load_config():
    cfg_path = BASE_DIR / "05_config.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required to read 05_config.yaml (pip install pyyaml).")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg_raw = load_config()

    # Build MockV3Config from YAML
    c_weights = cfg_raw["operators"]["c_weights"]
    conf = MockV3Config(
        N=cfg_raw["lattice"]["N"],
        L=cfg_raw["lattice"]["L"],
        dx=cfg_raw["lattice"]["dx"],
        sigma0=None,
        beta=cfg_raw["projection"]["beta"],
        c_weights=c_weights
    )

    engine = MOCKv3NumericalEngine(conf)
    validator = ValidationSuite(engine)

    # Step 1: sigma0 calibration (once)
    calibration = engine.calibrator.calibrate_sigma0_exact()
    if not calibration["converged"]:
        raise RuntimeError(f"sigma0 calibration failed: {calibration['final_residual']}")

    sigma0 = calibration["sigma0"]
    conf.sigma0 = sigma0

    # Step 2: kernel construction
    kernel = engine.construct_kernel(sigma0, conf.beta)

    # Step 3: full validation using existing calibration
    validation_report = validator.run_full_validation(kernel, calibration)

    # Serialize main kernel matrix as Layer II object (M_layer2)
    # Here we simply dump the real-space kernel as the structural matrix
    M = kernel.kappa_real

    results_dir = BASE_DIR / cfg_raw["outputs"]["results_dir"]
    results_dir.mkdir(exist_ok=True)

    metrics_path = BASE_DIR / cfg_raw["outputs"]["layer2_metrics"]
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "calibration": {
                "sigma0": calibration["sigma0"],
                "H_bare": calibration["H_bare"],
                "delta_H_over_H0": calibration["delta_H_over_H0"],
                "iterations": calibration["iterations"],
                "converged": calibration["converged"]
            },
            "validation": validation_report
        }, f, indent=2, default=json_serializer, ensure_ascii=False)

    M_path = BASE_DIR / cfg_raw["outputs"]["M_layer2"]
    np.save(M_path, M)

    print("Layer II completed.")
    print(f"  metrics : {metrics_path}")
    print(f"  M_layer2: {M_path}")

if __name__ == "__main__":
    sys.exit(main())
