#!/usr/bin/env python3
"""
01_layer1.py – Layer I algebraic diagnostics for SORT mock_v3

Reads 05_config.yaml and 06_operators.json, performs basic consistency checks
(operator count, light balance, seed) and writes layer1_metrics.json and
layer1_table2.csv into ./results.
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

BASE_DIR = Path(__file__).resolve().parent

def load_config():
    cfg_path = BASE_DIR / "05_config.yaml"
    if yaml is None:
        raise RuntimeError("PyYAML is required to read 05_config.yaml (pip install pyyaml).")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_operators():
    ops_path = BASE_DIR / "06_operators.json"
    with ops_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    cfg = load_config()
    ops = load_operators()

    results_dir = BASE_DIR / cfg["outputs"]["results_dir"]
    results_dir.mkdir(exist_ok=True)

    # Light-balance check
    c_weights = cfg["operators"]["c_weights"]
    sum_c = float(sum(c_weights))
    epsilon_lb = abs(sum_c)

    n_ops_declared = cfg["operators"]["n_operators"]
    n_ops_file = len(ops["operators"])

    metrics = {
        "n_operators_declared": n_ops_declared,
        "n_operators_file": n_ops_file,
        "epsilon_light_balance": epsilon_lb,
        "sum_c_weights": sum_c,
        "seed": cfg["seed"],
        "status": "ok" if (epsilon_lb < 1e-14 and n_ops_declared == n_ops_file) else "warning"
    }

    # JSON metrics
    metrics_path = BASE_DIR / cfg["outputs"]["layer1_metrics"]
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Simple CSV "Table 2" for operators
    table_path = BASE_DIR / cfg["outputs"]["layer1_table2"]
    with table_path.open("w", encoding="utf-8") as f:
        f.write("id,label,weight,role\n")
        for op in ops["operators"]:
            f.write(f"{op['id']},{op['label']},{op['weight']},{op['role']}\n")

    print("Layer I completed.")
    print(f"  metrics : {metrics_path}")
    print(f"  table   : {table_path}")

if __name__ == "__main__":
    sys.exit(main())
