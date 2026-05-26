from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_parameters"]["sigma0"] == 0.00190643
    assert manifest["canonical_parameters"]["n_operators"] == 22
    assert manifest["counts"]["applications"] == 3
    assert manifest["counts"]["scenarios"] == 20
    assert manifest["counts"]["metrics"] == 104
    for rel_path in manifest["entrypoints"].values():
        assert (ROOT / rel_path).exists(), rel_path
    print("manifest validation passed")


if __name__ == "__main__":
    main()
