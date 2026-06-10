"""SHA-256 hash manifest generation for SORT Version 7 Phase 7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from report_writer import write_text


SELF_REFERENCE_MARKER = "self-referential-manifest"


def resolve_paths(config_arg: str | Path) -> tuple[Path, Path, Path]:
    candidate = Path(config_arg)
    if candidate.exists():
        config_path = candidate.resolve()
    else:
        cwd_candidate = Path.cwd() / "phase_7_artifact_freeze" / candidate
        if cwd_candidate.exists():
            config_path = cwd_candidate.resolve()
        else:
            raise FileNotFoundError(f"Phase 7 config not found: {config_arg}")
    phase_root = config_path.parents[1]
    validation_root = phase_root.parent
    return config_path, phase_root, validation_root


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_inventory_files(root: Path, inventory_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    hash_rows = []
    for row in sorted(inventory_rows, key=lambda item: item["relative_path"]):
        rel = row["relative_path"]
        if row.get("included_in_zip") != "true" or row.get("sha256") == SELF_REFERENCE_MARKER:
            continue
        path = root / rel
        if not path.exists():
            hash_rows.append({"sha256": "ERROR", "size_bytes": "missing", "relative_path": rel})
            continue
        hash_rows.append(
            {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "relative_path": rel,
            }
        )
    return hash_rows


def write_hashes_txt(path: str | Path, hash_rows: list[dict[str, Any]], self_reference_paths: list[str] | None = None) -> None:
    lines = ["# SORT Version 7 workstation validation SHA-256 manifest"]
    for row in hash_rows:
        lines.append(f"{row['sha256']}  {row['size_bytes']}  {row['relative_path']}")
    if self_reference_paths:
        lines.append("")
        lines.append("# Self-reference policy")
        for rel in sorted(self_reference_paths):
            lines.append(f"# {rel} included in ZIP; final stable self-hash is not attempted.")
    write_text(path, "\n".join(lines) + "\n")


def read_inventory(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path, phase_root, validation_root = resolve_paths(args.config)
    config = load_config(config_path)
    inventory_path = phase_root / config["outputs"]["artifact_inventory"]
    rows = read_inventory(inventory_path)
    self_refs = [row["relative_path"] for row in rows if row.get("sha256") == SELF_REFERENCE_MARKER]
    hash_rows = hash_inventory_files(validation_root, rows)
    write_hashes_txt(phase_root / config["outputs"]["hashes"], hash_rows, self_refs)
    print(f"hash_count={len(hash_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
