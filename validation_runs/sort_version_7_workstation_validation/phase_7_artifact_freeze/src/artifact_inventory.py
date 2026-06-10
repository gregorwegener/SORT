"""Artifact inventory builder for SORT Version 7 Phase 7."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from report_writer import write_csv


FIELDNAMES = ["relative_path", "phase", "file_type", "size_bytes", "included_in_zip", "sha256"]

SELF_REFERENCE_OUTPUTS = {
    "phase_7_artifact_freeze/outputs/artifact_inventory.csv",
    "phase_7_artifact_freeze/outputs/hashes.txt",
    "phase_7_artifact_freeze/outputs/repro_manifest.json",
    "phase_7_artifact_freeze/outputs/freeze_report.json",
}


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


def iter_artifact_files(root: Path, exclude_patterns: list[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _is_excluded(path, root, exclude_patterns):
            continue
        yield path


def classify_phase(relative_path: str) -> str:
    first = relative_path.split("/", 1)[0]
    if first.startswith("phase_"):
        return first
    if first in {"manifests", "artifacts"}:
        return first
    if first == "README.md":
        return "validation_root"
    return "validation_root"


def classify_file_type(relative_path: str) -> str:
    path = Path(relative_path)
    if path.name == "README.md":
        return "readme"
    if "/config/" in relative_path or path.name.endswith("_setup.json"):
        return "config"
    if "/src/" in relative_path or path.suffix == ".py":
        return "source"
    if "/outputs/" in relative_path:
        return "output"
    if "/input/" in relative_path:
        return "input"
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return "manifest"
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() == ".txt":
        return "text"
    return path.suffix.lower().lstrip(".") or "file"


def build_inventory(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_path: dict[str, dict[str, Any]] = {}
    for path in iter_artifact_files(root, config["exclude_patterns"]):
        rel = path.relative_to(root).as_posix()
        rows_by_path[rel] = _inventory_row(root, rel, path)
    for rel in SELF_REFERENCE_OUTPUTS:
        rows_by_path[rel] = {
            "relative_path": rel,
            "phase": classify_phase(rel),
            "file_type": classify_file_type(rel),
            "size_bytes": "self-referential",
            "included_in_zip": "true",
            "sha256": "self-referential-manifest",
        }
    return [rows_by_path[key] for key in sorted(rows_by_path)]


def write_inventory_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, FIELDNAMES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_row(root: Path, rel: str, path: Path) -> dict[str, Any]:
    if rel in SELF_REFERENCE_OUTPUTS:
        size: int | str = "self-referential"
        file_hash = "self-referential-manifest"
    else:
        size = path.stat().st_size
        file_hash = sha256_file(path)
    return {
        "relative_path": rel,
        "phase": classify_phase(rel),
        "file_type": classify_file_type(rel),
        "size_bytes": size,
        "included_in_zip": "true",
        "sha256": file_hash,
    }


def _is_excluded(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    parts = path.relative_to(root).parts
    for pattern in patterns:
        if pattern in parts or fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel, pattern):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path, phase_root, validation_root = resolve_paths(args.config)
    config = load_config(config_path)
    rows = build_inventory(validation_root, config)
    write_inventory_csv(phase_root / config["outputs"]["artifact_inventory"], rows)
    print(f"artifact_inventory_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
