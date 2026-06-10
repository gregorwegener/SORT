"""Create the frozen SORT Version 7 workstation validation package."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from artifact_inventory import build_inventory, write_inventory_csv
from hash_manifest import hash_inventory_files, read_inventory, sha256_file, write_hashes_txt
from prefreeze_audit import build_audit_report, load_phase_7_setup
from report_writer import write_json, write_text


SELF_REFERENCE_PATHS = [
    "phase_7_artifact_freeze/outputs/artifact_inventory.csv",
    "phase_7_artifact_freeze/outputs/hashes.txt",
    "phase_7_artifact_freeze/outputs/repro_manifest.json",
    "phase_7_artifact_freeze/outputs/freeze_report.json",
]


def resolve_paths(config_arg: str | Path) -> tuple[Path, Path, Path, Path]:
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
    repo_root = validation_root.parent.parent
    return config_path, phase_root, validation_root, repo_root


def create_zip_package(root: Path, inventory_rows: list[dict[str, str]], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in sorted(inventory_rows, key=lambda item: item["relative_path"]):
            if row.get("included_in_zip") != "true":
                continue
            rel = row["relative_path"]
            source = root / rel
            if not source.exists() or source.resolve() == target.resolve():
                continue
            info = zipfile.ZipInfo(rel, date_time=(2026, 6, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())


def hash_zip_package(zip_path: str | Path) -> dict[str, Any]:
    target = Path(zip_path)
    return {
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def write_freeze_report(path: str | Path, report: dict[str, Any]) -> None:
    write_json(path, report)


def run_all(config_path: Path, phase_root: Path, validation_root: Path, repo_root: Path) -> dict[str, Any]:
    config = load_phase_7_setup(config_path)
    outputs = config["outputs"]
    (phase_root / "outputs").mkdir(parents=True, exist_ok=True)

    audit_report = build_audit_report(config_path, phase_root, validation_root)
    write_json(phase_root / outputs["prefreeze_audit_report"], audit_report)
    if audit_report["audit_status"] != "pass":
        raise RuntimeError("Pre-freeze audit failed; ZIP creation stopped.")

    inventory_rows = build_inventory(validation_root, config)
    write_inventory_csv(phase_root / outputs["artifact_inventory"], inventory_rows)

    inventory_rows = read_inventory(phase_root / outputs["artifact_inventory"])
    hash_rows = hash_inventory_files(validation_root, inventory_rows)
    write_hashes_txt(phase_root / outputs["hashes"], hash_rows, SELF_REFERENCE_PATHS)

    repro_manifest = _build_repro_manifest(config, repo_root, validation_root, audit_report)
    write_json(phase_root / outputs["repro_manifest"], repro_manifest)

    pending_report = _build_freeze_report(
        config=config,
        package_hash={"sha256": "pending-package-hash", "size_bytes": None},
        phase_6_status=_phase_6_status(audit_report),
        warnings=audit_report["warnings"],
        gate_7_passed=False,
    )
    write_json(phase_root / outputs["freeze_report"], pending_report)

    zip_path = phase_root / outputs["zip_package"]
    inventory_rows = read_inventory(phase_root / outputs["artifact_inventory"])
    create_zip_package(validation_root, inventory_rows, zip_path)
    package_hash = hash_zip_package(zip_path)
    _append_zip_hash(phase_root / outputs["hashes"], zip_path.relative_to(phase_root).as_posix(), package_hash)

    final_report = _build_freeze_report(
        config=config,
        package_hash=package_hash,
        phase_6_status=_phase_6_status(audit_report),
        warnings=audit_report["warnings"],
        gate_7_passed=True,
    )
    write_freeze_report(phase_root / outputs["freeze_report"], final_report)
    return final_report


def _build_repro_manifest(
    config: dict[str, Any],
    repo_root: Path,
    validation_root: Path,
    audit_report: dict[str, Any],
) -> dict[str, Any]:
    git_state = _git_state(repo_root)
    seed = _load_seed(validation_root)
    phase_6_included = audit_report["phase_status"].get("phase_6_workstation_scaling") == "included"
    machine = {"source": "../phase_0_setup/env_spec.yaml"}
    workstation_ref = validation_root / "phase_6_workstation_scaling" / "input" / "workstation_reference.json"
    if workstation_ref.exists():
        machine["workstation_reference"] = "../phase_6_workstation_scaling/input/workstation_reference.json"
    included_phases = [
        "Phase 0 — Setup and Reproducibility",
        "Phase 1 — Operator Integrity",
        "Phase 2 — Projection Kernel",
        "Phase 3 — Global Projector",
        "Phase 4 — Fixed-Point Structure",
        "Phase 5 — Drift and Stability",
    ]
    if phase_6_included:
        included_phases.append("Phase 6 — Workstation Scaling")
    return {
        "sort_version": "Version 7",
        "phase": "Phase 7 — Artifact Freeze",
        "package_name": config["freeze_package"]["zip_name"],
        "repository": {
            "name": config["repository"]["name"],
            "canonical_url": config["repository"]["canonical_url"],
            "validation_root": config["repository"]["validation_root"],
            "local_working_copy": str(repo_root),
        },
        "git": git_state,
        "machine": machine,
        "seed": {
            "source": "../phase_0_setup/seed_config.json",
            "global_seed": seed.get("global_seed", 117666),
        },
        "included_phases": included_phases,
        "recommended_phases": {
            "Phase 6 — Workstation Scaling": "included" if phase_6_included else "skipped",
        },
        "artifacts": {
            "inventory": "outputs/artifact_inventory.csv",
            "hashes": "outputs/hashes.txt",
            "zip_package": "outputs/SORT_Version_7_Workstation_Validation.zip",
        },
        "validation_scope": {
            "level": "Level-0 structural validation",
            "uses_empirical_data": False,
            "uses_production_telemetry": False,
            "uses_customer_data": False,
            "uses_hpc": False,
            "uses_sword": False,
            "uses_asdv": False,
        },
        "self_reference_policy": _self_reference_policy(),
        "non_claims": [
            "This frozen package records workstation validation artifacts only.",
            "This frozen package does not constitute empirical validation.",
            "This frozen package does not constitute SWORD execution.",
            "This frozen package does not constitute ASDV execution.",
            "This frozen package does not establish minimality.",
            "This frozen package does not establish structural necessity.",
            "This frozen package does not create a new MOCK version.",
        ],
    }


def _build_freeze_report(
    *,
    config: dict[str, Any],
    package_hash: dict[str, Any],
    phase_6_status: str,
    warnings: list[Any],
    gate_7_passed: bool,
) -> dict[str, Any]:
    return {
        "sort_version": "Version 7",
        "phase": "Phase 7 — Artifact Freeze",
        "freeze_status": "complete" if gate_7_passed else "pending_package_hash",
        "gate_7_passed": gate_7_passed,
        "package": {
            "name": config["freeze_package"]["zip_name"],
            "path": "outputs/SORT_Version_7_Workstation_Validation.zip",
            "sha256": package_hash["sha256"],
            "size_bytes": package_hash["size_bytes"],
        },
        "inputs": {
            "prefreeze_audit_report": "outputs/prefreeze_audit_report.json",
            "artifact_inventory": "outputs/artifact_inventory.csv",
            "hashes": "outputs/hashes.txt",
            "repro_manifest": "outputs/repro_manifest.json",
        },
        "phase_6_status": phase_6_status,
        "self_reference_policy": _self_reference_policy(),
        "warnings": warnings,
        "non_claims": [
            "The freeze operation packages validation artifacts only.",
            "The ZIP package is not a SWORD release.",
            "The ZIP package is not an ASDV evidence run.",
            "The ZIP package is not a new MOCK version.",
            "The ZIP package is not empirical validation.",
        ],
    }


def _self_reference_policy() -> dict[str, Any]:
    return {
        "policy": "Phase 7 self-referential manifest files are included in the ZIP, but stable self-hashes are not attempted.",
        "self_referential_files": SELF_REFERENCE_PATHS,
        "zip_hash_policy": "The ZIP is created with the Phase 7 reports available at packaging time. The repository-side freeze_report.json is then updated with the final ZIP SHA-256 and size.",
    }


def _phase_6_status(audit_report: dict[str, Any]) -> str:
    return "included" if audit_report["phase_status"].get("phase_6_workstation_scaling") == "included" else "skipped"


def _load_seed(validation_root: Path) -> dict[str, Any]:
    path = validation_root / "phase_0_setup" / "seed_config.json"
    if not path.exists():
        return {"global_seed": 117666}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_state(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": _git(repo_root, ["rev-parse", "HEAD"]),
        "branch": _git(repo_root, ["branch", "--show-current"]),
        "dirty": bool(_git(repo_root, ["status", "--porcelain"])),
        "remote_origin": _git(repo_root, ["remote", "get-url", "origin"]),
    }


def _git(repo_root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def _append_zip_hash(hashes_path: Path, zip_relative_path: str, package_hash: dict[str, Any]) -> None:
    existing = hashes_path.read_text(encoding="utf-8") if hashes_path.exists() else ""
    addition = (
        "\n# ZIP package hash\n"
        f"{package_hash['sha256']}  {package_hash['size_bytes']}  {zip_relative_path}\n"
    )
    write_text(hashes_path, existing.rstrip() + addition)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args(argv)
    config_path, phase_root, validation_root, repo_root = resolve_paths(args.config)
    if args.run_all:
        report = run_all(config_path, phase_root, validation_root, repo_root)
    else:
        config = load_phase_7_setup(config_path)
        audit_path = phase_root / config["outputs"]["prefreeze_audit_report"]
        if not audit_path.exists():
            raise FileNotFoundError("Run prefreeze_audit.py before freeze_package.py, or use --run-all.")
        audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_report["audit_status"] != "pass":
            raise RuntimeError("Pre-freeze audit failed; ZIP creation stopped.")
        inventory_path = phase_root / config["outputs"]["artifact_inventory"]
        if not inventory_path.exists():
            rows = build_inventory(validation_root, config)
            write_inventory_csv(inventory_path, rows)
        repro_manifest = _build_repro_manifest(config, repo_root, validation_root, audit_report)
        write_json(phase_root / config["outputs"]["repro_manifest"], repro_manifest)
        pending = _build_freeze_report(
            config=config,
            package_hash={"sha256": "pending-package-hash", "size_bytes": None},
            phase_6_status=_phase_6_status(audit_report),
            warnings=audit_report["warnings"],
            gate_7_passed=False,
        )
        write_json(phase_root / config["outputs"]["freeze_report"], pending)
        rows = read_inventory(inventory_path)
        zip_path = phase_root / config["outputs"]["zip_package"]
        create_zip_package(validation_root, rows, zip_path)
        package_hash = hash_zip_package(zip_path)
        report = _build_freeze_report(
            config=config,
            package_hash=package_hash,
            phase_6_status=_phase_6_status(audit_report),
            warnings=audit_report["warnings"],
            gate_7_passed=True,
        )
        write_freeze_report(phase_root / config["outputs"]["freeze_report"], report)
    print(f"freeze_status={report['freeze_status']}")
    print(f"gate_7_passed={report['gate_7_passed']}")
    print(f"zip_sha256={report['package']['sha256']}")
    return 0 if report["gate_7_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
