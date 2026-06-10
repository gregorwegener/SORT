"""Pre-freeze consistency audit for SORT Version 7 Phase 7."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

from report_writer import write_json


REQUIRED_OUTPUTS: dict[str, list[str]] = {
    "phase_0_setup": [
        "env_spec.yaml",
        "seed_config.json",
        "run_manifest.json",
        "README.md",
    ],
    "phase_1_operator_integrity": [
        "input/operator_registry.json",
        "config/phase_1_setup.json",
        "outputs/operator_integrity_report.json",
        "outputs/operator_residuals.csv",
        "README.md",
    ],
    "phase_2_projection_kernel": [
        "input/kernel_definitions.yaml",
        "config/phase_2_setup.json",
        "outputs/kernel_norm_report.json",
        "outputs/kernel_profiles.csv",
        "outputs/projection_residuals.csv",
        "README.md",
    ],
    "phase_3_global_projector": [
        "config/phase_3_setup.json",
        "outputs/projector_validation.json",
        "outputs/projector_residuals.csv",
        "outputs/closure_test_report.json",
        "outputs/composition_stability.csv",
        "README.md",
    ],
    "phase_4_fixed_point": [
        "config/phase_4_setup.json",
        "outputs/fixed_point_metrics.json",
        "outputs/convergence_series.csv",
        "outputs/norm_invariance_report.json",
        "outputs/iteration_stability.json",
        "README.md",
    ],
    "phase_5_drift_stability": [
        "config/phase_5_setup.json",
        "outputs/drift_profiles.csv",
        "outputs/stability_response.json",
        "outputs/synthetic_reference_catalog.json",
        "outputs/drift_metric_definition.json",
        "README.md",
    ],
}

PHASE_6_OUTPUTS = [
    "config/phase_6_setup.json",
    "outputs/scaling_results.csv",
    "outputs/runtime_profiles.json",
    "outputs/memory_profiles.json",
    "outputs/raw_logs",
    "README.md",
]

PHASE_7_REQUIRED = [
    "config/phase_7_setup.json",
    "src/prefreeze_audit.py",
    "src/artifact_inventory.py",
    "src/hash_manifest.py",
    "src/freeze_package.py",
    "src/report_writer.py",
    "README.md",
    "outputs",
]

NEGATIVE_MARKERS = [
    "does not",
    "do not",
    "did not",
    "is not",
    "are not",
    "not ",
    "no ",
    "non-claim",
    "non_claim",
    "without",
    "forbidden",
    "must not",
    "avoid",
    "excluded",
    "false",
]

ADDITIONAL_FORBIDDEN_TERMS = [
    "cloud readiness",
    "new MOCK version",
]

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def load_phase_7_setup(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def check_required_phase_folders(root: Path, config: dict[str, Any]) -> dict[str, str]:
    phase_status = {}
    for phase in config["required_phases"]:
        phase_status[phase] = "present" if (root / phase).is_dir() else "missing"
    for phase in config["recommended_phases"]:
        phase_status[phase] = "included" if (root / phase).is_dir() else "skipped"
    return phase_status


def check_required_outputs(root: Path, config: dict[str, Any]) -> list[str]:
    missing = []
    for phase in config["required_phases"]:
        for rel in REQUIRED_OUTPUTS.get(phase, []):
            path = root / phase / rel
            if not path.exists():
                missing.append(str(Path(phase) / rel).replace("\\", "/"))
    phase_6 = root / "phase_6_workstation_scaling"
    if phase_6.exists():
        for rel in PHASE_6_OUTPUTS:
            path = phase_6 / rel
            if not path.exists():
                missing.append(str(Path("phase_6_workstation_scaling") / rel).replace("\\", "/"))
        workstation_ref = phase_6 / "input" / "workstation_reference.json"
        if workstation_ref.exists() and workstation_ref.suffix.lower() != ".json":
            missing.append("phase_6_workstation_scaling/input/workstation_reference.json")
    phase_7 = root / "phase_7_artifact_freeze"
    for rel in PHASE_7_REQUIRED:
        path = phase_7 / rel
        if not path.exists():
            missing.append(str(Path("phase_7_artifact_freeze") / rel).replace("\\", "/"))
    return missing


def scan_for_forbidden_labels(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    hits = []
    labels = config["audit_rules"]["forbidden_version_labels"]
    for rel_path, text in _iter_text_files(root, config):
        if rel_path == "phase_7_artifact_freeze/config/phase_7_setup.json":
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label in labels:
                if label in line:
                    hits.append({"relative_path": rel_path, "line": line_no, "term": label})
    return hits


def scan_for_forbidden_claims(root: Path, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    terms = list(config["audit_rules"]["forbidden_claim_terms"]) + ADDITIONAL_FORBIDDEN_TERMS
    affirmative_hits = []
    non_claim_hits = []
    for rel_path, text in _iter_text_files(root, config):
        if rel_path in {
            "phase_7_artifact_freeze/config/phase_7_setup.json",
            "phase_7_artifact_freeze/src/prefreeze_audit.py",
        }:
            continue
        lines = text.splitlines()
        lower_text = text.lower()
        for term in terms:
            start = 0
            term_lower = term.lower()
            while True:
                index = lower_text.find(term_lower, start)
                if index == -1:
                    break
                line_no = text.count("\n", 0, index) + 1
                line = lines[line_no - 1].lower() if line_no - 1 < len(lines) else ""
                hit = {"relative_path": rel_path, "line": line_no, "term": term}
                if _is_non_claim_line(line) or _is_non_claim_context(lower_text, index):
                    non_claim_hits.append(hit)
                else:
                    affirmative_hits.append(hit)
                start = index + len(term)
    return {
        "affirmative_claim_hits": affirmative_hits,
        "non_claim_context_hits": non_claim_hits,
    }


def write_prefreeze_audit_report(path: str | Path, report: dict[str, Any]) -> None:
    write_json(path, report)


def build_audit_report(config_path: Path, phase_root: Path, validation_root: Path) -> dict[str, Any]:
    config = load_phase_7_setup(config_path)
    phase_status = check_required_phase_folders(validation_root, config)
    missing = check_required_outputs(validation_root, config)
    label_hits = scan_for_forbidden_labels(validation_root, config)
    claim_scan = scan_for_forbidden_claims(validation_root, config)
    raw_log_hits = _raw_log_hits(validation_root)
    cache_hits = _cache_or_ide_hits(validation_root, config)
    warnings = []
    if cache_hits:
        warnings.append(
            {
                "warning": "Excluded cache or IDE files are present in the working tree but are not eligible for ZIP inclusion.",
                "paths": cache_hits,
            }
        )
    if raw_log_hits:
        missing.extend(raw_log_hits)
    affirmative_hits = claim_scan["affirmative_claim_hits"]
    audit_passed = not missing and not label_hits and not affirmative_hits
    phase_6_status = phase_status.get("phase_6_workstation_scaling", "skipped")
    return {
        "sort_version": "Version 7",
        "phase": "Phase 7 — Artifact Freeze",
        "audit_status": "pass" if audit_passed else "fail",
        "repository": {
            "name": config["repository"]["name"],
            "validation_root": config["repository"]["validation_root"],
        },
        "phase_status": {
            **{phase: phase_status.get(phase, "missing") for phase in config["required_phases"]},
            "phase_6_workstation_scaling": phase_6_status,
        },
        "missing_required_files": missing,
        "forbidden_terms_detected": label_hits + affirmative_hits,
        "non_claim_context_hits": claim_scan["non_claim_context_hits"],
        "affirmative_claim_hits": affirmative_hits,
        "raw_log_hits": raw_log_hits,
        "excluded_cache_or_ide_hits": cache_hits,
        "warnings": warnings,
        "non_claims_verified": bool(claim_scan["non_claim_context_hits"]) and not affirmative_hits,
    }


def _iter_text_files(root: Path, config: dict[str, Any]) -> list[tuple[str, str]]:
    results = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_excluded(path, root, config["exclude_patterns"]):
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("phase_7_artifact_freeze/outputs/"):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            results.append((rel, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return results


def _is_non_claim_context(lower_text: str, index: int) -> bool:
    context = lower_text[max(0, index - 120): index + 120]
    return any(marker in context for marker in NEGATIVE_MARKERS)


def _is_non_claim_line(lower_line: str) -> bool:
    return any(marker in lower_line for marker in NEGATIVE_MARKERS)


def _raw_log_hits(root: Path) -> list[str]:
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix == ".LOG" or path.name.endswith(".log.raw")):
            hits.append(path.relative_to(root).as_posix())
    return hits


def _cache_or_ide_hits(root: Path, config: dict[str, Any]) -> list[str]:
    hits = []
    for path in root.rglob("*"):
        if _is_excluded(path, root, config["exclude_patterns"]):
            continue
        parts = set(path.relative_to(root).parts)
        if parts.intersection({".idea", "__pycache__", ".pytest_cache"}) or path.suffix == ".pyc":
            hits.append(path.relative_to(root).as_posix())
    return sorted(hits)


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
    config = load_phase_7_setup(config_path)
    report = build_audit_report(config_path, phase_root, validation_root)
    output_path = phase_root / config["outputs"]["prefreeze_audit_report"]
    write_prefreeze_audit_report(output_path, report)
    print(f"prefreeze_audit_status={report['audit_status']}")
    return 0 if report["audit_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
