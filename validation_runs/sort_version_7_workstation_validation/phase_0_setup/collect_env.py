"""Collect Phase 0 reproducibility metadata for SORT Version 7.

This script creates or updates env_spec.yaml and run_manifest.json, and verifies
or creates seed_config.json. It does not run Phase 1 or later validation logic.
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import importlib.metadata
import io
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


SORT_VERSION_VALUE = "Version 7"
SORT_LABEL = "SORT Version 7"
PHASE_NAME = "Phase 0 \u2014 Setup and Reproducibility"
GLOBAL_SEED = 117666
REPOSITORY_NAME = "gregorwegener/SORT"
CANONICAL_URL = "https://github.com/gregorwegener/SORT"
VALIDATION_ROOT = "validation_runs/sort_version_7_workstation_validation"
DEFAULT_BRANCH = "main"

PHASE_SEQUENCE = [
    "Phase 0 \u2014 Setup and Reproducibility",
    "Phase 1 \u2014 Operator Integrity",
    "Phase 2 \u2014 Projection Kernel",
    "Phase 3 \u2014 Global Projector",
    "Phase 4 \u2014 Fixed-Point Structure",
    "Phase 5 \u2014 Drift and Stability",
    "Phase 6 \u2014 Workstation Scaling",
    "Phase 7 \u2014 Artifact Freeze",
]

SEED_CONFIG = {
    "sort_version": SORT_VERSION_VALUE,
    "phase": PHASE_NAME,
    "global_seed": GLOBAL_SEED,
    "rng_backend": "numpy.random.default_rng",
    "deterministic": True,
    "adaptive_reseed_allowed": False,
    "seed_policy": (
        "The global seed is fixed for all SORT Version 7 workstation validation "
        "phases. Adaptive reseeding is not allowed."
    ),
    "allowed_usage": [
        "synthetic reference state generation",
        "deterministic perturbation generation",
        "reproducible validation runs",
    ],
    "not_allowed_usage": [
        "empirical fitting",
        "production telemetry sampling",
        "customer data processing",
        "adaptive optimization",
    ],
    "notes": [
        "This seed configuration supports reproducibility only.",
        "It does not establish empirical validity or structural necessity.",
    ],
}


def timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run_id(now: str) -> str:
    compact = (
        now.replace("-", "")
        .replace(":", "")
        .replace("+", "p")
        .replace(".", "")
    )
    return f"sort-version-7-workstation-validation-{compact}"


def script_paths() -> tuple[Path, Path, Path]:
    phase_dir = Path(__file__).resolve().parent
    validation_root = phase_dir.parent
    repository_root = validation_root.parent.parent
    return phase_dir, validation_root, repository_root


def command_output(args: list[str], cwd: Path, timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "unavailable"
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        return False, output or "unavailable"
    return True, output or "unavailable"


def git_state(repository_root: Path) -> dict[str, Any]:
    ok_commit, commit = command_output(["git", "rev-parse", "HEAD"], repository_root)
    ok_branch, branch = command_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], repository_root
    )
    ok_remote, remote_origin = command_output(
        ["git", "remote", "get-url", "origin"], repository_root
    )
    ok_status, status = command_output(["git", "status", "--porcelain"], repository_root)

    available = bool(ok_commit or ok_branch or ok_remote or ok_status)
    dirty = None if not ok_status else bool(status and status != "unavailable")

    return {
        "commit": commit if ok_commit else "unavailable",
        "branch": branch if ok_branch else "unavailable",
        "dirty": dirty,
        "available": available,
        "remote_origin": remote_origin if ok_remote else "unavailable",
        "default_branch": DEFAULT_BRANCH,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
    except Exception:
        return "unavailable"


def cpu_model() -> str:
    if platform.system().lower() == "windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if value:
                    return str(value).strip()
        except Exception:
            pass
    processor = platform.processor()
    return processor.strip() if processor else "unavailable"


def core_counts() -> tuple[int | None, int | None]:
    logical = os.cpu_count()
    physical = None
    try:
        import psutil  # type: ignore[import-not-found]

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:
        pass
    return physical, logical


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def ram_gb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass

    if platform.system().lower() == "windows":
        try:
            memory = MEMORYSTATUSEX()
            memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                return round(memory.ullTotalPhys / (1024**3), 2)
        except Exception:
            pass
    return None


def blas_info() -> dict[str, str]:
    try:
        import numpy as np  # type: ignore[import-not-found]

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.__config__.show()
        details = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines())
        lower_details = details.lower()
        if "mkl" in lower_details:
            implementation = "MKL"
        elif "openblas" in lower_details:
            implementation = "OpenBLAS"
        elif "blis" in lower_details:
            implementation = "BLIS"
        elif details:
            implementation = "detected"
        else:
            implementation = "unavailable"
        return {"implementation": implementation, "details": details or "unavailable"}
    except Exception:
        return {"implementation": "unavailable", "details": "unavailable"}


def ensure_seed_config(path: Path) -> None:
    write_required = True
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            write_required = not (
                current.get("sort_version") == SORT_VERSION_VALUE
                and current.get("phase") == PHASE_NAME
                and current.get("global_seed") == GLOBAL_SEED
                and current.get("deterministic") is True
                and current.get("adaptive_reseed_allowed") is False
            )
        except json.JSONDecodeError:
            write_required = True

    if write_required:
        path.write_text(
            json.dumps(SEED_CONFIG, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_lines(data: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_lines(value, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
        return lines
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(data)}"]


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text("\n".join(yaml_lines(data)) + "\n", encoding="utf-8")


def build_common_context(now: str, repository_root: Path) -> dict[str, Any]:
    physical_cores, logical_threads = core_counts()
    return {
        "generated_at": now,
        "repository": {
            "name": REPOSITORY_NAME,
            "canonical_url": CANONICAL_URL,
            "validation_root": VALIDATION_ROOT,
            "local_working_copy": str(repository_root),
        },
        "machine": {
            "hostname": socket.gethostname() or "unavailable",
            "platform": platform.platform() or "unavailable",
            "os_name": platform.system() or "unavailable",
            "os_version": platform.version() or "unavailable",
            "cpu_model": cpu_model(),
            "physical_cores": physical_cores,
            "logical_threads": logical_threads,
            "ram_gb": ram_gb(),
        },
        "python": {
            "version": platform.python_version() or "unavailable",
            "executable": sys.executable or "unavailable",
        },
        "libraries": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "sympy": package_version("sympy"),
        },
        "blas": blas_info(),
        "git": git_state(repository_root),
    }


def build_env_spec(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "sort_version": SORT_VERSION_VALUE,
        "phase": PHASE_NAME,
        "generated_at": context["generated_at"],
        "repository": context["repository"],
        "machine": context["machine"],
        "python": context["python"],
        "libraries": context["libraries"],
        "blas": context["blas"],
        "git": {
            "commit": context["git"]["commit"],
            "branch": context["git"]["branch"],
            "dirty": context["git"]["dirty"],
            "remote_origin": context["git"]["remote_origin"],
            "default_branch": context["git"]["default_branch"],
        },
        "seed_reference": {
            "file": "seed_config.json",
            "global_seed": GLOBAL_SEED,
        },
        "notes": [
            "Phase 0 records environment, repository, and reproducibility metadata only.",
            "No scientific results are generated in this phase.",
        ],
    }


def build_run_manifest(context: dict[str, Any], current_run_id: str) -> dict[str, Any]:
    return {
        "run_id": current_run_id,
        "sort_version": SORT_VERSION_VALUE,
        "phase": PHASE_NAME,
        "generated_at": context["generated_at"],
        "repository": context["repository"],
        "machine": context["machine"],
        "python": context["python"],
        "git": context["git"],
        "references": {
            "env_spec": "phase_0_setup/env_spec.yaml",
            "seed_config": "phase_0_setup/seed_config.json",
        },
        "phase_0_outputs": {
            "env_spec": "phase_0_setup/env_spec.yaml",
            "seed_config": "phase_0_setup/seed_config.json",
            "run_manifest": "phase_0_setup/run_manifest.json",
            "collect_env": "phase_0_setup/collect_env.py",
            "readme": "phase_0_setup/README.md",
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
        "phase_sequence": PHASE_SEQUENCE,
        "non_claims": [
            "Phase 0 produces no scientific results.",
            "This run does not constitute empirical validation.",
            "This run does not constitute SWORD execution.",
            "This run does not constitute a new MOCK version.",
            "This run does not establish minimality or structural necessity.",
        ],
    }


def main() -> int:
    phase_dir, _, repository_root = script_paths()
    phase_dir.mkdir(parents=True, exist_ok=True)

    seed_path = phase_dir / "seed_config.json"
    env_path = phase_dir / "env_spec.yaml"
    manifest_path = phase_dir / "run_manifest.json"

    ensure_seed_config(seed_path)

    now = timestamp()
    context = build_common_context(now, repository_root)
    current_run_id = run_id(now)

    write_yaml(env_path, build_env_spec(context))
    manifest_path.write_text(
        json.dumps(
            build_run_manifest(context, current_run_id),
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"{PHASE_NAME}: metadata collection complete")
    print(f"repository: {REPOSITORY_NAME}")
    print(f"validation_root: {VALIDATION_ROOT}")
    print(f"global_seed: {GLOBAL_SEED}")
    print(f"env_spec: {env_path}")
    print(f"run_manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
