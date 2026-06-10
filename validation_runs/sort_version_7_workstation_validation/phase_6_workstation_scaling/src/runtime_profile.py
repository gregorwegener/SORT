"""Runtime environment capture for SORT Version 7 Phase 6."""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def capture_runtime_environment() -> dict[str, Any]:
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine_id": socket.gethostname() or "unavailable",
        "platform": platform.platform() or "unavailable",
        "os_name": platform.system() or "unavailable",
        "os_version": platform.version() or "unavailable",
        "cpu": platform.processor() or "unavailable",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": _package_version("numpy"),
        "blas": _blas_info(),
        "git": _git_state(),
    }


def capture_thread_environment() -> dict[str, str]:
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    return {key: os.environ.get(key, "unavailable") for key in keys}


def write_runtime_profile(path: str | Path, profile: dict[str, Any]) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def _blas_info() -> dict[str, str]:
    try:
        import numpy as np

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.__config__.show()
        details = buffer.getvalue().strip() or "unavailable"
        lower = details.lower()
        if "mkl" in lower:
            implementation = "MKL"
        elif "openblas" in lower:
            implementation = "OpenBLAS"
        else:
            implementation = "detected" if details != "unavailable" else "unavailable"
        return {"implementation": implementation, "details": details}
    except Exception:
        return {"implementation": "unavailable", "details": "unavailable"}


def _git_state() -> dict[str, str]:
    return {
        "commit": _git_output(["git", "rev-parse", "HEAD"]),
        "branch": _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote_origin": _git_output(["git", "remote", "get-url", "origin"]),
    }


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=10)
    except Exception:
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip() or "unavailable"
