"""Thread scaling helpers for SORT Version 7 Phase 6."""

from __future__ import annotations

import os
from typing import Any


THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def set_thread_environment(thread_count: int) -> dict[str, str]:
    for key in THREAD_ENV_KEYS:
        os.environ[key] = str(int(thread_count))
    return {key: os.environ.get(key, "unavailable") for key in THREAD_ENV_KEYS}


def run_thread_scaling(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["scaling"]["grids"][0]
    return [
        {"grid": grid, "threads": thread, "mode": "threads"}
        for thread in config["scaling"]["threads"]
    ]


def collect_thread_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "result_count": len(results),
        "threads": sorted({result.get("threads") for result in results}),
        "statuses": sorted({result.get("status") for result in results}),
    }
