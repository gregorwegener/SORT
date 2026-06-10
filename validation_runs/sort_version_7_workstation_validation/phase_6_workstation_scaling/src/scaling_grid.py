"""Grid scaling helpers for SORT Version 7 Phase 6."""

from __future__ import annotations

from typing import Any


def run_grid_scaling(config: dict[str, Any]) -> list[dict[str, Any]]:
    threads = [config["scaling"]["threads"][0]]
    return [
        {"grid": grid, "threads": thread, "mode": "grid"}
        for grid in config["scaling"]["grids"]
        for thread in threads
    ]


def collect_grid_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "result_count": len(results),
        "grids": sorted({result.get("grid") for result in results}),
        "statuses": sorted({result.get("status") for result in results}),
    }
