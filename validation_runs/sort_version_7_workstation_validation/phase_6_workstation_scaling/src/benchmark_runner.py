"""Phase 6 workstation scaling runner.

This runner records controlled workstation execution behavior. It does not
assert speed, readiness, or distributed scaling conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark_kernel import (  # noqa: E402
    estimate_memory_mb,
    identity_projector,
    numpy_available,
    rank_one_projection,
    run_benchmark_kernel,
)
from memory_profile import (  # noqa: E402
    memory_trace,
    peak_memory_mb,
    start_memory_monitor,
    stop_memory_monitor,
)
from runtime_profile import (  # noqa: E402
    capture_runtime_environment,
    capture_thread_environment,
    write_runtime_profile,
)
from scaling_threads import set_thread_environment  # noqa: E402


SCALING_FIELDS = [
    "timestamp",
    "run_id",
    "mode",
    "grid",
    "threads",
    "repeat_index",
    "wall_time_sec",
    "peak_rss_mb",
    "final_residual",
    "git_commit",
    "machine_id",
    "status",
    "skip_reason",
    "error_message",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    phase_root = Path(__file__).resolve().parents[1]
    config_path = (phase_root / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    _ensure_outputs(phase_root, config)
    _update_reference_manifest(phase_root, config)

    runtime_environment = capture_runtime_environment()
    run_id = _run_id()
    selected_runs = _selected_run_specs(config, args)
    skipped_runs = _safe_gate_skips(config, args.mode) if args.mode == "safe-gate" else []

    results: list[dict[str, Any]] = []
    memory_runs: list[dict[str, Any]] = []
    raw_logs: list[dict[str, Any]] = []

    if args.dry_run:
        for spec in selected_runs:
            result = _skipped_result(
                run_id=run_id,
                mode=args.mode,
                grid=spec["grid"],
                threads=spec["threads"],
                repeat_index="dry-run",
                runtime_environment=runtime_environment,
                reason="Dry run requested; workload was not executed.",
            )
            results.append(result)
            raw_logs.append(_raw_log(args, config_path, result, capture_thread_environment()))
    else:
        for spec in selected_runs:
            _run_warmups(phase_root, config, args, spec, runtime_environment, raw_logs)
            repeats = int(args.repeats if args.repeats is not None else spec["repeats"])
            for repeat_index in range(1, repeats + 1):
                result, memory_record, raw_log = _execute_measured_run(
                    phase_root=phase_root,
                    config=config,
                    args=args,
                    spec=spec,
                    repeat_index=repeat_index,
                    run_id=run_id,
                    runtime_environment=runtime_environment,
                    config_path=config_path,
                )
                results.append(result)
                memory_runs.append(memory_record)
                raw_logs.append(raw_log)

    for skip_spec in skipped_runs:
        result = _skipped_result(
            run_id=run_id,
            mode=args.mode,
            grid=skip_spec["grid"],
            threads=skip_spec["threads"],
            repeat_index="skipped",
            runtime_environment=runtime_environment,
            reason=config["execution_policy"]["skip_reason_for_unexecuted_declared_runs"],
        )
        results.append(result)
        memory_runs.append(
            {
                "run_id": result["run_id"],
                "grid": result["grid"],
                "threads": result["threads"],
                "repeat_index": result["repeat_index"],
                "peak_memory_mb": "unavailable",
                "memory_trace": [],
                "completeness_status": "skipped",
                "skip_reason": result["skip_reason"],
            }
        )
        raw_logs.append(_raw_log(args, config_path, result, capture_thread_environment()))

    _write_scaling_results(phase_root / config["outputs"]["scaling_results"], results)
    gate_summary = _gate_6_summary(
        phase_root=phase_root,
        config=config,
        results=results,
        raw_logs=raw_logs,
        memory_runs=memory_runs,
    )
    runtime_profile = _runtime_profile(
        phase_root=phase_root,
        config=config,
        runtime_environment=runtime_environment,
        args=args,
        results=results,
        gate_summary=gate_summary,
    )
    memory_profile = _memory_profile(config=config, memory_runs=memory_runs, gate_summary=gate_summary)
    write_runtime_profile(phase_root / config["outputs"]["runtime_profiles"], runtime_profile)
    write_runtime_profile(phase_root / config["outputs"]["memory_profiles"], memory_profile)
    _write_raw_logs(phase_root, config, raw_logs)

    print(f"Phase 6 runner completed mode={args.mode}")
    print(f"measured_runs={sum(1 for result in results if result['status'] == 'success')}")
    print(f"skipped_runs={sum(1 for result in results if result['status'] == 'skipped')}")
    print(f"failed_runs={sum(1 for result in results if result['status'] == 'failed')}")
    return 0 if any(result["status"] == "success" for result in results) else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SORT Version 7 Phase 6 scaling profile.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["single", "grid", "threads", "full", "safe-gate"], required=True)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("sort_version") != "Version 7":
        raise ValueError("Phase 6 config sort_version must be exactly 'Version 7'.")
    if config.get("phase") != "Phase 6 — Workstation Scaling":
        raise ValueError("Phase 6 config phase label is invalid.")
    if config["benchmark_kernel"]["early_stopping"]:
        raise ValueError("Phase 6 early stopping must be disabled.")
    if config["benchmark_kernel"]["adaptive_parameters"]:
        raise ValueError("Phase 6 adaptive parameters must be disabled.")


def _ensure_outputs(phase_root: Path, config: dict[str, Any]) -> None:
    for key in ["scaling_results", "runtime_profiles", "memory_profiles"]:
        (phase_root / config["outputs"][key]).parent.mkdir(parents=True, exist_ok=True)
    (phase_root / config["outputs"]["raw_logs"]).mkdir(parents=True, exist_ok=True)


def _update_reference_manifest(phase_root: Path, config: dict[str, Any]) -> None:
    manifest_path = phase_root / config["inputs"]["validation_reference_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    availability = {}
    for group_name, paths in manifest["required_references"].items():
        availability[group_name] = {
            path: (phase_root / path).resolve().exists()
            for path in paths
        }
    for group_name, paths in manifest["optional_references"].items():
        availability[group_name] = {
            path: (phase_root / path).resolve().exists()
            for path in paths
        }
    manifest["availability_status"] = availability
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _required_references_available(phase_root: Path, config: dict[str, Any]) -> bool:
    manifest_path = phase_root / config["inputs"]["validation_reference_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    availability = manifest.get("availability_status", {})
    required = availability.get("phase_0", {}) | availability.get("phase_2", {}) | availability.get("phase_3", {})
    return bool(required) and all(required.values())


def _gate_6_summary(
    *,
    phase_root: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    raw_logs: list[dict[str, Any]],
    memory_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    measured_count = sum(1 for result in results if result["status"] == "success")
    skipped_count = sum(1 for result in results if result["status"] == "skipped")
    failed_count = sum(1 for result in results if result["status"] == "failed")
    safe_gate_requires_skips = config["execution_policy"].get("record_skipped_declared_runs", False)
    skipped_runs_recorded = skipped_count > 0 if safe_gate_requires_skips else True
    required_references_available = _required_references_available(phase_root, config)
    memory_recorded_or_explicit = all(
        "peak_memory_mb" in run and run.get("completeness_status") for run in memory_runs
    )
    gate_6_passed = (
        measured_count > 0
        and failed_count == 0
        and skipped_runs_recorded
        and required_references_available
        and memory_recorded_or_explicit
        and len(raw_logs) > 0
    )
    return {
        "measured_run_count": measured_count,
        "skipped_run_count": skipped_count,
        "failed_run_count": failed_count,
        "skipped_declared_runs_recorded": skipped_runs_recorded,
        "required_references_available": required_references_available,
        "memory_recorded_or_explicitly_unavailable": memory_recorded_or_explicit,
        "raw_log_count": len(raw_logs),
        "gate_6_passed": gate_6_passed,
    }


def _selected_run_specs(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.mode == "safe-gate":
        grids = config["execution_policy"]["safe_gate_grids"]
        threads = config["execution_policy"]["safe_gate_threads"]
        repeats = config["execution_policy"]["safe_gate_repeats"]
    elif args.mode == "single":
        grids = [args.grid or config["scaling"]["grids"][0]]
        threads = [args.threads or config["scaling"]["threads"][0]]
        repeats = args.repeats or config["scaling"]["repeats"]
    elif args.mode == "grid":
        grids = config["scaling"]["grids"]
        threads = [args.threads or config["scaling"]["threads"][0]]
        repeats = args.repeats or config["scaling"]["repeats"]
    elif args.mode == "threads":
        grids = [args.grid or config["scaling"]["grids"][0]]
        threads = config["scaling"]["threads"]
        repeats = args.repeats or config["scaling"]["repeats"]
    elif args.mode == "full":
        grids = config["scaling"]["grids"]
        threads = config["scaling"]["threads"]
        repeats = args.repeats or config["scaling"]["repeats"]
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    return [
        {"grid": int(grid), "threads": int(thread), "repeats": int(repeats)}
        for grid in grids
        for thread in threads
    ]


def _safe_gate_skips(config: dict[str, Any], mode: str) -> list[dict[str, int]]:
    if mode != "safe-gate" or not config["execution_policy"]["record_skipped_declared_runs"]:
        return []
    executed = {
        (int(grid), int(thread))
        for grid in config["execution_policy"]["safe_gate_grids"]
        for thread in config["execution_policy"]["safe_gate_threads"]
    }
    declared = {
        (int(grid), int(thread))
        for grid in config["scaling"]["grids"]
        for thread in config["scaling"]["threads"]
    }
    return [
        {"grid": grid, "threads": thread}
        for grid, thread in sorted(declared - executed)
    ]


def _run_warmups(
    phase_root: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    spec: dict[str, Any],
    runtime_environment: dict[str, Any],
    raw_logs: list[dict[str, Any]],
) -> None:
    for warmup_index in range(1, int(config["scaling"]["warmup_runs"]) + 1):
        set_thread_environment(spec["threads"])
        try:
            run_benchmark_kernel(
                spec["grid"],
                rank_one_projection,
                identity_projector,
                config,
                int(config["seed"]["global_seed"]) + warmup_index,
            )
            status = "warmup"
            error = ""
        except Exception as exc:
            status = "warmup_failed"
            error = f"{type(exc).__name__}: {exc}"
        raw_logs.append(
            {
                "timestamp": _timestamp(),
                "command": " ".join(sys.argv),
                "config_path": args.config,
                "mode": args.mode,
                "grid": spec["grid"],
                "threads": spec["threads"],
                "repeat_index": f"warmup_{warmup_index}",
                "thread_environment": capture_thread_environment(),
                "status": status,
                "error_message": error,
                "output_paths": config["outputs"],
            }
        )


def _execute_measured_run(
    *,
    phase_root: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    spec: dict[str, Any],
    repeat_index: int,
    run_id: str,
    runtime_environment: dict[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    timestamp = _timestamp()
    set_thread_environment(spec["threads"])
    thread_env = capture_thread_environment()
    git = runtime_environment["git"]
    machine_id = runtime_environment["machine_id"]
    unsafe_reason = _skip_reason_for_grid(spec["grid"])
    if unsafe_reason:
        result = _skipped_result(
            run_id=run_id,
            mode=args.mode,
            grid=spec["grid"],
            threads=spec["threads"],
            repeat_index=repeat_index,
            runtime_environment=runtime_environment,
            reason=unsafe_reason,
        )
        memory_record = {
            "run_id": run_id,
            "grid": spec["grid"],
            "threads": spec["threads"],
            "repeat_index": repeat_index,
            "peak_memory_mb": "unavailable",
            "memory_trace": [],
            "completeness_status": "skipped",
            "skip_reason": unsafe_reason,
        }
        return result, memory_record, _raw_log(args, config_path, result, thread_env)

    monitor = start_memory_monitor(int(config["runtime"]["memory_sampling_interval_ms"]))
    start = time.perf_counter()
    status = "success"
    error_message = ""
    final_residual: float | str = "unavailable"
    try:
        kernel_result = run_benchmark_kernel(
            spec["grid"],
            rank_one_projection,
            identity_projector,
            config,
            int(config["seed"]["global_seed"]) + repeat_index + spec["threads"],
        )
        final_residual = kernel_result["final_residual"]
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        error_message += "\n" + traceback.format_exc()
    wall_time = time.perf_counter() - start
    stop_memory_monitor()
    peak_mb = peak_memory_mb()
    trace = memory_trace()
    result = {
        "timestamp": timestamp,
        "run_id": run_id,
        "mode": args.mode,
        "grid": spec["grid"],
        "threads": spec["threads"],
        "repeat_index": repeat_index,
        "wall_time_sec": f"{wall_time:.9f}" if status == "success" else "",
        "peak_rss_mb": peak_mb,
        "final_residual": final_residual,
        "git_commit": git.get("commit", "unavailable"),
        "machine_id": machine_id,
        "status": status,
        "skip_reason": "",
        "error_message": error_message,
    }
    memory_record = {
        "run_id": run_id,
        "grid": spec["grid"],
        "threads": spec["threads"],
        "repeat_index": repeat_index,
        "peak_memory_mb": peak_mb,
        "memory_trace": trace[:10],
        "completeness_status": "complete" if monitor.psutil_available else "partial",
        "sampling_method": monitor.method,
    }
    return result, memory_record, _raw_log(args, config_path, result, thread_env)


def _skip_reason_for_grid(grid: int) -> str:
    estimate = estimate_memory_mb(grid)
    if not numpy_available() and int(grid) > 64:
        return "Skipped because NumPy is unavailable and list fallback would be unsafe."
    if estimate > 2048.0:
        return f"Skipped because estimated transient memory {estimate:.1f} MB exceeds safe runner limit."
    return ""


def _skipped_result(
    *,
    run_id: str,
    mode: str,
    grid: int,
    threads: int,
    repeat_index: int | str,
    runtime_environment: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "timestamp": _timestamp(),
        "run_id": run_id,
        "mode": mode,
        "grid": grid,
        "threads": threads,
        "repeat_index": repeat_index,
        "wall_time_sec": "",
        "peak_rss_mb": "",
        "final_residual": "",
        "git_commit": runtime_environment["git"].get("commit", "unavailable"),
        "machine_id": runtime_environment["machine_id"],
        "status": "skipped",
        "skip_reason": reason,
        "error_message": "",
    }


def _write_scaling_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCALING_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in SCALING_FIELDS})


def _runtime_profile(
    *,
    phase_root: Path,
    config: dict[str, Any],
    runtime_environment: dict[str, Any],
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    gate_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sort_version": "Version 7",
        "phase": "Phase 6 — Workstation Scaling",
        "repository": {
            "name": "gregorwegener/SORT",
            "canonical_url": "https://github.com/gregorwegener/SORT",
            "validation_root": "validation_runs/sort_version_7_workstation_validation",
            "phase_path": "validation_runs/sort_version_7_workstation_validation/phase_6_workstation_scaling",
        },
        "configuration_snapshot": config,
        "declared_scaling_matrix": {
            "grids": config["scaling"]["grids"],
            "threads": config["scaling"]["threads"],
        },
        "executed_profile": args.mode,
        "machine_profile": runtime_environment,
        "workstation_reference": _load_workstation_reference(phase_root),
        "python_version": runtime_environment["python_version"],
        "library_versions": {"numpy": runtime_environment["numpy_version"]},
        "blas": runtime_environment["blas"],
        "os": {
            "platform": runtime_environment["platform"],
            "os_name": runtime_environment["os_name"],
            "os_version": runtime_environment["os_version"],
        },
        "cpu": runtime_environment["cpu"],
        "git_state": runtime_environment["git"],
        "thread_environment_variables": capture_thread_environment(),
        "run_summaries": [result for result in results if result["status"] == "success"],
        "skipped_run_summaries": [result for result in results if result["status"] == "skipped"],
        "failed_run_summaries": [result for result in results if result["status"] == "failed"],
        "gate_6_completion": gate_summary,
        "overall_passed": gate_summary["gate_6_passed"],
        "gate_6_passed": gate_summary["gate_6_passed"],
        "non_claims": config["non_claims"],
    }


def _load_workstation_reference(phase_root: Path) -> dict[str, Any] | None:
    path = phase_root / "input" / "workstation_reference.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _memory_profile(
    config: dict[str, Any],
    memory_runs: list[dict[str, Any]],
    gate_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sort_version": "Version 7",
        "phase": "Phase 6 — Workstation Scaling",
        "peak_memory_per_run": memory_runs,
        "memory_sampling_method": "psutil RSS when available, standard-library tracemalloc fallback",
        "memory_sampling_interval_ms": config["runtime"]["memory_sampling_interval_ms"],
        "memory_trace": [
            {
                "run_id": run["run_id"],
                "grid": run["grid"],
                "threads": run["threads"],
                "repeat_index": run["repeat_index"],
                "trace": run.get("memory_trace", []),
            }
            for run in memory_runs
            if run.get("memory_trace")
        ],
        "completeness_status": "complete"
        if all(run.get("completeness_status") == "complete" for run in memory_runs if run.get("completeness_status") != "skipped")
        else "partial",
        "skipped_or_failed_memory_traces": [
            run for run in memory_runs if run.get("completeness_status") in {"skipped", "failed", "partial"}
        ],
        "gate_6_passed": gate_summary["gate_6_passed"],
        "non_claims": config["non_claims"],
    }


def _write_raw_logs(phase_root: Path, config: dict[str, Any], raw_logs: list[dict[str, Any]]) -> None:
    raw_dir = phase_root / config["outputs"]["raw_logs"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, log in enumerate(raw_logs, start=1):
        path = raw_dir / f"phase_6_run_{index:04d}.json"
        path.write_text(
            json.dumps(log, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )


def _raw_log(
    args: argparse.Namespace,
    config_path: Path,
    result: dict[str, Any],
    thread_env: dict[str, str],
) -> dict[str, Any]:
    return {
        "timestamp": result["timestamp"],
        "command": " ".join(sys.argv),
        "config_path": str(config_path),
        "mode": result["mode"],
        "grid_size": result["grid"],
        "thread_count": result["threads"],
        "repeat_index": result["repeat_index"],
        "environment_variables": thread_env,
        "status": result["status"],
        "error_message": result.get("error_message", ""),
        "skip_reason": result.get("skip_reason", ""),
        "output_paths": {
            "scaling_results": "outputs/scaling_results.csv",
            "runtime_profiles": "outputs/runtime_profiles.json",
            "memory_profiles": "outputs/memory_profiles.json",
            "raw_logs": "outputs/raw_logs/",
        },
    }


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_id() -> str:
    return "sort-version-7-phase-6-" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
