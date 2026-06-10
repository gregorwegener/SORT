"""Memory profiling helpers for SORT Version 7 Phase 6."""

from __future__ import annotations

import os
import threading
import time
import tracemalloc
from typing import Any


_monitor = None


class MemoryMonitor:
    def __init__(self, interval_ms: int):
        self.interval_sec = max(float(interval_ms) / 1000.0, 0.001)
        self.trace: list[dict[str, Any]] = []
        self.running = False
        self.thread: threading.Thread | None = None
        self.psutil_available = False
        self.method = "tracemalloc"
        self._process = None

    def start(self) -> None:
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self.psutil_available = True
            self.method = "psutil_rss_and_tracemalloc"
        except Exception:
            self._process = None
        tracemalloc.start()
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._sample()
        tracemalloc.stop()

    def peak_mb(self) -> float | str:
        values = [
            sample.get("rss_mb")
            for sample in self.trace
            if isinstance(sample.get("rss_mb"), (int, float))
        ]
        if values:
            return max(values)
        values = [
            sample.get("tracemalloc_current_mb")
            for sample in self.trace
            if isinstance(sample.get("tracemalloc_current_mb"), (int, float))
        ]
        return max(values) if values else "unavailable"

    def _sample_loop(self) -> None:
        while self.running:
            self._sample()
            time.sleep(self.interval_sec)

    def _sample(self) -> None:
        current, peak = tracemalloc.get_traced_memory()
        sample = {
            "timestamp": time.time(),
            "tracemalloc_current_mb": current / (1024.0 * 1024.0),
            "tracemalloc_peak_mb": peak / (1024.0 * 1024.0),
        }
        if self._process is not None:
            try:
                sample["rss_mb"] = self._process.memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                sample["rss_mb"] = "unavailable"
        else:
            sample["rss_mb"] = "unavailable"
        self.trace.append(sample)


def start_memory_monitor(interval_ms: int):
    global _monitor
    _monitor = MemoryMonitor(interval_ms)
    _monitor.start()
    return _monitor


def stop_memory_monitor():
    if _monitor is not None:
        _monitor.stop()


def peak_memory_mb():
    if _monitor is None:
        return "unavailable"
    return _monitor.peak_mb()


def memory_trace():
    if _monitor is None:
        return []
    return list(_monitor.trace)
