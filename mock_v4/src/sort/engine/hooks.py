"""Engine hook interfaces.

This module is intentionally minimal.
It defines hook points for future extensions without implementing execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EngineHookContext:
    schema_version: str
    metadata: Dict[str, Any]


class EngineHook:
    def on_pre_evaluate(self, ctx: EngineHookContext) -> None:
        _ = ctx
        return None

    def on_post_evaluate(self, ctx: EngineHookContext) -> None:
        _ = ctx
        return None
