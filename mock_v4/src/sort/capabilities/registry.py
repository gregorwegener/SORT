from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CapabilityHandle:
    """Capability handle. Default disabled. execute is a silent no-op."""
    capability_id: str
    enabled: bool = False

    def execute(self, *args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs
        return None


class CapabilityRegistry:
    """Registry with silent no-op semantics."""
    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilityHandle] = {}

    def register(self, handle: CapabilityHandle) -> None:
        self._capabilities[handle.capability_id] = handle

    def get(self, capability_id: str) -> CapabilityHandle:
        return self._capabilities.get(
            capability_id,
            CapabilityHandle(capability_id=capability_id, enabled=False),
        )

    def is_enabled(self, capability_id: str) -> bool:
        return self.get(capability_id).enabled
