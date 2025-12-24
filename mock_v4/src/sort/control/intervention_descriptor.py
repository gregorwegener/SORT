from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class InterventionDescriptor:
    """Descriptor only. No control code is executed."""
    intervention_id: str
    target: str
    parameters: Dict[str, Any] = None  # type: ignore[assignment]
