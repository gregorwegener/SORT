from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .intervention_descriptor import InterventionDescriptor


@dataclass(frozen=True)
class ControlIntent:
    """Descriptor only. Allowed modes: observe, intervene, validate."""
    mode: str
    interventions: Tuple[InterventionDescriptor, ...] = tuple()
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mode not in ("observe", "intervene", "validate"):
            raise ValueError("mode must be observe, intervene, validate")
