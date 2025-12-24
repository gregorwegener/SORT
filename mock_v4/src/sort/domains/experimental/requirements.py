from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

DOMAIN_ID = "experimental"


@dataclass(frozen=True)
class ExperimentalRequirements:
    """Declarative domain requirements."""
    required_capabilities: Tuple[str, ...] = ("experimental.reserved.v1",)
    required_inputs: Tuple[str, ...] = tuple()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": DOMAIN_ID,
            "required_capabilities": list(self.required_capabilities),
            "required_inputs": list(self.required_inputs),
        }
