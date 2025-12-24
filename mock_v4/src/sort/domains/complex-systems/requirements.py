from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

DOMAIN_ID = "complex-systems"


@dataclass(frozen=True)
class ComplexSystemsRequirements:
    """Declarative domain requirements."""
    required_capabilities: Tuple[str, ...] = tuple()
    required_inputs: Tuple[str, ...] = tuple()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": DOMAIN_ID,
            "required_capabilities": list(self.required_capabilities),
            "required_inputs": list(self.required_inputs),
        }
