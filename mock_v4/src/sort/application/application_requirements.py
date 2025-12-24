from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ApplicationRequirements:
    schema_version: str
    domain_ids: Tuple[str, ...]
    required_capabilities: Tuple[str, ...] = tuple()
    required_inputs: Tuple[str, ...] = tuple()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain_ids": list(self.domain_ids),
            "required_capabilities": list(self.required_capabilities),
            "required_inputs": list(self.required_inputs),
        }
