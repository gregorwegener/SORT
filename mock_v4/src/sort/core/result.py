from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .internal_validation import InternalValidationPayload


@dataclass(frozen=True)
class CoreResult:
    """Public serializable core result."""
    schema_version: str
    run_id: str
    seed: int
    config_fingerprint: str
    domain_id: str
    status: str
    summary: Dict[str, Any]
    _internal: Optional[InternalValidationPayload] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seed": self.seed,
            "config_fingerprint": self.config_fingerprint,
            "domain_id": self.domain_id,
            "status": self.status,
            "summary": self.summary,
        }

    def internal_payload(self) -> Optional[InternalValidationPayload]:
        return self._internal
