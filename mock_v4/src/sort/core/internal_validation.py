from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InternalValidationPayload:
    """Internal only payload. NEVER serialized."""
    jacobi_residuals: Optional[Dict[str, float]] = None
    idempotency_checks: Optional[Dict[str, bool]] = None
    notes: Optional[Dict[str, str]] = None
    _opaque: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        raise RuntimeError(
            "InternalValidationPayload must never be serialized. "
            "This is an internal-only object."
        )

    def to_json(self) -> str:
        raise RuntimeError(
            "InternalValidationPayload must never be serialized. "
            "This is an internal-only object."
        )
