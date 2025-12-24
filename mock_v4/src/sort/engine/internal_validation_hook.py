"""Internal validation hook wiring.

This is a placeholder for future internal validations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sort.core.internal_validation import InternalValidationPayload


@dataclass(frozen=True)
class InternalValidationHookResult:
    payload: Optional[InternalValidationPayload]
