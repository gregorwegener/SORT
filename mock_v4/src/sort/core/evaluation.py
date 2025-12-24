from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .internal_validation import InternalValidationPayload
from .kernel import ProjectionKernel
from .operators import OperatorRegistry
from .projector import GlobalProjector
from .result import CoreResult


@dataclass(frozen=True)
class EvaluationRequest:
    """Core evaluation request. Placeholder for future execution."""
    schema_version: str
    domain_id: str
    seed: int
    config: Dict[str, Any]


class CoreEvaluator:
    """Validates structural invariants and returns a stub CoreResult."""
    def __init__(
        self,
        registry: Optional[OperatorRegistry] = None,
        kernel: Optional[ProjectionKernel] = None,
    ) -> None:
        self._registry = registry or OperatorRegistry.create_stub_canonical()
        self._kernel = kernel or ProjectionKernel()

    def evaluate(self, request: EvaluationRequest) -> CoreResult:
        projector = GlobalProjector.from_registry(self._registry)
        self._kernel.spec.validate_invariants()
        _ = projector

        run_id = str(uuid.uuid4())
        fingerprint = self._fingerprint_config(request.config)
        internal = InternalValidationPayload(
            idempotency_checks={"stub": True},
            notes={"mode": "stub"},
        )
        return CoreResult(
            schema_version=request.schema_version,
            run_id=run_id,
            seed=request.seed,
            config_fingerprint=fingerprint,
            domain_id=request.domain_id,
            status="stub",
            summary={"message": "MOCK v4 core evaluation is a structural stub"},
            _internal=internal,
        )

    @staticmethod
    def _fingerprint_config(config: Dict[str, Any]) -> str:
        payload = repr(sorted(config.items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
