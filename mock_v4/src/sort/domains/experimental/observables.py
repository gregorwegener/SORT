from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sort.core.result import CoreResult

DOMAIN_ID = "experimental"


@dataclass(frozen=True)
class ExperimentalObservables:
    """Observables derived from CoreResult. Stub only."""
    def compute(self, core_result: CoreResult) -> Dict[str, Any]:
        return {
            "domain_id": DOMAIN_ID,
            "core_status": core_result.status,
            "observable_stub": True,
        }
