from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sort.core.evaluation import CoreEvaluator, EvaluationRequest
from sort.core.result import CoreResult

DOMAIN_ID = "experimental"


@dataclass(frozen=True)
class ExperimentalDomain:
    """Domain module experimental. Experimental domain is stub only."""
    evaluator: CoreEvaluator

    def run_core(self, seed: int, config: Dict[str, Any]) -> CoreResult:
        req = EvaluationRequest(schema_version="0.5.1", domain_id=DOMAIN_ID, seed=seed, config=config)
        return self.evaluator.evaluate(req)
