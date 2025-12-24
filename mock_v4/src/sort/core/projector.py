from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .operators import InvariantViolationError, OperatorRegistry, Operator


@dataclass(frozen=True)
class GlobalProjector:
    """Global projector descriptor. Structural only."""
    operators: Tuple[Operator, ...]
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.operators) != OperatorRegistry.N_OPERATORS:
            raise InvariantViolationError("GlobalProjector requires exactly 22 operators")
        if len(self.weights) != OperatorRegistry.N_OPERATORS:
            raise InvariantViolationError("GlobalProjector requires exactly 22 weights")
        if abs(sum(self.weights)) > 1e-14:
            raise InvariantViolationError("projector weights must sum to zero")

    @classmethod
    def from_registry(cls, registry: OperatorRegistry) -> "GlobalProjector":
        ops = tuple(registry.iter_operators())
        weights = tuple(op.weight for op in ops)
        return cls(operators=ops, weights=weights)
