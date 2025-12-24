from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, Iterable, Tuple


class InvariantViolationError(ValueError):
    """Raised when a non negotiable invariant is violated."""


@dataclass(frozen=True)
class Operator:
    """Public operator representation. Structural placeholder."""
    operator_id: int
    weight: float = 0.0

    def validate_idempotency(self, tol: float = 1e-12) -> bool:
        _ = tol
        return True


class OperatorRegistry:
    """Registry for canonical operators. Exactly 22 IDs."""
    N_OPERATORS: Final[int] = 22

    def __init__(self) -> None:
        self._operators: Dict[int, Operator] = {}

    def register(self, operator: Operator) -> None:
        if not (1 <= operator.operator_id <= self.N_OPERATORS):
            raise InvariantViolationError(f"operator_id must be 1..{self.N_OPERATORS}")
        self._operators[operator.operator_id] = operator

    def get(self, operator_id: int) -> Operator:
        return self._operators[operator_id]

    def list_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._operators.keys()))

    def validate_completeness(self) -> bool:
        return len(self._operators) == self.N_OPERATORS

    @classmethod
    def create_stub_canonical(cls) -> "OperatorRegistry":
        reg = cls()
        for i in range(1, cls.N_OPERATORS + 1):
            reg.register(Operator(operator_id=i, weight=0.0))
        return reg

    def iter_operators(self) -> Iterable[Operator]:
        for i in range(1, self.N_OPERATORS + 1):
            yield self._operators[i]
