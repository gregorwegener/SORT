from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence, Tuple


class KernelInvariantError(ValueError):
    """Raised when kernel invariants are violated."""


@dataclass(frozen=True)
class KernelSpec:
    """Kernel specification. Public immutable contract."""
    schema_version: str = "0.5.1"
    sigma0_calibrated: float = 0.00190643
    SIGMA0_CANONICAL: Final[float] = 0.00190643

    def validate_invariants(self) -> None:
        if self.sigma0_calibrated != self.SIGMA0_CANONICAL:
            raise KernelInvariantError("sigma0_calibrated must equal the canonical value")


@dataclass(frozen=True)
class ProjectionKernel:
    """Projection kernel placeholder."""
    spec: KernelSpec = KernelSpec()

    def evaluate(self, k: Sequence[float]) -> Tuple[float, ...]:
        if len(k) == 0:
            return tuple()
        return tuple(1.0 for _ in k)

    def validate_normalization(self) -> bool:
        return self.evaluate([0.0])[0] == 1.0
