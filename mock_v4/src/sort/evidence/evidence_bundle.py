from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class EvidenceBundle:
    schema_version: str
    application_id: str
    items: Tuple[EvidenceItem, ...] = tuple()

    def with_item(self, item: EvidenceItem) -> "EvidenceBundle":
        return EvidenceBundle(
            schema_version=self.schema_version,
            application_id=self.application_id,
            items=self.items + (item,),
        )
