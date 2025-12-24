from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sort.evidence.evidence_bundle import EvidenceBundle


@dataclass(frozen=True)
class ApplicationResult:
    schema_version: str
    application_id: str
    status: str
    outputs: Dict[str, Any]
    evidence: EvidenceBundle
