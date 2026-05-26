from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class CatalogEntry:
    application_id: str
    display_name: str
    maturity: str
    domain_ids: Tuple[str, ...]
    entrypoint: str
    description: str
    version: str
    cluster: Optional[str] = None
    core_3: bool = False
    related_whitepaper: str = ""
    dimensions: Dict[str, str] | None = None


@dataclass(frozen=True)
class CatalogDocument:
    schema_version: str
    generated_at_utc: str
    applications: Tuple[CatalogEntry, ...]

    def public_only(self) -> "CatalogDocument":
        return CatalogDocument(
            schema_version=self.schema_version,
            generated_at_utc=self.generated_at_utc,
            applications=tuple(a for a in self.applications if a.maturity == "public"),
        )
