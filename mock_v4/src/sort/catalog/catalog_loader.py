from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .catalog_schema import CatalogDocument, CatalogEntry


class CatalogLoader:
    """Loads catalog assets from repo root. Enforces public maturity for public catalog."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self._repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

    def load_public_catalog(self) -> CatalogDocument:
        path = self._repo_root / "catalog" / "catalog.public.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._parse_document(raw).public_only()

    def load_private_catalog_optional(self) -> Optional[CatalogDocument]:
        path = self._repo_root / "catalog" / "catalog.private.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._parse_document(raw)

    @staticmethod
    def _parse_document(raw: Dict[str, Any]) -> CatalogDocument:
        apps = []
        for item in raw.get("applications", []):
            if not isinstance(item, dict):
                continue
            apps.append(
                CatalogEntry(
                    application_id=str(item.get("application_id", "")),
                    display_name=str(item.get("display_name", "")),
                    maturity=str(item.get("maturity", "")),
                    domain_ids=tuple(item.get("domain_ids", [])),
                    entrypoint=str(item.get("entrypoint", "")),
                    description=str(item.get("description", "")),
                    version=str(item.get("version", "")),
                )
            )
        return CatalogDocument(
            schema_version=str(raw.get("schema_version", "")),
            generated_at_utc=str(raw.get("generated_at_utc", "")),
            applications=tuple(apps),
        )
