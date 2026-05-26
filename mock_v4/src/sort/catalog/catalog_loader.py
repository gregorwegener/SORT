from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .catalog_schema import CatalogDocument, CatalogEntry


class CatalogLoader:
    """Loads catalog assets from repo root.

    The loader supports both the original MOCK v4 flat catalog schema and the
    public catalog v6.2 grouped domain schema. The public runtime view remains
    a flat tuple of CatalogEntry objects so existing callers keep a stable API.
    """

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
    def _parse_document(raw: Any) -> CatalogDocument:
        if isinstance(raw, list):
            return CatalogLoader._parse_flat_legacy_document(raw)

        if not isinstance(raw, dict):
            return CatalogDocument(schema_version="", generated_at_utc="", applications=tuple())

        applications_raw = raw.get("applications", [])
        if isinstance(applications_raw, dict):
            apps = CatalogLoader._parse_grouped_applications(
                applications_raw,
                version=str(raw.get("version", raw.get("schema_version", ""))),
            )
        elif isinstance(applications_raw, list):
            apps = CatalogLoader._parse_flat_applications(
                applications_raw,
                version=str(raw.get("version", raw.get("schema_version", ""))),
            )
        else:
            apps = []

        return CatalogDocument(
            schema_version=str(raw.get("version", raw.get("schema_version", ""))),
            generated_at_utc=str(raw.get("date", raw.get("generated_at_utc", ""))),
            applications=tuple(apps),
        )

    @staticmethod
    def _parse_flat_legacy_document(raw: Iterable[Dict[str, Any]]) -> CatalogDocument:
        apps = CatalogLoader._parse_flat_applications(raw, version="")
        return CatalogDocument(schema_version="", generated_at_utc="", applications=tuple(apps))

    @staticmethod
    def _parse_flat_applications(raw: Iterable[Dict[str, Any]], version: str) -> list[CatalogEntry]:
        apps: list[CatalogEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            application_id = str(item.get("application_id", item.get("id", "")))
            display_name = str(item.get("display_name", item.get("title", "")))
            description = str(item.get("description", item.get("one_liner", "")))
            domain_ids_value = item.get("domain_ids", None)
            if domain_ids_value is None:
                domain_id = item.get("domain_id", "")
                domain_ids = (str(domain_id),) if domain_id else tuple()
            else:
                domain_ids = tuple(str(x) for x in domain_ids_value)
            apps.append(
                CatalogEntry(
                    application_id=application_id,
                    display_name=display_name,
                    maturity=str(item.get("maturity", "public")),
                    domain_ids=domain_ids,
                    entrypoint=str(item.get("entrypoint", "")),
                    description=description,
                    version=str(item.get("version", version)),
                    cluster=item.get("cluster"),
                    core_3=bool(item.get("core_3", False)),
                    related_whitepaper=str(item.get("related_whitepaper", "")),
                    dimensions=item.get("dimensions") if isinstance(item.get("dimensions"), dict) else None,
                )
            )
        return apps

    @staticmethod
    def _parse_grouped_applications(raw: Dict[str, Any], version: str) -> list[CatalogEntry]:
        apps: list[CatalogEntry] = []
        for domain_id, items in raw.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                apps.append(
                    CatalogEntry(
                        application_id=str(item.get("id", item.get("application_id", ""))),
                        display_name=str(item.get("title", item.get("display_name", ""))),
                        maturity=str(item.get("maturity", "public")),
                        domain_ids=(str(domain_id),),
                        entrypoint=str(item.get("entrypoint", "")),
                        description=str(item.get("one_liner", item.get("description", ""))),
                        version=str(item.get("version", version)),
                        cluster=item.get("cluster"),
                        core_3=bool(item.get("core_3", False)),
                        related_whitepaper=str(item.get("related_whitepaper", "")),
                        dimensions=item.get("dimensions") if isinstance(item.get("dimensions"), dict) else None,
                    )
                )
        return apps
