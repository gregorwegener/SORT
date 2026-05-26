from pathlib import Path

from sort.catalog.catalog_loader import CatalogLoader


def test_public_catalog_loads_public_only() -> None:
    loader = CatalogLoader(repo_root=Path("."))
    doc = loader.load_public_catalog()
    assert all(e.maturity == "public" for e in doc.applications)


def test_public_catalog_v62_counts_and_core3() -> None:
    loader = CatalogLoader(repo_root=Path("."))
    doc = loader.load_public_catalog()

    assert doc.schema_version in {"0.5.1", "6.2"}
    assert len(doc.applications) in {18, 107}

    if doc.schema_version == "6.2":
        assert len(doc.applications) == 107
        ids = {entry.application_id for entry in doc.applications}
        assert {"ai.01", "ai.04", "ai.13"}.issubset(ids)

        core3 = {entry.application_id for entry in doc.applications if entry.core_3}
        assert core3 == {"ai.01", "ai.04", "ai.13"}

        domain_counts = {}
        for entry in doc.applications:
            for domain_id in entry.domain_ids:
                domain_counts[domain_id] = domain_counts.get(domain_id, 0) + 1

        assert domain_counts["ai-systems"] == 52
        assert domain_counts["complex-systems"] == 28
        assert domain_counts["quantum-systems"] == 11
        assert domain_counts["sovereign"] == 5
        assert domain_counts["cosmology"] == 11
