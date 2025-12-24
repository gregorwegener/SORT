from pathlib import Path
from sort.catalog.catalog_loader import CatalogLoader


def test_public_catalog_loads_public_only() -> None:
    loader = CatalogLoader(repo_root=Path("."))
    doc = loader.load_public_catalog()
    assert all(e.maturity == "public" for e in doc.applications)
