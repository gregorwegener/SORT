from pathlib import Path
from sort.application.application_module import ApplicationContext


def test_domains_loadable() -> None:
    ctx = ApplicationContext(repo_root=Path("."))
    for did in ["ai-systems", "complex-systems", "quantum-systems", "cosmology", "experimental"]:
        dm = ctx.load_domain_module(did)
        assert hasattr(dm, "DOMAIN_ID")
