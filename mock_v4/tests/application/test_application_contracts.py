from pathlib import Path
from sort.application.application_module import ApplicationContext


def test_context_evidence_bundle() -> None:
    ctx = ApplicationContext(repo_root=Path("."))
    eb = ctx.new_evidence_bundle(application_id="demo")
    assert eb.application_id == "demo"
