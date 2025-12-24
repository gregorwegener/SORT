"""Demo application module demonstrating MOCK v4 application contracts.

This is a minimal ApplicationModule implementation for reference.
It loads domains via ApplicationContext, creates an EvidenceBundle,
and returns a stub result. No actual computation is performed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sort.application.application_module import ApplicationContext
from sort.application.application_requirements import ApplicationRequirements
from sort.application.application_result import ApplicationResult
from sort.core.schema import SCHEMA_VERSION
from sort.evidence.evidence_bundle import EvidenceItem


# -----------------------------------------------------------------------------
# Application Metadata
# -----------------------------------------------------------------------------

APPLICATION_ID: str = "demo.stub"
DISPLAY_NAME: str = "Demo Stub Application"
VERSION: str = "0.1.0"


def get_requirements() -> ApplicationRequirements:
    """Declare application requirements per specification.

    Returns:
        ApplicationRequirements with domain_ids and optional capabilities.
    """
    return ApplicationRequirements(
        schema_version=SCHEMA_VERSION,
        domain_ids=("ai-systems", "cosmology"),
        required_capabilities=(),
        required_inputs=(),
    )


def run(context: ApplicationContext, inputs: Dict[str, Any]) -> ApplicationResult:
    """Execute the demo application.

    This is a stub implementation that:
    1. Loads domain modules via ApplicationContext
    2. Creates an EvidenceBundle with a stub item
    3. Returns a result with status="stub"

    Args:
        context: ApplicationContext providing domain loading and capabilities.
        inputs: Input parameters (unused in this stub).

    Returns:
        ApplicationResult with status="stub" and empty outputs.
    """
    requirements = get_requirements()

    # Load domains to verify they exist (no cross-domain imports)
    loaded_domains = []
    for domain_id in requirements.domain_ids:
        try:
            domain_mod = context.load_domain_module(domain_id)
            loaded_domains.append(domain_id)
        except FileNotFoundError:
            pass  # Domain not available in this context

    # Create evidence bundle
    evidence = context.new_evidence_bundle(APPLICATION_ID)
    evidence = evidence.with_item(
        EvidenceItem(
            evidence_id="demo.stub.init",
            kind="stub",
            payload={
                "message": "Demo application executed successfully",
                "domains_loaded": loaded_domains,
            },
        )
    )

    return ApplicationResult(
        schema_version=SCHEMA_VERSION,
        application_id=APPLICATION_ID,
        status="stub",
        outputs={},
        evidence=evidence,
    )


# -----------------------------------------------------------------------------
# Standalone execution for smoke testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Determine repo root (assumes running from repo root or tests)
    repo_root = Path(__file__).resolve().parents[4]

    ctx = ApplicationContext(repo_root=repo_root)
    result = run(ctx, {})

    print(f"Application: {result.application_id}")
    print(f"Status: {result.status}")
    print(f"Schema: {result.schema_version}")
    print(f"Evidence items: {len(result.evidence.items)}")

    sys.exit(0 if result.status == "stub" else 1)
