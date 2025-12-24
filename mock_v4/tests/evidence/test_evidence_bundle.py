from sort.evidence.evidence_bundle import EvidenceBundle, EvidenceItem


def test_evidence_bundle_append() -> None:
    eb = EvidenceBundle(schema_version="0.5.1", application_id="a")
    eb2 = eb.with_item(EvidenceItem(evidence_id="x", kind="k", payload={"ok": True}))
    assert len(eb2.items) == 1
