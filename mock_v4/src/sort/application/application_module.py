from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional

from sort.capabilities.registry import CapabilityRegistry
from sort.control.control_intent import ControlIntent
from sort.evidence.evidence_bundle import EvidenceBundle


@dataclass(frozen=True)
class ApplicationContext:
    schema_version: str = "0.5.1"
    capability_registry: CapabilityRegistry = CapabilityRegistry()
    control_intent: Optional[ControlIntent] = None
    repo_root: Path = Path(".")

    def new_evidence_bundle(self, application_id: str) -> EvidenceBundle:
        return EvidenceBundle(schema_version=self.schema_version, application_id=application_id)

    def _load_file_module(self, module_name: str, path: Path) -> ModuleType:
        import sys

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load module from: {path}")

        mod = importlib.util.module_from_spec(spec)

        # Important for dataclasses and annotation resolution in Python 3.13
        sys.modules[module_name] = mod

        spec.loader.exec_module(mod)
        return mod

    def load_domain_module(self, domain_id: str) -> ModuleType:
        p = self.repo_root / "src" / "sort" / "domains" / domain_id / "domain.py"
        if not p.exists():
            raise FileNotFoundError(f"domain module not found for domain_id: {domain_id}")
        return self._load_file_module(f"sort_domains_{domain_id.replace('-', '_')}_domain", p)

    def load_domain_observables(self, domain_id: str) -> ModuleType:
        p = self.repo_root / "src" / "sort" / "domains" / domain_id / "observables.py"
        if not p.exists():
            raise FileNotFoundError(f"observables module not found for domain_id: {domain_id}")
        return self._load_file_module(f"sort_domains_{domain_id.replace('-', '_')}_observables", p)

    def load_domain_requirements(self, domain_id: str) -> ModuleType:
        p = self.repo_root / "src" / "sort" / "domains" / domain_id / "requirements.py"
        if not p.exists():
            raise FileNotFoundError(f"requirements module not found for domain_id: {domain_id}")
        return self._load_file_module(f"sort_domains_{domain_id.replace('-', '_')}_requirements", p)
