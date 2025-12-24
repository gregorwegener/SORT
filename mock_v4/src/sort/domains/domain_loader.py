"""Thin wrapper for domain loading via ApplicationContext."""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

from sort.application.application_module import ApplicationContext


def load_domain_module(repo_root: Path, domain_id: str) -> ModuleType:
    """Load a domain module by domain_id from the given repo root.

    Args:
        repo_root: Path to the repository root directory.
        domain_id: Canonical domain identifier (e.g., "ai-systems", "cosmology").

    Returns:
        The loaded domain module.

    Raises:
        FileNotFoundError: If the domain module does not exist.
    """
    return ApplicationContext(repo_root=repo_root).load_domain_module(domain_id)


def load_domain_observables(repo_root: Path, domain_id: str) -> ModuleType:
    """Load observables module for a domain.

    Args:
        repo_root: Path to the repository root directory.
        domain_id: Canonical domain identifier.

    Returns:
        The loaded observables module.

    Raises:
        FileNotFoundError: If the observables module does not exist.
    """
    return ApplicationContext(repo_root=repo_root).load_domain_observables(domain_id)


def load_domain_requirements(repo_root: Path, domain_id: str) -> ModuleType:
    """Load requirements module for a domain.

    Args:
        repo_root: Path to the repository root directory.
        domain_id: Canonical domain identifier.

    Returns:
        The loaded requirements module.

    Raises:
        FileNotFoundError: If the requirements module does not exist.
    """
    return ApplicationContext(repo_root=repo_root).load_domain_requirements(domain_id)


__all__ = ["load_domain_module", "load_domain_observables", "load_domain_requirements"]
