from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .application_module import ApplicationContext
from .application_result import ApplicationResult
from .application_requirements import ApplicationRequirements


class ApplicationModule(ABC):
    """Orchestrates domains and produces evidence."""

    @property
    @abstractmethod
    def application_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def requirements(self) -> ApplicationRequirements:
        raise NotImplementedError

    @abstractmethod
    def run(self, ctx: ApplicationContext, inputs: Dict[str, Any]) -> ApplicationResult:
        raise NotImplementedError
