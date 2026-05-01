from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePipeline(ABC):
    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate(self, data: Any) -> Any:
        raise NotImplementedError

