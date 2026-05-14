from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BasePolicy(ABC):
    @abstractmethod
    def train(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, model: Any, path_to_model: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path_to_model: str | Path) -> Any:
        raise NotImplementedError

