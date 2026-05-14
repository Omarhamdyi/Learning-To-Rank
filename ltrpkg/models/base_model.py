from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        raise NotImplementedError

