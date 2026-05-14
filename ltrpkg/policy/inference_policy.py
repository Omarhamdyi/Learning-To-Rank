from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from ltrpkg.policy.base_policy import BasePolicy
from ltrpkg.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePolicy(BasePolicy):
    """Inference policy used by runtime search engines."""

    def train(self, data: Any) -> Any:
        logger.warning("InferencePolicy.train was called; this policy is inference-first.")
        return data

    def predict(self, input_data: Any) -> Any:
        model, payload = input_data
        if not hasattr(model, "predict"):
            raise TypeError("Provided model does not expose `predict`.")
        return model.predict(payload)

    def save(self, model: Any, path_to_model: str | Path) -> None:
        path = Path(path_to_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving inference artifact to %s", path)
        joblib.dump(model, path)

    def load(self, path_to_model: str | Path) -> Any:
        path = Path(path_to_model)
        logger.info("Loading inference artifact from %s", path)
        return joblib.load(path)

