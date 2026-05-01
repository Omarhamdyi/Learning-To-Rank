from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from ltrpkg.policy.base_policy import BasePolicy
from ltrpkg.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingPolicy(BasePolicy):
    """Generic training policy wrapper for script-based training flows."""

    def train(self, data: Any) -> Any:
        logger.info("Starting training policy execution.")
        if not callable(data):
            raise TypeError("TrainingPolicy.train expects a callable training entrypoint.")
        model = data()
        logger.info("Training policy execution completed.")
        return model

    def predict(self, input_data: Any) -> Any:
        logger.info("Running prediction via TrainingPolicy.")
        model, payload = input_data
        if not hasattr(model, "predict"):
            raise TypeError("Provided model does not expose `predict`.")
        return model.predict(payload)

    def save(self, model: Any, path_to_model: str | Path) -> None:
        path = Path(path_to_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model artifact to %s", path)
        joblib.dump(model, path)

    def load(self, path_to_model: str | Path) -> Any:
        path = Path(path_to_model)
        logger.info("Loading model artifact from %s", path)
        return joblib.load(path)

