from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb

from ltrpkg.models.base_model import BaseModel
from ltrpkg.utils.logging import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    Compatibility wrapper matching required name.
    Under the hood this wraps LightGBM Booster currently used by the project.
    """

    def __init__(self, booster: lgb.Booster | None = None) -> None:
        self.booster = booster

    def fit(self, X: Any, y: Any) -> "XGBoostModel":
        raise NotImplementedError("Training path is script-based; use train_regressor*.py for fitting.")

    def predict(self, X: Any) -> Any:
        if self.booster is None:
            raise RuntimeError("Model booster not loaded.")
        return self.booster.predict(X)

    def load_booster(self, model_file: str | Path) -> "XGBoostModel":
        path = Path(model_file)
        logger.info("Loading LightGBM booster from %s", path)
        self.booster = lgb.Booster(model_file=str(path))
        return self

    def save(self, path: str | Path) -> None:
        if self.booster is None:
            raise RuntimeError("Cannot save empty model.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving wrapped booster to %s", out)
        joblib.dump(self.booster, out)

