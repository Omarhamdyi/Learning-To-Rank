from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ltrpkg.config.settings import AppSettings
from ltrpkg.datapipeline.base_pipeline import BasePipeline
from ltrpkg.utils.logging import get_logger
from ltrpkg.utils.text import normalize_text

logger = get_logger(__name__)


class FeaturePipeline(BasePipeline):
    """Shared corpus loading and normalization pipeline for training/inference."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def load(self) -> dict[str, pd.DataFrame]:
        logger.info("Loading raw datasets from %s", self.settings.data_dir)
        data_dir = Path(self.settings.data_dir)
        desc_df = pd.read_csv(data_dir / "product_descriptions.csv", encoding="latin1")
        attrs_df = pd.read_csv(data_dir / "attributes.csv", encoding="latin1")
        train_df = pd.read_csv(data_dir / "train.csv", encoding="latin1")
        return {"descriptions": desc_df, "attributes": attrs_df, "train": train_df}

    def prepare(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        desc_df = data["descriptions"].copy()
        attrs = data["attributes"].copy()
        train_df = data["train"].copy()

        attrs["value"] = attrs["value"].astype(str)
        all_attrs = attrs.groupby("product_uid")["value"].apply(lambda x: " ".join(map(str, x))).reset_index()
        all_attrs = all_attrs.rename(columns={"value": "all_attributes"})

        brand = attrs.loc[attrs["name"].str.lower() == "mfg brand name", ["product_uid", "value"]]
        brand = brand.drop_duplicates("product_uid").rename(columns={"value": "brand"})

        unique_products = train_df[["product_uid", "product_title"]].drop_duplicates("product_uid")
        corpus = desc_df.merge(unique_products, on="product_uid", how="inner")
        corpus = corpus.merge(brand, on="product_uid", how="left")
        corpus = corpus.merge(all_attrs, on="product_uid", how="left")

        corpus["product_title_norm"] = corpus["product_title"].fillna("").map(normalize_text)
        corpus["product_description_norm"] = corpus["product_description"].fillna("").map(normalize_text)
        corpus["all_attributes_norm"] = corpus["all_attributes"].fillna("").map(normalize_text)
        corpus["brand_norm"] = corpus["brand"].fillna("").map(normalize_text)
        return corpus

    def validate(self, data: Any) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("FeaturePipeline.validate expects a pandas DataFrame.")
        required = {"product_uid", "product_title", "product_description"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Prepared corpus is missing required columns: {sorted(missing)}")
        return data

