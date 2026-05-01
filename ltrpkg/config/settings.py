from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppSettings:
    base_dir: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    app_log_file: Path
    st_model_name: str
    st_remote_timeout_sec: int
    st_remote_chunk_size: int
    st_batch_size_local: int
    st_batch_size_remote: int


def _from_env() -> AppSettings:
    base_dir = Path(os.getenv("LTR_BASE_DIR", Path(__file__).resolve().parents[2]))
    data_dir = Path(os.getenv("LTR_DATA_DIR", base_dir / "data/raw"))
    models_dir = Path(os.getenv("LTR_MODELS_DIR", base_dir / "models"))
    logs_dir = Path(os.getenv("LTR_LOGS_DIR", base_dir / "logs"))
    app_log_file = Path(os.getenv("LTR_APP_LOG_FILE", logs_dir / "ltr_app.log"))
    return AppSettings(
        base_dir=base_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        app_log_file=app_log_file,
        st_model_name=os.getenv("LTR_ST_MODEL_NAME", "all-MiniLM-L6-v2"),
        st_remote_timeout_sec=_to_int(os.getenv("LTR_ST_REMOTE_TIMEOUT_SEC"), 30),
        st_remote_chunk_size=_to_int(os.getenv("LTR_ST_REMOTE_CHUNK_SIZE"), 64),
        st_batch_size_local=_to_int(os.getenv("LTR_ST_BATCH_SIZE_LOCAL"), 16),
        st_batch_size_remote=_to_int(os.getenv("LTR_ST_BATCH_SIZE_REMOTE"), 64),
    )


def _from_pydantic() -> AppSettings:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class _PydanticSettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="LTR_", extra="ignore")

        base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
        data_dir: Path | None = None
        models_dir: Path | None = None
        logs_dir: Path | None = None
        app_log_file: Path | None = None

        st_model_name: str = "all-MiniLM-L6-v2"
        st_remote_timeout_sec: int = 30
        st_remote_chunk_size: int = 64
        st_batch_size_local: int = 16
        st_batch_size_remote: int = 64

        def model_post_init(self, __context: Any) -> None:
            if self.data_dir is None:
                self.data_dir = self.base_dir / "data/raw"
            if self.models_dir is None:
                self.models_dir = self.base_dir / "models"
            if self.logs_dir is None:
                self.logs_dir = self.base_dir / "logs"
            if self.app_log_file is None:
                self.app_log_file = self.logs_dir / "ltr_app.log"

    cfg = _PydanticSettings()
    return AppSettings(
        base_dir=cfg.base_dir,
        data_dir=cfg.data_dir or (cfg.base_dir / "data/raw"),
        models_dir=cfg.models_dir or (cfg.base_dir / "models"),
        logs_dir=cfg.logs_dir or (cfg.base_dir / "logs"),
        app_log_file=cfg.app_log_file or ((cfg.logs_dir or (cfg.base_dir / "logs")) / "ltr_app.log"),
        st_model_name=cfg.st_model_name,
        st_remote_timeout_sec=cfg.st_remote_timeout_sec,
        st_remote_chunk_size=cfg.st_remote_chunk_size,
        st_batch_size_local=cfg.st_batch_size_local,
        st_batch_size_remote=cfg.st_batch_size_remote,
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    try:
        return _from_pydantic()
    except Exception:
        return _from_env()
