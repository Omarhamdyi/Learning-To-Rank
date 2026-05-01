from __future__ import annotations

import logging
from pathlib import Path

from ltrpkg.config.settings import get_settings

def configure_logging(level: int = logging.INFO, force: bool = False) -> None:
    settings = get_settings()
    log_path = Path(settings.app_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if not root.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
        root.addHandler(file_handler)
    else:
        file_attached = False
        target = str(log_path.resolve())
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if handler.baseFilename == target:
                        file_attached = True
                except Exception:
                    continue
        if not file_attached:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
