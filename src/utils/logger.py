"""Merkezi loglama yapılandırması — konsol ve dosya çıktısı.

Format: 2024-01-01 12:00:00 | modül_adı | SEVİYE | mesaj
Loglar logs/ klasörüne yazılır (.gitignore'da).
"""

import logging
import sys
from pathlib import Path

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Modül adına göre yapılandırılmış logger döndürür."""
    global _configured
    if not _configured:
        _setup_logging()
        _configured = True
    return logging.getLogger(name)


def _setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
