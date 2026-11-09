"""Логирование: цветная консоль + ротация файлов + опциональный JSON."""
import json
import logging
import logging.handlers
from pathlib import Path

import colorlog

from backend.core.config import Settings

_LOG_FORMAT = "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_MAX_BYTES = 5 * 1024 * 1024
_FILE_BACKUPS = 5


class JsonFormatter(logging.Formatter):
    """Плоский JSON-формат (для сборщиков логов)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(settings: Settings) -> None:
    """Идемпотентная настройка корневого логгера backend."""
    root = logging.getLogger("backend")
    if root.handlers:  # уже настроен (важно для reload и повторных вызовов)
        return
    root.setLevel(settings.LOG_LEVEL.upper())

    console = colorlog.StreamHandler()
    console.setFormatter(colorlog.ColoredFormatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=_FILE_MAX_BYTES,
        backupCount=_FILE_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        JsonFormatter() if settings.LOG_JSON else logging.Formatter(_FILE_FORMAT)
    )
    root.addHandler(file_handler)

    root.propagate = False
