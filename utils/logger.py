"""Project logging helpers.

Set ``LOG_FORMAT=json`` for structured logs, or leave it unset for compact
human-readable console logs. All modules should call ``get_logger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional


_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Small JSON formatter with stable keys for downstream parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, Mapping):
            payload.update(extra)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: Optional[str] = None, log_file: Optional[Path] = None) -> None:
    """Configure root logging once per process."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    structured = os.getenv("LOG_FORMAT", "").lower() == "json"

    if structured:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Use stdout so PYTHONUNBUFFERED=1 / python -u makes every log line
    # appear immediately in the GUI (QProcess MergedChannels captures stdout).
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str = __name__) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


@contextmanager
def crash_logged(
    logger: logging.Logger,
    action: str,
    on_crash: Optional[Callable[[BaseException], Any]] = None,
) -> Iterator[None]:
    """Log exceptions with stack traces and optionally run a recovery callback."""

    try:
        yield
    except Exception as exc:
        logger.error("%s failed: %s", action, exc, exc_info=True)
        if on_crash is not None:
            try:
                on_crash(exc)
            except Exception as recovery_exc:
                logger.error(
                    "Crash recovery failed after %s: %s",
                    action,
                    recovery_exc,
                    exc_info=True,
                )
        raise


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
