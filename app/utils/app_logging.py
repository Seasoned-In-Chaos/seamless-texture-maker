"""Application logging and crash handling."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler

try:
    from PyQt6.QtCore import qInstallMessageHandler
    from PyQt6.QtWidgets import QApplication
except Exception:  # pragma: no cover - PyQt may be unavailable in tooling contexts.
    qInstallMessageHandler = None
    QApplication = object


LOGGER_NAME = "seams"


def user_data_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "SeamlessTextureMaker")
    os.makedirs(path, exist_ok=True)
    return path


def log_path() -> str:
    logs_dir = os.path.join(user_data_dir(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, "seams.log")


def get_logger(name: str | None = None) -> logging.Logger:
    root = LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(root)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        handler = RotatingFileHandler(log_path(), maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)

    if qInstallMessageHandler is not None:
        qInstallMessageHandler(_qt_message_handler)

    return logger


def install_exception_hook() -> None:
    def _hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        get_logger("crash").critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _hook


class LoggingApplication(QApplication):
    """QApplication that logs uncaught exceptions raised by Qt event handlers."""

    def notify(self, receiver, event):  # noqa: N802
        try:
            return super().notify(receiver, event)
        except Exception as exc:
            log_exception(get_logger("qt"), "Unhandled exception in Qt event loop", exc)
            return False


def log_exception(logger: logging.Logger, message: str, exc: BaseException) -> None:
    logger.error("%s: %s\n%s", message, exc, "".join(traceback.format_exception(exc)))


def _qt_message_handler(mode, context, message) -> None:
    logger = get_logger("qt")
    level_name = getattr(mode, "name", str(mode)).lower()
    if "fatal" in level_name or "critical" in level_name:
        logger.error(message)
    elif "warning" in level_name:
        logger.warning(message)
    else:
        logger.debug(message)
