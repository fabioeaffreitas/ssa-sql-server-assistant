"""Small, idempotent standard-library logging configuration for SSA."""

import logging
import os


LOGGER_NAME = "ssa"
DEFAULT_LOG_LEVEL = logging.INFO
_HANDLER_MARKER = "_ssa_owned_handler"
_ALLOWED_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _resolve_level(level: str | int | None) -> int:
    configured = os.getenv("SSA_LOG_LEVEL", "INFO") if level is None else level
    if isinstance(configured, int):
        return configured if configured in _ALLOWED_LEVELS.values() else DEFAULT_LOG_LEVEL
    return _ALLOWED_LEVELS.get(str(configured).strip().upper(), DEFAULT_LOG_LEVEL)


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Configure the ``ssa`` logger once and return it.

    Only handlers owned by this module are changed. The root logger and
    unrelated third-party loggers are left untouched.
    """

    logger = logging.getLogger(LOGGER_NAME)
    resolved_level = _resolve_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False

    owned_handler = None
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            owned_handler = handler
            break

    if owned_handler is None:
        owned_handler = logging.StreamHandler()
        setattr(owned_handler, _HANDLER_MARKER, True)
        owned_handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(owned_handler)

    owned_handler.setLevel(resolved_level)
    return logger
