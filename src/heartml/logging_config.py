from __future__ import annotations

import logging
import os
import sys
from pythonjsonlogger import jsonlogger


def setup_logging(app_name: str = "heartml") -> logging.Logger:
    """
    JSON structured logger with console + optional file handler.
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    # Avoid duplicate handlers in reloads/tests
    if logger.handlers:
        return logger

    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d"
    formatter = jsonlogger.JsonFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_file = os.getenv("LOG_FILE")  # e.g. logs/app.jsonl
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
