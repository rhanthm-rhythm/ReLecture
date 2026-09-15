from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_run_logger(
    project_dir: str | Path,
    name: str = "relecture",
    log_filename: str = "run.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger that writes immediately-flushed logs to project_dir/run.log
    as well as streaming to sys.stderr.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times for the same logger
    if logger.handlers:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger.removeHandler(handler)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler writing to project_dir/run.log
    log_path = Path(project_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Stream handler for console/terminal
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


def get_run_logger(name: str = "relecture") -> logging.Logger:
    """Get the active logger or standard fallback logger."""
    return logging.getLogger(name)
