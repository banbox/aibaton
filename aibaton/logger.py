"""
Simple file logger for aibaton.

Usage:
    from aibaton.logger import setup_logger, logger

    # Setup once at the start
    setup_logger(
        filepath="/path/to/app.log",
        level="DEBUG",  # or logging.DEBUG
        fmt="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Then use logger anywhere
    logger.info("something happened")
    logger.debug("debug info")
    logger.error("error occurred")
"""

import logging
from pathlib import Path
from typing import Optional, Union

# Global logger instance
logger = logging.getLogger("aibaton")


def setup_logger(
    filepath: Optional[Union[str, Path]] = None,
    level: Union[str, int] = logging.INFO,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    mode: str = "a",
    encoding: str = "utf-8",
    console: bool = True,
) -> logging.Logger:
    """
    Configure the aibaton logger.

    Args:
        filepath: Path to the log file. If None, no file handler is added.
                  Parent directories will be created if needed.
        level: Logging level (e.g., "DEBUG", "INFO", logging.DEBUG, logging.INFO).
        fmt: Log message format string.
        datefmt: Date format string.
        mode: File mode, 'a' for append, 'w' for overwrite.
        encoding: File encoding.
        console: If True, also output logs to stdout.

    Returns:
        Configured logger instance.
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Create file handler if filepath is provided
    if filepath is not None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(filepath, mode=mode, encoding=encoding)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Create console handler if enabled
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def add_console_handler(
    level: Union[str, int] = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Add a console handler to the logger (optional, for debugging).

    Args:
        level: Logging level.
        fmt: Log format string. Uses the same format as file handler if not specified.

    Returns:
        Logger instance.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if fmt:
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
