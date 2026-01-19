import logging
import os
from pathlib import Path
# --------------------------------------------------
# Project root = parent directory of "logs"
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # logs.py is in TradeX/utils/common/
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger instance that logs both to console and to a separate file
    named after the module.

    Args:
        name (str): Logger name (usually __name__)

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        # File log path is based on module name
        safe_name = name.replace(".", "_")  # Replace dots with underscores
        log_file = os.path.join(LOG_DIR, f"{safe_name}.log")

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
