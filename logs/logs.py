import logging
import os

# --------------------------------------------------
# Project root = parent directory of "logs"
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "tradex_logs.log")

# logs folder already exists, but this is safe
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger instance.

    Args:
        name (str): Logger name (usually __name__)

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)
