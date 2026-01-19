import yaml
from TradeX.logs.logs import get_logger
import os
import inspect

logger = get_logger(__name__)

def read_config(filename: str = "config.yml") -> dict:
    """
    Load exchange configuration from a YAML file.
    Automatically resolves the path relative to the caller's file.

    Args:
        filename (str): Config file name (default "config.yml")

    Returns:
        dict: Dictionary with keys:
            - exchange_name (str)
            - symbols (list[str])
            - start_date (str)
            - end_date (str)

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required keys are missing.
        yaml.YAMLError: If YAML is malformed.
    """

    try:
        # ---------------------------
        # Get the caller's file location
        # ---------------------------
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        caller_dir = os.path.dirname(os.path.abspath(caller_file))

        # Build config path relative to caller
        config_path = os.path.join(caller_dir, filename)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # ---------------------------
        # Load YAML
        # ---------------------------
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        exchange_name = str(config.get("exchange_name", "")).lower()
        symbols = config.get("symbols", [])
        start_date = config.get("start_date")
        end_date = config.get("end_date", "now")

        if not symbols or not start_date:
            raise ValueError("Config must include 'symbols' and 'start_date'.")

        logger.info(
            f"Configuration loaded | exchange={exchange_name} | "
            f"symbols={symbols} | start_date={start_date} | end_date={end_date}"
        )

        return {
            "exchange_name": exchange_name,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date
        }

    except FileNotFoundError:
        logger.exception(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError:
        logger.exception(f"Error parsing YAML file: {config_path}")
        raise
