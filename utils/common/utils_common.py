import yaml
from TradeX.utils.common.logs import get_logger
import os

logger = get_logger(__name__)

def read_config(config_path : str) -> dict:
    """
    Load exchange configuration from a YAML file.

    This function reads a YAML config file and extracts common parameters:
        - exchange_name
        - symbols (list of trading pairs)
        - start_date
        - end_date (optional, defaults to "now")

    Args:
        config_path (str, optional): Path to the YAML config file. Defaults to "config.yml".

    Returns:
        dict: Dictionary with keys:
            - exchange_name (str)
            - symbols (list[str])
            - start_date (str)
            - end_date (str)

    Raises:
        ValueError: If 'symbols' or 'start_date' are missing in the YAML file.
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        exchange_name = config.get("exchange_name", "").lower()
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
