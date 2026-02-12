import yaml
from TradeX.utils.common.logs import get_logger
import os

logger = get_logger("config_loader")

def read_config(
    exchange_name: str,
    filename: str = "config.yml"
) -> dict:
    """
    Load exchange configuration from:
    TradeX/data/<exchange_name>/config.yml
    """

    try:
        # ---------------------------
        # Resolve project root
        # ---------------------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(
            os.path.join(current_dir, "..", "..")
        )

        # ---------------------------
        # Build path: TradeX/data/<exchange>/config.yml
        # ---------------------------
        config_path = os.path.join(
            project_root,
            "data",
            exchange_name.lower(),
            filename
        )

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

    except Exception:
        logger.exception("Failed to load config.")
        raise
