import yaml
from TradeX.utils.common.logs import get_logger
import os

logger = get_logger("config_loader")

def read_config(
    exchange_name: str = None,
    filename: str = "config.yml"
) -> dict:
    """
    Load configuration dynamically from either:
    1. New ML path: TradeX/ai/ml/config.yml
    2. Old path: TradeX/data/<exchange>/config.yml
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
        # Try new ML path first
        # ---------------------------
        new_path = os.path.join(project_root, "ai", "ml", filename)
        old_path = None
        if exchange_name:
            old_path = os.path.join(project_root, "data", exchange_name.lower(), filename)

        # ---------------------------
        # Determine which config to use
        # ---------------------------
        if os.path.exists(new_path):
            config_path = new_path
        elif old_path and os.path.exists(old_path):
            config_path = old_path
        else:
            raise FileNotFoundError(
                f"Config file not found in either:\n"
                f"New path: {new_path}\n"
                f"Old path: {old_path if old_path else 'N/A'}"
            )

        # ---------------------------
        # Load YAML
        # ---------------------------
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # ---------------------------
        # Basic exchange info
        # ---------------------------
        exchange_name = str(config.get("exchange_name", exchange_name if exchange_name else "")).lower()
        symbols = config.get("symbols", [])
        start_date = config.get("start_date")
        end_date = config.get("end_date", "now")

        # Optional fields
        timehorizon = config.get("timehorizon", "1h")
        limit = config.get("limit", 1000)
        classifiers = config.get("classifiers", {})
        regressors = config.get("regressors", {})
        indicators = config.get("indicators", {})

        if not symbols:
            raise ValueError("Config must include 'symbols'")

        logger.info(
            f"Configuration loaded |"
            f"symbols={symbols} | start_date={start_date} | end_date={end_date} | "
            f"timehorizon={timehorizon} | limit={limit} | path={config_path}"
        )

        return {
            "exchange_name": exchange_name,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "timehorizon": timehorizon,
            "limit": limit,
            "classifiers": classifiers,
            "regressors": regressors,
            "indicators": indicators
        }

    except Exception:
        logger.exception("Failed to load config.")
        raise
