import yaml
from TradeX.utils.common.logs import get_logger
import os

logger = get_logger("config_loader")


def read_config(config_path: str) -> dict:
    """
    Load configuration from a given YAML file path.

    Args:
        config_path (str): Full path to the config.yml file.

    Returns:
        dict: Configuration including symbols, classifiers, regressors, indicators,
              and XGBoost hyperparameters.
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        # Load YAML
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Basic exchange info
        exchange_name = str(config.get("exchange_name", "")).lower()
        symbols = config.get("symbols", [])
        start_date = config.get("start_date")
        end_date = config.get("end_date", "now")
        split_date = config.get("split_date")
        train_ratio =  config.get("train_ratio")

        # Optional fields
        timehorizon = config.get("timehorizon", "1h")
        classifiers = config.get("classifiers", {})
        regressors = config.get("regressors", {})
        indicators = config.get("indicators", {})

        # XGBoost hyperparameters
        xgboost_classifier_params = config.get("xgboost_classifier_params", {})
        xgboost_regressor_params = config.get("xgboost_regressor_params", {})

        # =========================
        # Forecasting Models
        # =========================

        forecasting_models = config.get("forecasting_models", {})

        # Model parameters
        arima_params = config.get("arima_params", {})
        varima_params = config.get("varima_params", {})
        nbeats_params = config.get("nbeats_params", {})
        transformer_params = config.get("transformer_params", {})

        # =========================
        # Training Settings
        # =========================

        training = config.get("training", {})
        covariates = config.get("covariates", {})

        if not symbols:
            raise ValueError("Config must include 'symbols'")

        logger.info(
            f"Configuration loaded | symbols={symbols} | start_date={start_date} | "
            f"end_date={end_date} | timehorizon={timehorizon} | path={config_path}"
        )

        return {
            "exchange_name": exchange_name,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "timehorizon": timehorizon,
            "classifiers": classifiers,
            "regressors": regressors,
            "indicators": indicators,
            "xgboost_classifier_params": xgboost_classifier_params,
            "xgboost_regressor_params": xgboost_regressor_params,
            "split_date":split_date,
            "train_ratio":train_ratio,
            # models
            "forecasting_models": forecasting_models,

            # model params
            "arima_params": arima_params,
            "varima_params": varima_params,
            "nbeats_params": nbeats_params,
            "transformer_params": transformer_params,

            # training
            "training": training,
            "covariates": covariates,

        }

    except Exception:
        logger.exception("Failed to load config.")
        raise
