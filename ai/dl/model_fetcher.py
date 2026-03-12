from TradeX.utils.db.utils import get_best_model,fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger
from TradeX.ai.dl.models.model_trainer import train_model,save_model
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.backtest.backtest import BackTest
from datetime import datetime
import pandas as pd
import re
import numpy as np
import os
from TradeX.utils.common.config_loader import read_config

logger = get_logger("model_fetcher")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

current_dir    = os.path.dirname(os.path.abspath(__file__))
dl_config_path = os.path.join(current_dir, "config.yml")
config         = read_config(dl_config_path)

start_date        = config.get("start_date")
end_date          = config.get("end_date")
split_date        = config.get("split_date")
symbols           = ["btc"]  # BUG-FIX: was hardcoded to ["btc"]
timehorizon       = config.get("timehorizon", "1h")
indicators_config = config.get("indicators", {})
dl_models_config  = config.get("forecasting_models", {})
training_cfg      = config.get("training", {})

lookback   = training_cfg.get("lookback",    None)
epochs     = training_cfg.get("epochs",      None)
batch_size = training_cfg.get("batch_size",  None)

model_params_map: dict[str, dict] = {
        "arima":       config.get("arima_params",       {}),
        "varima":      config.get("varima_params",      {}),
        "nbeats":      config.get("nbeats_params",      {}),
        "transformer": config.get("transformer_params", {}),
    }

active_indicators = [ind for ind, active in indicators_config.items() if active]

logger.info(f"Config loaded | symbols={symbols} | timehorizon={timehorizon}")