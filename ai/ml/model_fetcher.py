import numpy as np
import pickle
from pathlib import Path
import os
import pickle
from pathlib import Path
from TradeX.utils.db.utils import get_best_model, get_important_features
from TradeX.utils.common.logs import get_logger
from TradeX.utils.common.config_loader import read_config

logger = get_logger("model_fetcher")

# ==================================================
# MAIN EXECUTION
# ==================================================

best_model = get_best_model()

if best_model:
    logger.info(f"The best model: {best_model}")
else:
    logger.info("No valid model found")
    best_model = None

important_features = None
model = None

if best_model:
    important_features = get_important_features(best_model)

    if important_features:
        logger.info(f"The best features for {best_model} are: {important_features}")
    else:
        logger.info("No valid features found")


current_dir = os.path.dirname(os.path.abspath(__file__))
ml_config_path = os.path.join(current_dir, "config.yml")
config = read_config(ml_config_path)
