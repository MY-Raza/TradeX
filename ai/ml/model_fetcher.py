import numpy as np
from TradeX.utils.db.utils import get_best_model
from TradeX.utils.common.logs import get_logger

logger = get_logger("model_fetcher")

best_model = get_best_model()

if best_model:
    logger.info(f"The best model: {best_model}")
else:
    logger.info("No valid model found")