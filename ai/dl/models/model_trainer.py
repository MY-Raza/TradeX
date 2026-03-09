import os
import pickle
from TradeX.utils.common.config_loader import get_logger
import pandas as pd

# ----------------------------
# Import DL training functions
# ----------------------------
from TradeX.ai.dl.models.arima import train as train_arima
from TradeX.ai.dl.models.varima import train as train_varima
from TradeX.ai.dl.models.nbeats import train as train_nbeats
from TradeX.ai.dl.models.transformer import train as train_transformer

logger = get_logger("dl_model_trainer")

# ----------------------------
# Map model names to train functions
# ----------------------------
DL_MODELS = {
    "arima": train_arima,
    "varima": train_varima,
    "nbeats": train_nbeats,
    "transformer": train_transformer
}

# ----------------------------
# Train DL Model
# ----------------------------
def train_model(model_type: str, model_name: str, df, df_1m=None,
                split_date="2024-01-01 00:00", **kwargs):
    """
    Train a DL model (ARIMA, VARIMA, NBEATS, Transformer).

    Args:
        model_type (str): Must be 'dl'
        model_name (str): Model name string
        df (pd.DataFrame): Time-series data
        df_1m (pd.DataFrame): Optional 1-min data for backtesting
        split_date (str): Train/test split
        **kwargs: Extra kwargs for the model

    Returns:
        model: Trained model object
        preds: Predictions object (Darts TimeSeries or DataFrame)
        test_index: Index of test set
        df_test: Test DataFrame (for covariates)
    """
    if model_type != "dl":
        raise ValueError("model_type must be 'dl'")

    trainer = DL_MODELS.get(model_name)
    if trainer is None:
        raise ValueError(f"Unknown DL model: {model_name}")

    model, preds, test_index, df_test = trainer(
        df=df,
        split_date=split_date,
        **kwargs
    )

    return model, preds, test_index, df_test

# ----------------------------
# Save DL Model
# ----------------------------
def save_model(model, feature_columns, symbol, model_name, folder="saved_models"):
    """
    Save trained DL model along with covariate feature names.

    Args:
        model: Trained model object
        feature_columns (list): Covariates/features used in training
        symbol (str): Symbol name (e.g., BTC)
        model_name (str): Model name string
        folder (str): Folder path to save
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{symbol}_{model_name}_dl.pkl")

    # Some Darts models may be PyTorch objects; pickle works if not on GPU tensors
    try:
        with open(file_path, "wb") as f:
            pickle.dump({
                "model": model,
                "features": feature_columns
            }, f)
        logger.info(f"Saved DL model {model_name} for {symbol} at {file_path}")
    except Exception as e:
        logger.error(f"Failed to save DL model {model_name}: {e}")