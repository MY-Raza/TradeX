import os
import pickle
from TradeX.utils.common.logs import get_logger   # BUG-FIX: original imported from wrong module
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
# Map model names → train functions
# ----------------------------
DL_MODELS = {
    "arima":       train_arima,
    "varima":      train_varima,
    "nbeats":      train_nbeats,
    "transformer": train_transformer,
}

# Models whose trainers do NOT accept lookback / epochs / batch_size
_STATISTICAL_MODELS = {"arima", "varima"}


# ----------------------------
# Train DL Model
# ----------------------------
def train_model(
    model_type: str,
    model_name: str,
    df: pd.DataFrame,
    df_1m: pd.DataFrame = None,
    split_date: str = "2024-01-01 00:00",
    lookback: int = None,
    epochs: int = 50,
    batch_size: int = 32,
    **kwargs,
):
    """
    Dispatch to the correct model trainer and return its outputs.

    Bug-fixes vs original:
    - `get_logger` was imported from `config_loader` which does not export it;
      the correct source is `TradeX.utils.common.logs`.
    - Statistical models (ARIMA, VARIMA) don't accept `lookback`, `epochs`,
      or `batch_size`.  The original code only guarded `lookback` but still
      leaked `epochs` and `batch_size` via **kwargs on the trainer call —
      those trainers use **kwargs, so the extra keys were silently absorbed
      but could cause unexpected behaviour in future Darts versions.
      Now all three are stripped for statistical models.
    - The trainer was called with `df=df` but the df may still have its
      datetime as a column (not the index).  Each individual trainer now
      owns that normalisation, so no double-normalisation occurs.

    Performance:
    - `kwargs.copy()` was always made even for statistical models that ignore
      it; we still copy (safe practice) but only enrich for DL models.

    Args:
        model_type  : Must be 'dl'.
        model_name  : One of 'arima', 'varima', 'nbeats', 'transformer'.
        df          : Feature-engineered OHLCV DataFrame.
        df_1m       : 1-minute OHLCV DataFrame (for backtesting, unused here).
        split_date  : Train / test boundary.
        lookback    : Lookback window for DL models (input_chunk_length).
        epochs      : Training epochs for DL models.
        batch_size  : Mini-batch size for DL models.
        **kwargs    : Forwarded verbatim to the model trainer.

    Returns:
        model, preds, test_index, df_test
    """
    if model_type != "dl":
        raise ValueError(f"model_type must be 'dl', got '{model_type}'.")

    trainer = DL_MODELS.get(model_name)
    if trainer is None:
        raise ValueError(
            f"Unknown DL model: '{model_name}'. "
            f"Available: {list(DL_MODELS.keys())}"
        )

    trainer_kwargs = kwargs.copy()

    if model_name not in _STATISTICAL_MODELS:
        # DL models: pass training hyper-parameters
        if lookback is not None:
            trainer_kwargs["lookback"] = lookback
        trainer_kwargs["epochs"]     = epochs
        trainer_kwargs["batch_size"] = batch_size
    else:
        # Statistical models: strip DL-specific keys that would be silently
        # absorbed by **kwargs and could shadow legitimate model params.
        for key in ("lookback", "epochs", "batch_size"):
            trainer_kwargs.pop(key, None)

    try:
        model, preds, test_index, df_test = trainer(
            df=df,
            split_date=split_date,
            **trainer_kwargs,
        )
        logger.info(f"DL model '{model_name}' trained successfully.")
        return model, preds, test_index, df_test

    except Exception as e:
        logger.error(f"Training failed for DL model '{model_name}': {e}")
        raise


# ----------------------------
# Save DL Model
# ----------------------------
def save_model(
    model,
    feature_columns: list,
    symbol: str,
    model_name: str,
    folder: str = "saved_models",
):
    """
    Persist a trained DL model and its covariate feature list to disk.

    Bug-fix:
    - The original swallowed the exception after logging it, meaning the
      caller could not know a save had failed.  We now re-raise so the
      caller in main.py can react (e.g. skip downstream DB writes).

    Performance:
    - `pickle.HIGHEST_PROTOCOL` is already used — no change needed there.

    Args:
        model           : Trained model object.
        feature_columns : List of feature/covariate column names.
        symbol          : Ticker symbol (e.g. 'btc').
        model_name      : Model identifier string.
        folder          : Directory to write the pickle file.
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{symbol}_{model_name}_dl.pkl")

    try:
        with open(file_path, "wb") as f:
            pickle.dump(
                {"model": model, "features": feature_columns},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"Saved DL model '{model_name}' for '{symbol}' → {file_path}")

    except Exception as e:
        logger.error(f"Failed to save DL model '{model_name}': {e}")
        raise   # BUG-FIX: caller must know about save failures