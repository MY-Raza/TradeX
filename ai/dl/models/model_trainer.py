"""
model_trainer.py — DL model dispatch & persistence
====================================================
Bug-fixes over original:

1. `get_logger` was imported from `config_loader` in the original code, which
   does not export it.  Correct import is from `TradeX.utils.common.logs`.
   (Already fixed in the provided version; preserved here.)

2. `train_model` passed `split_date` as a positional then also via **kwargs
   when callers included it — causing "duplicate keyword argument" TypeError.
   `split_date` is now always extracted from kwargs before the trainer call.

3. Model-specific params (arima_params, nbeats_params, etc.) from config.yml
   were never read and forwarded by `train_model`.  The dispatcher now accepts
   an optional `model_params` dict and merges it into trainer_kwargs so
   callers can pass config values without monkey-patching.

4. `save_model` swallowed exceptions (original); the provided version already
   re-raises — preserved.

5. `DL_MODELS` dict was built at import time, which means circular-import
   errors surface as confusing AttributeErrors.  Moved to a lazy lookup
   function `_get_trainer` so import failures are caught at call-time with a
   clear message.

Performance:
- Trainer kwargs dict is built once per call, not rebuilt inside branches.
- Logging level changed from INFO to DEBUG for per-step noise; ERROR kept
  for failures.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd

from TradeX.utils.common.logs import get_logger

logger = get_logger("dl_model_trainer")


# ---------------------------------------------------------------------------
# Lazy model registry — avoids circular imports at module load time
# ---------------------------------------------------------------------------

def _get_trainer(model_name: str):
    """Return the train function for *model_name*, importing lazily."""
    if model_name == "arima":
        from TradeX.ai.dl.models.arima import train
        return train
    if model_name == "varima":
        from TradeX.ai.dl.models.varima import train
        return train
    if model_name == "nbeats":
        from TradeX.ai.dl.models.nbeats import train
        return train
    if model_name == "transformer":
        from TradeX.ai.dl.models.transformer import train
        return train
    raise ValueError(
        f"Unknown DL model: '{model_name}'. "
        f"Available: arima, varima, nbeats, transformer."
    )


_STATISTICAL_MODELS: frozenset[str] = frozenset({"arima", "varima"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(
    model_type: str,
    model_name: str,
    df: pd.DataFrame,
    df_1m: pd.DataFrame | None = None,
    split_date: str = "2024-01-01 00:00",
    lookback: int | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    model_params: dict[str, Any] | None = None,
    **kwargs,
) -> tuple:
    """
    Dispatch to the correct model trainer and return its outputs.

    Args:
        model_type   : Must be 'dl'.
        model_name   : One of 'arima', 'varima', 'nbeats', 'transformer'.
        df           : Feature-engineered OHLCV DataFrame.
        df_1m        : 1-minute OHLCV DataFrame (for backtesting, unused here).
        split_date   : Train / test boundary (ISO date string).
        lookback     : Lookback window for DL models (input_chunk_length).
        epochs       : Training epochs for DL models.
        batch_size   : Mini-batch size for DL models.
        model_params : Optional dict of model-specific kwargs from config
                       (e.g. arima_params, nbeats_params).  These are merged
                       into trainer kwargs at lowest priority so explicit
                       arguments always win.
        **kwargs     : Extra kwargs forwarded verbatim to the trainer.

    Returns:
        (model, preds, test_index, df_test)
    """
    if model_type != "dl":
        raise ValueError(f"model_type must be 'dl', got '{model_type}'.")

    trainer = _get_trainer(model_name)

    # --- Build kwargs dict ------------------------------------------------
    # Priority (high → low): explicit args > **kwargs > model_params
    trainer_kwargs: dict[str, Any] = {}

    # Merge model_params at lowest priority
    if model_params:
        trainer_kwargs.update(model_params)

    # Merge caller's **kwargs (higher priority than model_params)
    trainer_kwargs.update(kwargs)

    # Always pass split_date explicitly (avoids duplicate-kwarg from **kwargs)
    trainer_kwargs.pop("split_date", None)  # remove if caller put it in kwargs

    if model_name not in _STATISTICAL_MODELS:
        if lookback is not None:
            trainer_kwargs["lookback"] = lookback
        trainer_kwargs["epochs"]     = epochs
        trainer_kwargs["batch_size"] = batch_size
    else:
        # Strip DL-only keys that statistical trainers don't accept
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

    except Exception as exc:
        logger.error(f"Training failed for DL model '{model_name}': {exc}")
        raise


def save_model(
    model: Any,
    feature_columns: list[str],
    symbol: str,
    model_name: str,
    folder: str = "saved_models",
) -> None:
    """
    Persist a trained DL model and its feature list to disk.

    Raises on failure so the caller can decide how to handle (e.g. skip DB
    writes rather than proceeding with an unsaved model).

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
        with open(file_path, "wb") as fh:
            pickle.dump(
                {"model": model, "features": feature_columns},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(
            f"Saved DL model '{model_name}' for '{symbol}' → {file_path}"
        )

    except Exception as exc:
        logger.error(f"Failed to save DL model '{model_name}': {exc}")
        raise   # caller must know about save failures