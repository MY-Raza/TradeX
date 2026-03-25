from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd

from TradeX.utils.common.logs import get_logger

logger = get_logger("dl_model_trainer")

# Keys that statistical trainers (ARIMA/VARIMA) actually accept.
_STATISTICAL_ALLOWED_PARAMS = frozenset({
    "p", "d", "q", "rolling_rows",
    "signal_threshold",        # dead-band for trade signal generation
    "seasonal_order",          # ARIMA seasonal component
    "use_log_returns",         # VARIMA: model log-returns vs raw price
    "fast",                    # VARIMA: drop volume column
})

_STATISTICAL_MODELS: frozenset[str] = frozenset({"arima", "varima"})


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


def train_model(
    model_type: str,
    model_name: str,
    df: pd.DataFrame,
    df_1m: pd.DataFrame | None = None,
    split_date: str = "2024-01-01 00:00",
    lookback: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
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
        lookback     : Lookback window override (input_chunk_length).
        epochs       : Training epochs override.
        batch_size   : Mini-batch size override (None = use model default).
        model_params : Optional dict of model-specific kwargs from config.
                       These are merged at LOWEST priority so explicit
                       arguments always win.
        **kwargs     : Extra kwargs forwarded verbatim to the trainer.

    Returns:
        (model, preds, test_index, df_test)
    """
    # BUG-3 FIX: fast-fail before any work is done.
    if model_type != "dl":
        raise ValueError(f"model_type must be 'dl', got '{model_type}'.")

    trainer = _get_trainer(model_name)

    # --- Build kwargs dict ------------------------------------------------
    # Priority (high → low): explicit named args > **kwargs > model_params
    trainer_kwargs: dict[str, Any] = {}

    # Merge model_params at lowest priority
    if model_params:
        if model_name in _STATISTICAL_MODELS:
            # BUG-4 FIX: only forward params that statistical trainers accept.
            filtered = {k: v for k, v in model_params.items()
                        if k in _STATISTICAL_ALLOWED_PARAMS}
            trainer_kwargs.update(filtered)
        else:
            trainer_kwargs.update(model_params)

    # Merge caller's **kwargs (higher priority than model_params)
    trainer_kwargs.update(kwargs)

    # Always pass split_date explicitly (avoids duplicate-kwarg from **kwargs)
    trainer_kwargs.pop("split_date", None)

    # BUG-1 FIX: only set DL-specific keys when the caller explicitly provided
    # them (not None). This prevents the explicit-arg values from overriding
    # the same keys that model_params may have set with better model-specific
    # defaults (e.g. nbeats_params.n_epochs=20 vs training.epochs=50).
    if model_name not in _STATISTICAL_MODELS:
        if lookback is not None:
            trainer_kwargs["lookback"] = lookback
        if epochs is not None:
            trainer_kwargs["epochs"] = epochs
        if batch_size is not None:
            trainer_kwargs["batch_size"] = batch_size
    else:
        # Strip any DL-only keys that slipped in via **kwargs
        for key in ("lookback", "epochs", "batch_size",
                    "n_epochs", "input_chunk_length",
                    "output_chunk_length", "num_blocks",
                    "num_layers", "layer_widths"):
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

    Note: feature_columns should be the covariate/indicator columns only —
    callers should filter out raw OHLCV and target columns before passing.
    Raises on failure.

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
        raise