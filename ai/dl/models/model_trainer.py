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
    "high_performance",        # resource-scaling flag
})

_STATISTICAL_MODELS: frozenset[str] = frozenset({"arima", "varima"})

# ---------------------------------------------------------------------------
# Inline Optuna helper
# ---------------------------------------------------------------------------

def _inline_optuna_tune(
    model_name: str,
    df: pd.DataFrame,
    df_1m: pd.DataFrame | None,
    split_date: str,
    n_trials: int,
    high_performance: bool,
    take_profit: float = 2.0,
    stop_loss: float = 1.0,
) -> dict[str, Any]:
    """
    Run a quick Optuna search *inside* train_model (before the final fit)
    and return the best hyperparameter dict.

    This is intentionally lightweight (default 10 trials, in-memory storage,
    no SQLite overhead) so it adds minimal wall-clock time relative to the
    benefit of avoiding a badly-initialised model.

    Returns {} (empty dict) if optuna is not installed, df_1m is None, or
    any error occurs — the caller falls back to its original params.
    """
    if df_1m is None or df_1m.empty:
        logger.warning(
            f"[inline-optuna/{model_name}] df_1m not available — "
            "skipping inline tuning and using config params."
        )
        return {}

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning(
            "[inline-optuna] optuna not installed — skipping inline tuning."
        )
        return {}

    try:
        from TradeX.ai.dl.optuna_tuner import tune_model  # noqa: PLC0415
        best = tune_model(
            model_name=model_name,
            df=df,
            df_1m=df_1m,
            split_date=split_date,
            n_trials=n_trials,
            timeout=None,
            take_profit=take_profit,
            stop_loss=stop_loss,
            high_performance=high_performance,
            storage=None,          # in-memory: fast, no disk I/O
            show_progress_bar=False,
        )
        logger.info(
            f"[inline-optuna/{model_name}] best params after {n_trials} "
            f"trials: {best}"
        )
        return best
    except Exception as exc:
        logger.warning(
            f"[inline-optuna/{model_name}] tuning failed ({exc}); "
            "falling back to config params."
        )
        return {}


# ---------------------------------------------------------------------------
# Trainer dispatch
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
    high_performance: bool = True,
    # ── inline Optuna knobs ──────────────────────────────────────────────────
    run_inline_optuna: bool = True,
    inline_optuna_trials: int = 10,
    inline_optuna_take_profit: float = 2.0,
    inline_optuna_stop_loss: float = 1.0,
    **kwargs,
) -> tuple:
    """
    Dispatch to the correct model trainer and return its outputs.

    New behaviour (vs original):
    ────────────────────────────
    When *run_inline_optuna* is True (default) **and** df_1m is supplied,
    an Optuna mini-search is executed *before* the final model fit.  The
    best hyperparameters found in ``inline_optuna_trials`` trials are merged
    on top of *model_params*, giving the final fit the benefit of that search
    at the cost of ``inline_optuna_trials × train_time`` extra wall-clock
    seconds.

    Set ``run_inline_optuna=False`` (or omit df_1m) to skip it and behave
    exactly like the original implementation.

    Args:
        model_type               : Must be 'dl'.
        model_name               : One of 'arima', 'varima', 'nbeats', 'transformer'.
        df                       : Feature-engineered OHLCV DataFrame.
        df_1m                    : 1-minute OHLCV DataFrame (used for inline
                                   Optuna backtests; optional but recommended).
        split_date               : Train / test boundary (ISO date string).
        lookback                 : Lookback window override (input_chunk_length).
        epochs                   : Training epochs override.
        batch_size               : Mini-batch size override (None = model default).
        model_params             : Optional dict of model-specific kwargs from
                                   config.  Merged at LOWEST priority.
        high_performance         : Full 4-core resources when True; half when False.
        run_inline_optuna        : Run a quick Optuna search before the final fit.
                                   Default True.
        inline_optuna_trials     : Number of Optuna trials.  Default 10.
        inline_optuna_take_profit: BackTest take-profit used during tuning.
        inline_optuna_stop_loss  : BackTest stop-loss used during tuning.
        **kwargs                 : Extra kwargs forwarded verbatim to the trainer.

    Returns:
        (model, preds, test_index, df_test)
    """
    # BUG-3 FIX: fast-fail before any work is done.
    if model_type != "dl":
        raise ValueError(f"model_type must be 'dl', got '{model_type}'.")

    trainer = _get_trainer(model_name)

    # --- Build kwargs dict ------------------------------------------------
    # Priority (high → low):
    #   explicit named args > **kwargs > inline_optuna_best > model_params
    trainer_kwargs: dict[str, Any] = {}

    # 1. Lowest priority: model_params from config
    if model_params:
        if model_name in _STATISTICAL_MODELS:
            filtered = {k: v for k, v in model_params.items()
                        if k in _STATISTICAL_ALLOWED_PARAMS}
            trainer_kwargs.update(filtered)
        else:
            trainer_kwargs.update(model_params)

    # 2. Inline Optuna: run 10-trial search and overlay best params
    if run_inline_optuna:
        optuna_best = _inline_optuna_tune(
            model_name=model_name,
            df=df,
            df_1m=df_1m,
            split_date=split_date,
            n_trials=inline_optuna_trials,
            high_performance=high_performance,
            take_profit=inline_optuna_take_profit,
            stop_loss=inline_optuna_stop_loss,
        )
        if optuna_best:
            if model_name in _STATISTICAL_MODELS:
                filtered_best = {k: v for k, v in optuna_best.items()
                                 if k in _STATISTICAL_ALLOWED_PARAMS}
                trainer_kwargs.update(filtered_best)
            else:
                trainer_kwargs.update(optuna_best)
            logger.info(
                f"[inline-optuna/{model_name}] applied {len(optuna_best)} "
                "tuned params on top of config params."
            )

    # 3. Caller's **kwargs (higher priority than optuna + model_params)
    trainer_kwargs.update(kwargs)

    # Always pass split_date explicitly
    trainer_kwargs.pop("split_date", None)

    # Forward high_performance to every trainer
    trainer_kwargs["high_performance"] = high_performance

    # BUG-1 FIX: only set DL-specific keys when explicitly provided
    if model_name not in _STATISTICAL_MODELS:
        if lookback is not None:
            trainer_kwargs["lookback"] = lookback
        if epochs is not None:
            trainer_kwargs["epochs"] = epochs
        if batch_size is not None:
            trainer_kwargs["batch_size"] = batch_size
    else:
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