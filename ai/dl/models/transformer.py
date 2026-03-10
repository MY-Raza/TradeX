"""
transformer.py — Transformer trainer (Darts)
==============================================
Bug-fixes over original:

1. Same tz_localize TypeError — fixed with tz_convert pattern.

2. `random_state=42` hard-coded in the constructor body would raise
   "duplicate keyword argument" if the caller also passed random_state in
   **kwargs.  Promoted to an explicit parameter.

3. Same `epochs` / `n_epochs` alias-collision as nbeats.py — added warning.

4. `d_model % nhead != 0` guard was correct; no change needed.

5. No guard for `num_encoder_layers` / `num_decoder_layers` being 0 or
   negative.  Added a minimum-value check (PyTorch throws a cryptic error
   deep in MultiheadAttention otherwise).

Performance:
- Lightning progress bars suppressed by default (overridable via
  pl_trainer_kwargs kwarg).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts.models import TransformerModel

from TradeX.ai.dl.utils import prepare_series, train_test_split
from TradeX.utils.common.logs import get_logger

logger = get_logger("transformer")


def train(
    df: pd.DataFrame,
    target_col: str = "close",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 48,
    output_chunk_length: int = 12,
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    n_epochs: int = 50,
    batch_size: int = 32,
    random_state: int = 42,
    lookback: int | None = None,
    epochs: int | None = None,
    **kwargs,
) -> tuple:
    """
    Train a Transformer model and return (model, preds, test_index, df_test).

    Args:
        df                  : OHLCV (+ indicator) DataFrame.
        target_col          : Column to forecast (default 'close').
        split_date          : ISO date string for train/test boundary.
        input_chunk_length  : Lookback window.
        output_chunk_length : Forecast horizon.
        d_model             : Embedding dimension (must be divisible by nhead).
        nhead               : Number of attention heads.
        num_encoder_layers  : Encoder depth (≥1).
        num_decoder_layers  : Decoder depth (≥1).
        n_epochs            : Training epochs.
        batch_size          : Mini-batch size.
        random_state        : RNG seed.
        lookback            : Alias → input_chunk_length (model_trainer compat).
        epochs              : Alias → n_epochs (model_trainer compat).
        **kwargs            : Forwarded to TransformerModel constructor.

    Returns:
        model      : Trained TransformerModel.
        preds      : Darts TimeSeries of predictions.
        test_index : 1-D integer array indexing the test rows.
        df_test    : Empty DataFrame indexed by the test period timestamps.
    """
    # --- Resolve interface aliases ----------------------------------------
    if lookback is not None:
        input_chunk_length = lookback
    if epochs is not None:
        if "n_epochs" in kwargs:
            logger.warning(
                "Both 'epochs' alias and 'n_epochs' in kwargs supplied; "
                "'epochs' alias takes precedence."
            )
        n_epochs = epochs

    # Prevent duplicate-kwarg TypeError
    kwargs.pop("random_state", None)

    # --- Architecture guards ----------------------------------------------
    if d_model % nhead != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"Adjust so d_model % nhead == 0."
        )
    if num_encoder_layers < 1 or num_decoder_layers < 1:
        raise ValueError(
            f"num_encoder_layers and num_decoder_layers must each be ≥ 1 "
            f"(got {num_encoder_layers}, {num_decoder_layers})."
        )
    if output_chunk_length >= input_chunk_length:
        raise ValueError(
            f"output_chunk_length ({output_chunk_length}) must be strictly "
            f"less than input_chunk_length ({input_chunk_length})."
        )

    # --- 1. Clean copy + datetime normalisation ---------------------------
    df = df.copy()

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        df["datetime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.set_index("datetime")

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found. Available: {list(df.columns)}"
        )

    df_target = df[[target_col]].dropna()

    # --- Minimum-row guard ------------------------------------------------
    min_rows = input_chunk_length + output_chunk_length
    if len(df_target) < min_rows:
        raise ValueError(
            f"Not enough data ({len(df_target)} rows) for "
            f"input_chunk_length={input_chunk_length} + "
            f"output_chunk_length={output_chunk_length} (need {min_rows})."
        )

    # --- 2. Build Darts TimeSeries ----------------------------------------
    series = prepare_series(df_target.reset_index(), target_col)

    # --- 3. Train / test split --------------------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 4. Suppress Lightning verbosity (overridable) --------------------
    pl_trainer_kwargs = kwargs.pop(
        "pl_trainer_kwargs",
        {"enable_progress_bar": False, "enable_model_summary": False},
    )

    # --- 5. Fit -----------------------------------------------------------
    model = TransformerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=random_state,
        pl_trainer_kwargs=pl_trainer_kwargs,
        **kwargs,
    )
    model.fit(train_series)

    # --- 6. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 7. Return artifacts ----------------------------------------------
    n_train    = len(train_series)
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test