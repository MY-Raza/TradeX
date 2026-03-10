"""
nbeats.py — N-BEATS trainer (Darts)
=====================================
Bug-fixes over original:

1. Same tz_localize TypeError — fixed with tz_convert pattern.

2. `output_chunk_length >= input_chunk_length` guard was correct, but the
   error message said "must be <" without mentioning equal-to; clarified.

3. When `epochs` alias is set to a value AND `n_epochs` is also in **kwargs,
   the alias silently overrides the kwarg.  Added a warning for this case so
   callers aren't confused.

4. `random_state=42` was hard-coded inside the function but also forwarded
   via **kwargs if the caller included it — causing a "duplicate keyword
   argument" TypeError.  Fixed by hoisting it to an explicit parameter with
   a default, then removing it from kwargs if present.

Performance:
- `pl_trainer_kwargs` suppresses Lightning progress bars; callers can
  override by passing their own dict.
- Model is constructed with `random_state` as an explicit kwarg so
  reproducibility is guaranteed without manual seeding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts.models import NBEATSModel

from TradeX.ai.dl.utils import prepare_series, train_test_split
from TradeX.utils.common.logs import get_logger

logger = get_logger("nbeats")


def train(
    df: pd.DataFrame,
    target_col: str = "close",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 48,
    output_chunk_length: int = 12,
    n_epochs: int = 50,
    batch_size: int = 32,
    random_state: int = 42,
    lookback: int | None = None,
    epochs: int | None = None,
    **kwargs,
) -> tuple:
    """
    Train an N-BEATS model and return (model, preds, test_index, df_test).

    Args:
        df                  : OHLCV (+ indicator) DataFrame.
        target_col          : Column to forecast (default 'close').
        split_date          : ISO date string for train/test boundary.
        input_chunk_length  : Lookback window fed to the model.
        output_chunk_length : Forecast horizon.
        n_epochs            : Training epochs.
        batch_size          : Mini-batch size.
        random_state        : RNG seed for reproducibility.
        lookback            : Alias → input_chunk_length (model_trainer compat).
        epochs              : Alias → n_epochs (model_trainer compat).
        **kwargs            : Forwarded to NBEATSModel constructor.

    Returns:
        model      : Trained NBEATSModel.
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

    # Prevent duplicate-kwarg TypeError if caller passes random_state in kwargs
    kwargs.pop("random_state", None)

    # --- Validate chunk lengths -------------------------------------------
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
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
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