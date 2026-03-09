from darts.models import NBEATSModel
from TradeX.ai.dl.utils import prepare_series, train_test_split
import pandas as pd
import numpy as np


def train(
    df: pd.DataFrame,
    target_col: str = "close",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 48,
    output_chunk_length: int = 12,
    n_epochs: int = 50,
    batch_size: int = 32,
    lookback: int = None,   # kept for interface parity; maps to input_chunk_length if set
    epochs: int = None,     # kept for interface parity; maps to n_epochs if set
    **kwargs,
):
    """
    Train an N-BEATS model using Darts.

    Bug-fixes vs original:
    - `df.set_index(..., inplace=True)` mutated the caller's DataFrame.
      We now work on an explicit copy.
    - `model_trainer` passes `lookback` and `epochs` (its own names) but
      NBEATSModel expects `input_chunk_length` and `n_epochs`.  Added an
      explicit mapping so neither kwarg is silently dropped.
    - Leading NaN rows from indicator warm-up caused Darts to raise inside
      fit(); they are now stripped before the series is built.
    - `train_test_split` now validates the split date instead of letting
      Darts raise an opaque error.

    Performance:
    - Only `target_col` is passed to `prepare_series`, avoiding construction
      of a large multi-column TimeSeries that NBEATS ignores.
    - `pl_trainer_kwargs` defaults are set to suppress unnecessary progress
      bars in batch runs (overridable via **kwargs).

    Args:
        df                  : OHLCV (+ indicator) DataFrame.
        target_col          : Column to forecast.
        split_date          : ISO date string for train/test boundary.
        input_chunk_length  : Lookback window (past steps fed to the model).
        output_chunk_length : Forecast horizon.
        n_epochs            : Training epochs.
        batch_size          : Mini-batch size.
        lookback            : Alias for input_chunk_length (model_trainer compat).
        epochs              : Alias for n_epochs (model_trainer compat).
        **kwargs            : Forwarded to NBEATSModel constructor.

    Returns:
        model      : Trained NBEATSModel.
        preds      : Darts TimeSeries of predictions.
        test_index : Numeric array index of the test rows.
        df_test    : Empty DataFrame whose index matches the test period.
    """
    # Resolve interface-level aliases ------------------------------------------
    if lookback is not None:
        input_chunk_length = lookback
    if epochs is not None:
        n_epochs = epochs

    # Validate chunk lengths ---------------------------------------------------
    if output_chunk_length >= input_chunk_length:
        raise ValueError(
            f"output_chunk_length ({output_chunk_length}) must be < "
            f"input_chunk_length ({input_chunk_length})."
        )

    # --- 1. Clean copy --------------------------------------------------------
    df = df.copy()

    if "datetime" in df.columns:
        df["datetime"] = (
            pd.to_datetime(df["datetime"], utc=True)
            .dt.tz_localize(None)   # strip tz → tz-naive UTC (Darts requirement)
        )
        df = df.set_index("datetime")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df_target = df[[target_col]].dropna()

    # Need at least input_chunk_length + output_chunk_length rows to train
    min_rows = input_chunk_length + output_chunk_length
    if len(df_target) < min_rows:
        raise ValueError(
            f"Not enough data ({len(df_target)} rows) for "
            f"input_chunk_length={input_chunk_length} + "
            f"output_chunk_length={output_chunk_length}."
        )

    # --- 2. Build Darts TimeSeries --------------------------------------------
    series = prepare_series(df_target.reset_index(), target_col)

    # --- 3. Train / test split (validated) -----------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 4. Suppress verbose Lightning output unless caller asks for it ------
    pl_trainer_kwargs = kwargs.pop(
        "pl_trainer_kwargs",
        {"enable_progress_bar": False, "enable_model_summary": False},
    )

    # --- 5. Fit --------------------------------------------------------------
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=42,
        pl_trainer_kwargs=pl_trainer_kwargs,
        **kwargs,
    )
    model.fit(train_series)

    # --- 6. Predict ----------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 7. Build return artifacts --------------------------------------------
    n_train    = len(train_series)
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test