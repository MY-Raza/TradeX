from darts.models import TransformerModel
from TradeX.ai.dl.utils import prepare_series, train_test_split
import pandas as pd
import numpy as np


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
    lookback: int = None,   # model_trainer alias → input_chunk_length
    epochs: int = None,     # model_trainer alias → n_epochs
    **kwargs,
):
    """
    Train a Transformer model using Darts.

    Bug-fixes vs original:
    - `df.set_index(..., inplace=True)` mutated the caller's DataFrame.
      We now work on an explicit copy.
    - `model_trainer` passes `lookback` and `epochs`, but TransformerModel
      expects `input_chunk_length` and `n_epochs`.  Added an explicit mapping.
    - d_model must be divisible by nhead; added a guard that raises early
      instead of crashing deep inside PyTorch with an opaque error.
    - Leading NaN rows from indicator warm-up caused Darts to raise inside
      fit(); they are now stripped before the series is built.
    - `train_test_split` now validates the split date.

    Performance:
    - Only `target_col` is passed to `prepare_series`, skipping construction
      of a massive multi-column TimeSeries.
    - `pl_trainer_kwargs` suppresses Lightning progress bars in batch runs
      (saves seconds per epoch when running many experiments).

    Args:
        df                  : OHLCV (+ indicator) DataFrame.
        target_col          : Column to forecast.
        split_date          : ISO date string for train/test boundary.
        input_chunk_length  : Lookback window.
        output_chunk_length : Forecast horizon.
        d_model             : Transformer embedding dimension.
        nhead               : Number of attention heads (must divide d_model).
        num_encoder_layers  : Transformer encoder depth.
        num_decoder_layers  : Transformer decoder depth.
        n_epochs            : Training epochs.
        batch_size          : Mini-batch size.
        lookback            : Alias for input_chunk_length (model_trainer compat).
        epochs              : Alias for n_epochs (model_trainer compat).
        **kwargs            : Forwarded to TransformerModel constructor.

    Returns:
        model      : Trained TransformerModel.
        preds      : Darts TimeSeries of predictions.
        test_index : Numeric array index of the test rows.
        df_test    : Empty DataFrame whose index matches the test period.
    """
    # Resolve interface-level aliases ------------------------------------------
    if lookback is not None:
        input_chunk_length = lookback
    if epochs is not None:
        n_epochs = epochs

    # --- Guard: nhead must divide d_model ------------------------------------
    if d_model % nhead != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"Adjust one of them so d_model % nhead == 0."
        )

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

    # --- 4. Suppress verbose Lightning output unless caller overrides --------
    pl_trainer_kwargs = kwargs.pop(
        "pl_trainer_kwargs",
        {"enable_progress_bar": False, "enable_model_summary": False},
    )

    # --- 5. Fit --------------------------------------------------------------
    model = TransformerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=42,
        pl_trainer_kwargs=pl_trainer_kwargs,
        **kwargs,
    )
    model.fit(train_series)

    # --- 6. Predict ----------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 7. Build return artifacts -------------------------------------------
    n_train    = len(train_series)
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test