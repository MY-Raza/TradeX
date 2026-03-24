from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from darts.models import NBEATSModel

from TradeX.ai.dl.utils import prepare_series, train_test_split
from TradeX.utils.common.logs import get_logger

logger = get_logger("nbeats")

_DEFAULT_ROLLING_ROWS = 4_320   # ~6 months at 1h


def _set_cpu_threads() -> None:
    """Use all physical cores for PyTorch CPU ops."""
    try:
        import torch, os
        n = int(os.environ.get("OMP_NUM_THREADS", 0)) or os.cpu_count() or 1
        torch.set_num_threads(n)
        logger.info(f"N-BEATS: PyTorch using {n} CPU threads.")
    except Exception:
        pass


def train(
    df: pd.DataFrame,
    target_col: str = "log_return",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 24,
    output_chunk_length: int = 1,
    n_epochs: int = 20,
    batch_size: int = 64,
    num_blocks: int = 2,
    num_layers: int = 2,
    layer_widths: int = 128,
    random_state: int = 42,
    rolling_rows: int = _DEFAULT_ROLLING_ROWS,
    lookback: int | None = None,
    epochs: int | None = None,
    **kwargs,
) -> tuple:
    """
    Train an N-BEATS model (CPU-optimised) and return
    (model, preds, test_index, df_test).
    """
    _set_cpu_threads()

    # --- Resolve aliases --------------------------------------------------
    if lookback is not None:
        input_chunk_length = lookback
    if epochs is not None:
        if "n_epochs" in kwargs:
            logger.warning(
                "Both 'epochs' alias and 'n_epochs' in kwargs; "
                "'epochs' alias takes precedence."
            )
        n_epochs = epochs

    # Prevent duplicate-kwarg TypeError
    kwargs.pop("random_state", None)

    # --- Chunk length guard -----------------------------------------------
    if output_chunk_length >= input_chunk_length:
        raise ValueError(
            f"output_chunk_length ({output_chunk_length}) must be strictly "
            f"less than input_chunk_length ({input_chunk_length})."
        )

    # --- 1. Datetime normalisation ----------------------------------------

    if target_col == "log_return" and "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"]).diff()

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

    df_target = df[[target_col]].dropna().sort_index()

    # --- 2. Rolling window ------------------------------------------------
    split_ts = pd.Timestamp(split_date)
    if split_ts.tz is not None:
        split_ts = split_ts.tz_convert("UTC").tz_localize(None)

    df_before_split = df_target[df_target.index < split_ts]
    df_after_split  = df_target[df_target.index >= split_ts]

    if df_before_split.empty:
        raise ValueError(f"No training rows before split_date '{split_date}'.")
    if df_after_split.empty:
        raise ValueError(f"No test rows on/after split_date '{split_date}'.")

    if rolling_rows and len(df_before_split) > rolling_rows:
        # BUG-1 FIX: capture n_dropped BEFORE slicing.
        n_total  = len(df_before_split)
        n_dropped = n_total - rolling_rows
        df_before_split = df_before_split.iloc[-rolling_rows:]
        logger.info(
            f"N-BEATS rolling window: using last {rolling_rows} of "
            f"{n_total} training rows (dropped {n_dropped} older rows)."
        )

    # --- 3. Minimum-row guard ---------------------------------------------
    min_rows = input_chunk_length + output_chunk_length
    if len(df_before_split) < min_rows:
        raise ValueError(
            f"Not enough training data ({len(df_before_split)} rows) for "
            f"input_chunk_length={input_chunk_length} + "
            f"output_chunk_length={output_chunk_length} (need {min_rows})."
        )

    # BUG-2 FIX: deduplicate and sort to prevent Darts index-duplicate error.
    df_windowed = (
        pd.concat([df_before_split, df_after_split])
        .loc[~pd.concat([df_before_split, df_after_split]).index.duplicated(keep="last")]
        .sort_index()
    )

    # --- 4. Build Darts TimeSeries ----------------------------------------
    series = prepare_series(df_windowed.reset_index(), target_col)

    # --- 5. Train / test split --------------------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 6. CPU-optimised trainer kwargs ----------------------------------
    # BUG-4 FIX: copy the popped dict to avoid mutating the caller's kwargs.
    pl_trainer_kwargs = kwargs.pop("pl_trainer_kwargs", {}).copy()
    pl_trainer_kwargs.setdefault("accelerator",          "cpu")
    pl_trainer_kwargs.setdefault("enable_progress_bar",  False)
    pl_trainer_kwargs.setdefault("enable_model_summary", False)
    pl_trainer_kwargs.setdefault("log_every_n_steps",    10)

    # BUG-5 FIX: add EarlyStopping so training halts when converged.
    try:
        from pytorch_lightning.callbacks import EarlyStopping
        cb = pl_trainer_kwargs.pop("callbacks", [])
        cb.append(EarlyStopping(
            monitor="train_loss", patience=3, min_delta=1e-4, mode="min"
        ))
        pl_trainer_kwargs["callbacks"] = cb
    except ImportError:
        pass

    # --- 7. Fit -----------------------------------------------------------
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=random_state,
        pl_trainer_kwargs=pl_trainer_kwargs,
        **kwargs,
    )
    model.fit(train_series)

    # --- 8. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 9. Return artifacts ----------------------------------------------
    # BUG-3 FIX: test_index is relative to df_windowed (what preds aligns to),
    # NOT to the full df_target.  main.py uses test_index to slice df_gf, so
    # using the full-df offset caused silent wrong-row selection when a rolling
    # window was applied.
    n_train_windowed = len(df_before_split)
    n_test           = len(test_series)
    test_index       = np.arange(n_train_windowed, n_train_windowed + n_test)
    df_test          = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test