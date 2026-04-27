from __future__ import annotations

import numpy as np
import pandas as pd
from darts.models import NBEATSModel

from TradeX.ai.darts.utils import prepare_series, train_test_split
from TradeX.ai.darts.models.trainer_utils import normalise_datetime,ensure_log_return,rolling_train_test_split,check_min_rows,concat_dedup_sort,build_pl_trainer_kwargs,make_test_artifacts
from TradeX.utils.common.logs import get_logger

logger = get_logger("nbeats")

_DEFAULT_ROLLING_ROWS = 4_320   # ~6 months at 1h


def _set_cpu_threads(n_threads: int) -> None:
    """Pin PyTorch to *n_threads* CPU threads."""
    try:
        import torch, os
        torch.set_num_threads(n_threads)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        logger.info(f"N-BEATS: PyTorch using {n_threads} CPU threads.")
    except Exception:
        pass


def train(
    df: pd.DataFrame,
    target_col: str = "log_return",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 72,      # SIGNAL-3: was 24; 3 days of 1h context
    output_chunk_length: int = 4,      # SIGNAL-2: was 1; predict 4h forward
    n_epochs: int = 50,                # SIGNAL-4: was 20; more training
    batch_size: int = 64,
    num_blocks: int = 3,               # SIGNAL-5: was 2
    num_layers: int = 2,
    layer_widths: int = 256,           # SIGNAL-5: was 128
    random_state: int = 42,
    rolling_rows: int = _DEFAULT_ROLLING_ROWS,
    signal_threshold: float = 3e-4,   # SIGNAL-1: dead-band on log_return (~0.03%)
    high_performance: bool = True,     # True = 4 cores / full resources; False = 2 cores / half
    lookback: int | None = None,
    epochs: int | None = None,
    **kwargs,
) -> tuple:
    """
    Train an N-BEATS model (CPU-optimised) and return
    (model, preds, test_index, df_test).
    """
    # --- Resource scaling -------------------------------------------------
    # high_performance=True  → 4 cores, full epochs/batch/capacity
    # high_performance=False → 2 cores, halved epochs/batch/capacity
    n_threads = 4 if high_performance else 2
    _set_cpu_threads(n_threads)

    if not high_performance:
        n_epochs      = max(1, n_epochs      // 2)   # 50 → 25
        batch_size    = max(1, batch_size    // 2)   # 64 → 32
        rolling_rows  = max(1, rolling_rows  // 2)   # 4320 → 2160
        num_blocks    = max(1, num_blocks    - 1)    # 3 → 2
        layer_widths  = max(32, layer_widths // 2)   # 256 → 128
        logger.info(
            "N-BEATS: high_performance=False — using 2 threads, "
            f"n_epochs={n_epochs}, batch_size={batch_size}, "
            f"rolling_rows={rolling_rows}, num_blocks={num_blocks}, "
            f"layer_widths={layer_widths}."
        )
    else:
        logger.info("N-BEATS: high_performance=True — using full 4-core resources.")

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

    kwargs.pop("random_state", None)

    # --- Chunk length guard -----------------------------------------------
    if output_chunk_length >= input_chunk_length:
        raise ValueError(
            f"output_chunk_length ({output_chunk_length}) must be strictly "
            f"less than input_chunk_length ({input_chunk_length})."
        )

    # --- 1. Datetime normalisation + log_return ---------------------------
    df = normalise_datetime(df)
    if target_col == "log_return":
        df = ensure_log_return(df)

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found. Available: {list(df.columns)}"
        )

    df_target = df[[target_col]].dropna().sort_index()

    # Absolute position of first test row in the FULL df_target (before rolling cap).
    # Used by make_test_artifacts so test_index is consistent across all models.
    _split_ts   = pd.Timestamp(split_date)
    if _split_ts.tz is not None:
        _split_ts = _split_ts.tz_convert("UTC").tz_localize(None)
    _n_train_abs = int((df_target.index < _split_ts).sum())

    # --- 2. Rolling train / test split ------------------------------------
    df_train, df_test_raw = rolling_train_test_split(
        df_target,
        split_date=split_date,
        rolling_rows=rolling_rows,
        logger=logger,
        label="N-BEATS",
    )

    # --- 3. Minimum-row guard ---------------------------------------------
    check_min_rows(
        df_train,
        min_rows=input_chunk_length + output_chunk_length,
        context=f"N-BEATS input+output={input_chunk_length + output_chunk_length}",
    )

    # --- 4. Build Darts TimeSeries ----------------------------------------
    df_windowed = concat_dedup_sort(df_train, df_test_raw)
    series = prepare_series(df_windowed.reset_index(), target_col)

    # --- 5. Train / test split (Darts) ------------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 6. Build pl_trainer_kwargs ---------------------------------------
    pl_trainer_kwargs = build_pl_trainer_kwargs(
        base_kwargs=kwargs.pop("pl_trainer_kwargs", {}),
        use_early_stopping=True,
        monitor="train_loss",
        patience=5,                    # SIGNAL-4: was 3; allow fuller convergence
        high_performance=high_performance,
    )

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
    print(train_series)

    # SIGNAL-1: attach threshold so downstream prepare_predictions can apply it.
    model.signal_threshold = signal_threshold
    logger.info(f"N-BEATS signal_threshold set to {signal_threshold:.2e}")

    # --- 8. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 9. Return artifacts ----------------------------------------------
    test_index, df_test = make_test_artifacts(_n_train_abs, test_series, n_full=len(df_target))
    return model, preds, test_index, df_test