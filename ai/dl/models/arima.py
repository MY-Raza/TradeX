from __future__ import annotations

import numpy as np
import pandas as pd
from darts.models import ARIMA

from TradeX.ai.dl.utils import prepare_series, train_test_split
from TradeX.ai.dl.models.trainer_utils import (
    normalise_datetime, ensure_log_return, check_min_rows, make_test_artifacts,
)


def train(
    df: pd.DataFrame,
    target_col: str = "log_return",    # SIGNAL-1: was "close"
    split_date: str = "2024-01-01",
    p: int = 5,
    d: int = 0,                        # SIGNAL-3: was 1; log_return is I(0)
    q: int = 0,
    seasonal_order: tuple = (1, 0, 0, 24),  # SIGNAL-4: daily 24h seasonal AR
    signal_threshold: float = 3e-4,    # SIGNAL-2: dead-band on log_return
    high_performance: bool = True,     # accepted for interface parity; ARIMA has no heavy resources
    lookback: int = None,
    **kwargs,
) -> tuple:
    """
    Train an ARIMA model and return (model, preds, test_index, df_test).

    Args:
        df               : OHLCV (+ indicator) DataFrame.
        target_col       : Column to forecast (default 'log_return').
        split_date       : ISO date string for train/test boundary.
        p, d, q          : ARIMA order parameters.
        seasonal_order   : (P, D, Q, m) seasonal ARIMA order. Default (1,0,0,24)
                           captures the 24h daily BTC cycle on 1h data.
                           Pass (0,0,0,0) to disable seasonal component.
        signal_threshold : Minimum |log_return| prediction to generate a trade.
        lookback         : Ignored; kept for DL interface parity.
        **kwargs         : Forwarded to darts ARIMA constructor.

    Returns:
        model      : Trained ARIMA model (with .signal_threshold attribute).
        preds      : Darts TimeSeries of predictions.
        test_index : 1-D integer array indexing the test rows.
        df_test    : Empty DataFrame indexed by the test period timestamps.
    """
    # --- 1. Datetime normalisation ----------------------------------------
    df = normalise_datetime(df)

    # --- 2. Ensure log_return exists if needed ----------------------------
    if target_col == "log_return":
        df = ensure_log_return(df)

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    # --- 3. Slice target column and drop NaN warm-up rows -----------------
    df_target = df[[target_col]].dropna()

    # --- 4. Minimum-length guard ------------------------------------------
    check_min_rows(
        df_target,
        min_rows=p + d + 1,
        context=f"ARIMA({p},{d},{q})",
    )

    # --- 5. Build Darts TimeSeries ----------------------------------------
    series = prepare_series(df_target.reset_index(), target_col)

    # --- 6. Train / test split --------------------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 7. Build ARIMA kwargs: inject seasonal_order if non-trivial ------
    arima_kwargs = dict(kwargs)
    P, D, Q, m = seasonal_order
    if m > 1 and (P > 0 or D > 0 or Q > 0):
        arima_kwargs.setdefault("seasonal_order", (P, D, Q, m))

    # --- 8. Fit -----------------------------------------------------------
    model = ARIMA(p=p, d=d, q=q, **arima_kwargs)
    model.fit(train_series)

    # SIGNAL-2: store threshold for downstream signal filtering.
    model.signal_threshold = signal_threshold

    # --- 9. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 10. Return artifacts ---------------------------------------------
    from TradeX.ai.dl.models.trainer_utils import make_test_artifacts
    split_idx = len(df_target[df_target.index < train_series.end_time()])
    test_index, df_test = make_test_artifacts(split_idx, test_series, n_full=len(df_target))
    return model, preds, test_index, df_test