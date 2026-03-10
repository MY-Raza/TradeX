"""
arima.py — Univariate ARIMA trainer (Darts)
=============================================
Bug-fixes over original:

1. `pd.to_datetime(..., utc=True).dt.tz_localize(None)` raises TypeError when
   the Series is already tz-aware (tz_localize cannot re-localize).  Fixed by
   using tz_convert → tz_localize(None) pattern (same as utils.py).

2. `dropna(subset=[target_col])` only works when target_col is a column, not
   when it's the index.  Since we set the index *before* calling dropna, the
   target is in df_target as a column (because we slice df[[target_col]]),
   so the original was fine — but we add an explicit `.dropna()` call on the
   single-column frame for clarity.

3. No minimum-length guard: if the series has fewer rows than the ARIMA
   order, Darts raises an opaque statsmodels error.  Added an early check.

Performance:
- Raw arrays are never constructed; ARIMA only needs the Darts TimeSeries.
- df_target is sliced to a single column before index construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts.models import ARIMA

from TradeX.ai.dl.utils import prepare_series, train_test_split


def train(
    df: pd.DataFrame,
    target_col: str = "close",
    split_date: str = "2024-01-01",
    p: int = 5,
    d: int = 1,
    q: int = 0,
    lookback: int = None,   # interface-parity only; unused by ARIMA
    **kwargs,
) -> tuple:
    """
    Train an ARIMA model and return (model, preds, test_index, df_test).

    Args:
        df         : OHLCV (+ indicator) DataFrame.
        target_col : Column to forecast (default 'close').
        split_date : ISO date string for train/test boundary.
        p, d, q    : ARIMA order parameters.
        lookback   : Ignored; kept for DL interface parity.
        **kwargs   : Forwarded to darts ARIMA constructor.

    Returns:
        model      : Trained ARIMA model.
        preds      : Darts TimeSeries of in-sample predictions.
        test_index : 1-D integer array indexing the test rows.
        df_test    : Empty DataFrame indexed by the test period timestamps.
    """
    # --- 1. Validate inputs -----------------------------------------------
    if target_col not in df.columns and "datetime" not in df.columns:
        # Maybe target_col is the index — handled below; warn if genuinely missing
        if target_col not in df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in DataFrame columns: "
                f"{list(df.columns)}"
            )

    # --- 2. Prepare a clean, minimal copy ---------------------------------
    df = df.copy()

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        df["datetime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.set_index("datetime")

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found after index reset. "
            f"Available: {list(df.columns)}"
        )

    # Keep only the target column; drop NaN warm-up rows
    df_target = df[[target_col]].dropna()

    # --- 3. Minimum-length guard ------------------------------------------
    min_rows = p + d + 1
    if len(df_target) < min_rows:
        raise ValueError(
            f"ARIMA({p},{d},{q}): need at least {min_rows} rows, "
            f"got {len(df_target)}."
        )

    # --- 4. Build Darts TimeSeries ----------------------------------------
    series = prepare_series(df_target.reset_index(), target_col)

    # --- 5. Train / test split --------------------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 6. Fit -----------------------------------------------------------
    model = ARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # --- 7. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 8. Return artifacts ----------------------------------------------
    n_train    = len(train_series)
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test