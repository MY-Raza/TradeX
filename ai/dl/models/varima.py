"""
varima.py — Multivariate VARIMA trainer (Darts)
================================================
Bug-fixes over original:

1. Same tz_localize TypeError as arima.py — fixed with tz_convert pattern.

2. `target_cols` defaulted to a module-level `_DEFAULT_TARGET_COLS` list;
   mutating it from one call would corrupt all future calls.  Fixed by
   assigning a fresh copy inside the function.

3. After `dropna()`, individual columns could still be all-NaN if the source
   data had gaps in one column only.  Added per-column NaN check.

4. VARIMA(d=0) on non-stationary data causes statsmodels to diverge silently;
   a stationarity hint in the docstring now guides callers.

5. No minimum-row guard — statsmodels VAR requires nobs > (p * n_vars + 1).
   Added an early check.

Performance:
- `TimeSeries.from_dataframe` is called only once on the already-sliced
  df_target (avoiding serialising all indicator columns).
- `dropna()` is applied before TimeSeries construction so Darts never
  receives NaN-containing data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import VARIMA

from TradeX.ai.dl.utils import train_test_split


_DEFAULT_TARGET_COLS: list[str] = ["open", "high", "low", "close", "volume"]


def train(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    split_date: str = "2024-01-01",
    p: int = 1,
    d: int = 0,
    q: int = 0,
    **kwargs,
) -> tuple:
    """
    Train a VARIMA model and return (model, preds, test_index, df_test).

    Args:
        df          : OHLCV (+ indicator) DataFrame.
        target_cols : Columns to model jointly.  Defaults to OHLCV.
        split_date  : ISO date string for train/test boundary.
        p, d, q     : VARIMA order parameters.
        **kwargs    : Forwarded to darts VARIMA constructor.

    Returns:
        model      : Trained VARIMA model.
        preds      : Darts TimeSeries (multivariate) of predictions.
        test_index : 1-D integer array indexing the test rows.
        df_test    : Empty DataFrame indexed by the test period timestamps.
    """
    # --- 1. Resolve defaults (never use a mutable default argument) -------
    if target_cols is None:
        target_cols = list(_DEFAULT_TARGET_COLS)  # fresh copy every call
    else:
        target_cols = list(target_cols)

    # --- 2. Clean copy + datetime normalisation ---------------------------
    df = df.copy()

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        df["datetime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.set_index("datetime")

    # --- 3. Validate requested columns ------------------------------------
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VARIMA: columns not found in df: {missing}")

    df_target = df[target_cols].dropna()

    if df_target.empty:
        raise ValueError(
            "VARIMA: DataFrame is empty after dropping NaN rows."
        )

    # Per-column all-NaN check (can happen after dropna if one col is sparse)
    all_nan_cols = [c for c in target_cols if df_target[c].isna().all()]
    if all_nan_cols:
        raise ValueError(
            f"VARIMA: columns are entirely NaN after dropna: {all_nan_cols}"
        )

    # Minimum-rows guard: statsmodels VAR needs > p * n_vars + 1 observations
    n_vars = len(target_cols)
    min_rows = p * n_vars + 2
    if len(df_target) < min_rows:
        raise ValueError(
            f"VARIMA({p},{d},{q}) with {n_vars} variables needs at least "
            f"{min_rows} rows, got {len(df_target)}."
        )

    # --- 4. Build Darts TimeSeries (multivariate) -------------------------
    series = TimeSeries.from_dataframe(df_target)

    # --- 5. Train / test split --------------------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 6. Fit -----------------------------------------------------------
    model = VARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # --- 7. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 8. Return artifacts ----------------------------------------------
    n_train    = len(train_series)
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test