"""
varima.py — Multivariate VARIMA trainer (Darts)
================================================
Performance improvements over previous version:

1. COPY SCOPE: `df.copy()` was copying the entire feature-engineered DataFrame
   (100+ columns, years of hourly rows) before slicing to target_cols.
   Now we slice FIRST, then copy only the 5-column target frame — ~20x less
   memory allocated and copied.

2. FREQ INFERENCE: `TimeSeries.from_dataframe()` without `freq` scans the
   entire DatetimeIndex to infer frequency. We detect it once from the sliced
   df_target and pass it explicitly, skipping the scan.

3. DROPNA SCOPE: sort + dropna applied once on the small frame.

4. VARIMA COLUMN COUNT: Each additional target column multiplies the number of
   parameters to fit (p * n_vars^2 in the VAR component). fast=True drops
   'volume' (typically non-stationary, adds noise) reducing 5->4 columns and
   cutting parameter count by 36%.

5. STATSMODELS TREND: Default trend='c' fits an intercept per variable. For
   differenced (d>=1) series this is redundant. We default trend='n' when
   d >= 1 and fast=True.

Bug-fixes preserved:
- tz_convert pattern (no tz_localize TypeError)
- mutable default argument guard
- per-column all-NaN check
- minimum-row guard
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import VARIMA

from TradeX.ai.dl.utils import train_test_split


_DEFAULT_TARGET_COLS: list[str] = ["open", "high", "low", "close", "volume"]
_FAST_TARGET_COLS:    list[str] = ["open", "high", "low", "close"]


def _detect_freq(index: pd.DatetimeIndex) -> str | None:
    """
    Infer freq string from the last two index entries.
    Avoids Darts scanning the full index. Returns None on failure (safe).
    """
    if len(index) < 2:
        return None
    delta = index[-1] - index[-2]
    _MAP = {
        pd.Timedelta("1min"):  "1min",
        pd.Timedelta("5min"):  "5min",
        pd.Timedelta("15min"): "15min",
        pd.Timedelta("30min"): "30min",
        pd.Timedelta("1h"):    "1h",
        pd.Timedelta("4h"):    "4h",
        pd.Timedelta("1D"):    "1D",
    }
    return _MAP.get(delta)


def train(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    split_date: str = "2024-01-01",
    p: int = 1,
    d: int = 0,
    q: int = 0,
    fast: bool = True,
    **kwargs,
) -> tuple:
    """
    Train a VARIMA model and return (model, preds, test_index, df_test).

    Args:
        df          : OHLCV (+ indicator) DataFrame.
        target_cols : Columns to model jointly.  Defaults to OHLC (fast=True)
                      or OHLCV (fast=False).
        split_date  : ISO date string for train/test boundary.
        p, d, q     : VARIMA order parameters.
        fast        : Drop 'volume', set trend='n' when d>=1. Default True.
        **kwargs    : Forwarded to darts VARIMA constructor.

    Returns:
        model, preds, test_index, df_test
    """
    # --- 1. Resolve target columns ----------------------------------------
    if target_cols is None:
        target_cols = list(_FAST_TARGET_COLS if fast else _DEFAULT_TARGET_COLS)
    else:
        target_cols = list(target_cols)

    # --- 2. Validate columns exist before any copying ---------------------
    available = set(df.columns)
    missing = [c for c in target_cols if c not in available]
    if missing:
        raise ValueError(f"VARIMA: columns not found in df: {missing}")

    # --- 3. Slice to target cols FIRST, then normalise datetime -----------
    # This avoids copying 100+ indicator columns that VARIMA never reads.
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        dt_naive = dt.dt.tz_convert("UTC").dt.tz_localize(None)

        df_target = df[target_cols].copy()           # copy only 4-5 columns
        df_target.index = pd.DatetimeIndex(dt_naive.values, name="datetime")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "VARIMA: DataFrame must have a DatetimeIndex or 'datetime' column."
            )
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df_target = df[target_cols].copy()
        df_target.index = idx

    # --- 4. Sort + dropna on the small frame (single pass) ----------------
    df_target = df_target.sort_index().dropna()

    if df_target.empty:
        raise ValueError("VARIMA: DataFrame is empty after dropping NaN rows.")

    all_nan_cols = [c for c in target_cols if df_target[c].isna().all()]
    if all_nan_cols:
        raise ValueError(
            f"VARIMA: columns entirely NaN after dropna: {all_nan_cols}"
        )

    # --- 5. Minimum-row guard ---------------------------------------------
    n_vars   = len(target_cols)
    min_rows = p * n_vars + 2
    if len(df_target) < min_rows:
        raise ValueError(
            f"VARIMA({p},{d},{q}) with {n_vars} variables needs >= {min_rows} "
            f"rows, got {len(df_target)}."
        )

    # --- 6. Build TimeSeries — explicit freq skips Darts index scan -------
    freq = _detect_freq(df_target.index)
    ts_kwargs = {"freq": freq} if freq is not None else {}
    series = TimeSeries.from_dataframe(df_target, **ts_kwargs)

    # --- 7. Train / test split --------------------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 8. Fit -----------------------------------------------------------
    if fast and d >= 1 and "trend" not in kwargs:
        kwargs["trend"] = "n"   # no intercept needed after differencing

    model = VARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # --- 9. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 10. Return artifacts ---------------------------------------------
    n_train    = len(train_series)
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test