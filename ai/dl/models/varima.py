"""
varima.py — Multivariate VARIMA trainer (Darts)
================================================
Key fix in this version:
- ROLLING WINDOW: statsmodels VAR complexity is O(n * p^2 * k^2).
  Fitting 22,000 hourly rows (2023-2025) with p=2, k=4 requires solving a
  700,000-element OLS system — this is why the model appears "stuck".
  A rolling window of the most recent N rows is now applied before fitting.
  VAR does not benefit from data older than a few months for crypto (regime
  changes make old data harmful, not helpful). Default: 6 months = 4,320 rows.

- q=0 GUARD: VARMA(q>0) is non-identifiable in statsmodels — triggers slow/
  non-convergent fitting. Any q>0 is forced to 0 with a warning.

- All previous optimisations preserved:
  slice-first copy, explicit freq, fast=True (OHLC only), trend='n' for d>=1.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import VARIMA

from TradeX.ai.dl.utils import train_test_split


_DEFAULT_TARGET_COLS: list[str] = ["open", "high", "low", "close", "volume"]
_FAST_TARGET_COLS:    list[str] = ["open", "high", "low", "close"]

# At 1h bars: 6 months ~ 4,320 rows. Enough for VAR to capture regime,
# small enough to solve in seconds instead of minutes.
_DEFAULT_ROLLING_ROWS = 4_320   # ~6 months of 1h data


def _detect_freq(index: pd.DatetimeIndex) -> str | None:
    """Infer freq string from last two entries. Avoids Darts' full index scan."""
    if len(index) < 2:
        return None
    _MAP = {
        pd.Timedelta("1min"):  "1min",
        pd.Timedelta("5min"):  "5min",
        pd.Timedelta("15min"): "15min",
        pd.Timedelta("30min"): "30min",
        pd.Timedelta("1h"):    "1h",
        pd.Timedelta("4h"):    "4h",
        pd.Timedelta("1D"):    "1D",
    }
    return _MAP.get(index[-1] - index[-2])


def train(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    split_date: str = "2024-01-01",
    p: int = 1,
    d: int = 1,
    q: int = 0,
    fast: bool = True,
    rolling_rows: int = _DEFAULT_ROLLING_ROWS,
    **kwargs,
) -> tuple:
    """
    Train a VARIMA model and return (model, preds, test_index, df_test).

    Args:
        df           : OHLCV (+ indicator) DataFrame.
        target_cols  : Columns to model jointly. Defaults to OHLC (fast=True)
                       or OHLCV (fast=False).
        split_date   : ISO date string for train/test boundary.
        p, d, q      : VARIMA order. q MUST be 0 (VARMA is non-identifiable).
        fast         : Drop 'volume', set trend='n' when d>=1. Default True.
        rolling_rows : Cap the training set to this many most-recent rows
                       before the split. Default 4320 (~6 months at 1h).
                       Set to 0 or None to disable (warning: very slow on
                       multi-year datasets).
        **kwargs     : Forwarded to darts VARIMA constructor.

    Returns:
        model, preds, test_index, df_test
    """
    # --- 0. q guard -------------------------------------------------------
    if q != 0:
        warnings.warn(
            f"VARIMA: q={q} requested but VARMA(q>0) is non-identifiable "
            f"(statsmodels will hang or fail to converge). Forcing q=0. "
            f"Set q=0 in config.yml to suppress this warning.",
            UserWarning, stacklevel=2,
        )
        q = 0

    # --- 1. Resolve target columns ----------------------------------------
    if target_cols is None:
        target_cols = list(_FAST_TARGET_COLS if fast else _DEFAULT_TARGET_COLS)
    else:
        target_cols = list(target_cols)

    # --- 2. Validate columns exist before copying -------------------------
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VARIMA: columns not found in df: {missing}")

    # --- 3. Slice to target cols FIRST (avoid copying 100+ indicator cols) -
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        dt_naive = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df_target = df[target_cols].copy()
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

    # --- 4. Sort + dropna on small frame ----------------------------------
    df_target = df_target.sort_index().dropna()

    if df_target.empty:
        raise ValueError("VARIMA: DataFrame is empty after dropping NaN rows.")

    all_nan_cols = [c for c in target_cols if df_target[c].isna().all()]
    if all_nan_cols:
        raise ValueError(f"VARIMA: columns entirely NaN after dropna: {all_nan_cols}")

    # --- 5. Split into train / test at split_date -------------------------
    split_ts = pd.Timestamp(split_date)
    if split_ts.tz is not None:
        split_ts = split_ts.tz_convert("UTC").tz_localize(None)

    df_train_full = df_target[df_target.index < split_ts]
    df_test_raw   = df_target[df_target.index >= split_ts]

    if df_train_full.empty:
        raise ValueError(f"VARIMA: no training rows before split_date '{split_date}'.")
    if df_test_raw.empty:
        raise ValueError(f"VARIMA: no test rows on/after split_date '{split_date}'.")

    # --- 6. Apply rolling window to training set --------------------------
    # KEY PERFORMANCE FIX: cap training rows so statsmodels OLS stays fast.
    if rolling_rows and len(df_train_full) > rolling_rows:
        n_dropped = len(df_train_full) - rolling_rows
        df_train = df_train_full.iloc[-rolling_rows:]
        import logging
        logging.getLogger("varima").info(
            f"VARIMA rolling window: using last {rolling_rows} of "
            f"{len(df_train_full)} training rows (dropped {n_dropped} old rows)."
        )
    else:
        df_train = df_train_full

    # --- 7. Minimum-row guard ---------------------------------------------
    n_vars   = len(target_cols)
    min_rows = p * n_vars + 2
    if len(df_train) < min_rows:
        raise ValueError(
            f"VARIMA({p},{d},{q}) with {n_vars} variables needs >= {min_rows} "
            f"rows in training window, got {len(df_train)}."
        )

    # --- 8. Build TimeSeries objects (explicit freq skips Darts index scan) -
    freq = _detect_freq(df_target.index)
    ts_kwargs = {"freq": freq} if freq is not None else {}

    train_series = TimeSeries.from_dataframe(df_train, **ts_kwargs)
    test_series  = TimeSeries.from_dataframe(df_test_raw, **ts_kwargs)

    # --- 9. Fit -----------------------------------------------------------
    if fast and d >= 1 and "trend" not in kwargs:
        kwargs["trend"] = "n"   # intercept is redundant after differencing

    model = VARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # --- 10. Predict ------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 11. Return artifacts ---------------------------------------------
    # test_index refers to positions in the ORIGINAL df (pre-window),
    # needed so prepare_predictions can look up df_gf rows by integer position.
    n_train_full = len(df_train_full)
    n_test       = len(test_series)
    test_index   = np.arange(n_train_full, n_train_full + n_test)
    df_test      = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test