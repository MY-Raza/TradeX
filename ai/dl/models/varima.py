"""
varima.py — Multivariate VARIMA trainer (Darts)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import VARIMA

from TradeX.ai.dl.models.trainer_utils import normalise_datetime,rolling_train_test_split,check_min_rows,make_test_artifacts
_DEFAULT_TARGET_COLS: list[str] = ["open", "high", "low", "close", "volume"]
_FAST_TARGET_COLS:    list[str] = ["open", "high", "low", "close"]

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

    # --- 2. Validate columns exist before normalisation -------------------
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VARIMA: columns not found in df: {missing}")

    # --- 3. Datetime normalisation on target columns only -----------------
    # Slice first to avoid copying 100+ indicator columns unnecessarily.
    if "datetime" in df.columns:
        df_target = df[target_cols].copy()
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        df_target.index = pd.DatetimeIndex(
            dt.dt.tz_convert("UTC").dt.tz_localize(None).values, name="datetime"
        )
    else:
        df_slim = df[target_cols].copy()
        df_target = normalise_datetime(df_slim, copy=False)

    # --- 4. Sort + dropna -------------------------------------------------
    df_target = df_target.sort_index().dropna()

    if df_target.empty:
        raise ValueError("VARIMA: DataFrame is empty after dropping NaN rows.")

    all_nan_cols = [c for c in target_cols if df_target[c].isna().all()]
    if all_nan_cols:
        raise ValueError(f"VARIMA: columns entirely NaN after dropna: {all_nan_cols}")

    # --- 5. Rolling train / test split ------------------------------------
    df_train, df_test_raw = rolling_train_test_split(
        df_target,
        split_date=split_date,
        rolling_rows=rolling_rows,
        label="VARIMA",
    )

    # --- 6. Minimum-row guard ---------------------------------------------
    n_vars = len(target_cols)
    check_min_rows(
        df_train,
        min_rows=p * n_vars + 2,
        context=f"VARIMA({p},{d},{q}) with {n_vars} variables",
    )

    # --- 7. Build TimeSeries objects (explicit freq skips Darts index scan) -
    freq = _detect_freq(df_target.index)
    ts_kwargs = {"freq": freq} if freq is not None else {}

    # Store the full training length BEFORE the window for correct test_index.
    n_train_full = len(df_target[df_target.index < pd.Timestamp(split_date)])

    train_series = TimeSeries.from_dataframe(df_train, **ts_kwargs)
    test_series  = TimeSeries.from_dataframe(df_test_raw, **ts_kwargs)

    # --- 8. Fit -----------------------------------------------------------
    if fast and d >= 1 and "trend" not in kwargs:
        kwargs["trend"] = "n"

    model = VARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # --- 9. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 10. Return artifacts ---------------------------------------------
    # test_index refers to positions in the ORIGINAL df (pre-window) so that
    # prepare_predictions can look up df_gf rows by integer position.
    test_index, df_test = make_test_artifacts(n_train_full, test_series)
    return model, preds, test_index, df_test