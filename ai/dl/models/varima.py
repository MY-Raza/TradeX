from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import VARIMA

from TradeX.ai.dl.models.trainer_utils import (
    normalise_datetime, ensure_log_return, rolling_train_test_split,
    check_min_rows, make_test_artifacts,
)
_DEFAULT_TARGET_COLS: list[str] = ["open", "high", "low", "close", "volume"]
_FAST_TARGET_COLS:    list[str] = ["open", "high", "low", "close"]

# SIGNAL-1: log-return versions of the target columns for stationarity.
_LOG_RETURN_COLS: list[str] = [
    "open_lr", "high_lr", "low_lr", "close_lr"
]

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
    p: int = 2,
    d: int = 0,                         # SIGNAL-2: was 1; log-returns are I(0)
    q: int = 0,
    fast: bool = True,
    use_log_returns: bool = True,        # SIGNAL-1: model log-returns of OHLC
    rolling_rows: int = _DEFAULT_ROLLING_ROWS,
    signal_threshold: float = 3e-4,     # SIGNAL-3: dead-band on close_lr pred
    **kwargs,
) -> tuple:
    """
    Train a VARIMA model and return (model, preds, test_index, df_test).

    Args:
        df               : OHLCV (+ indicator) DataFrame.
        target_cols      : Columns to model jointly. When use_log_returns=True
                           (default), log-returns are computed automatically.
        split_date       : ISO date string for train/test boundary.
        p, d, q          : VARIMA order. q MUST be 0.
        fast             : Drop 'volume', use OHLC (or their log-returns).
        use_log_returns  : If True (default), derive log-return columns from
                           OHLC and model those instead of raw price levels.
                           This makes the series stationary so d=0 is correct.
        rolling_rows     : Cap training set size. Default 4320 (~6 months 1h).
        signal_threshold : |close_lr prediction| must exceed this to trade.
        **kwargs         : Forwarded to darts VARIMA constructor.

    Returns:
        model, preds, test_index, df_test
    """
    # --- 0. q guard -------------------------------------------------------
    if q != 0:
        import warnings
        warnings.warn(
            f"VARIMA: q={q} requested but VARMA(q>0) is non-identifiable "
            f"(statsmodels will hang or fail to converge). Forcing q=0.",
            UserWarning, stacklevel=2,
        )
        q = 0

    # --- 1. Datetime normalisation ----------------------------------------
    df = normalise_datetime(df)

    # --- 2. Derive log-return columns if requested ------------------------
    if use_log_returns:
        base_cols = list(_FAST_TARGET_COLS if fast else _DEFAULT_TARGET_COLS[:4])
        for col in base_cols:
            if col not in df.columns:
                raise ValueError(f"VARIMA: column '{col}' not found in df.")
        # Delegate to ensure_log_return: handles positive-only (log-diff) and
        # columns with zero/negative values (signed-log-diff) automatically.
        df = ensure_log_return(df, columns=base_cols)
        target_cols = [f"{c}_lr" for c in base_cols]
    else:
        # Fall back to raw price columns
        if target_cols is None:
            target_cols = list(_FAST_TARGET_COLS if fast else _DEFAULT_TARGET_COLS)
        target_cols = list(target_cols)

    # --- 3. Validate columns exist ----------------------------------------
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VARIMA: columns not found in df: {missing}")

    # --- 4. Sort + dropna on target cols only -----------------------------
    df_target = df[target_cols].sort_index().dropna()

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

    # --- 7. Build TimeSeries objects --------------------------------------
    freq = _detect_freq(df_target.index)
    ts_kwargs = {"freq": freq} if freq is not None else {}

    n_train_full = len(df_target[df_target.index < pd.Timestamp(split_date)])

    train_series = TimeSeries.from_dataframe(df_train, **ts_kwargs)
    test_series  = TimeSeries.from_dataframe(df_test_raw, **ts_kwargs)

    # --- 8. Fit -----------------------------------------------------------
    # SIGNAL-2: with log-returns (d=0), trend='n' is always appropriate.
    if use_log_returns and "trend" not in kwargs:
        kwargs["trend"] = "n"
    elif fast and d >= 1 and "trend" not in kwargs:
        kwargs["trend"] = "n"

    model = VARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # SIGNAL-3: attach threshold for downstream signal filtering.
    model.signal_threshold = signal_threshold

    # --- 9. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 10. Return artifacts ---------------------------------------------
    test_index, df_test = make_test_artifacts(n_train_full, test_series)
    return model, preds, test_index, df_test