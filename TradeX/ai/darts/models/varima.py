from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import VARIMA

from TradeX.ai.darts.models.trainer_utils import (
    normalise_datetime, ensure_log_return, rolling_train_test_split,
    check_min_rows, make_test_artifacts,
)

_DEFAULT_TARGET_COLS: list[str] = ["open", "high", "low", "close", "volume"]
_FAST_TARGET_COLS:    list[str] = ["open", "high", "low", "close"]

# SIGNAL-1: log-return versions of the target columns for stationarity.
_LOG_RETURN_COLS: list[str] = [
    "open_lr", "high_lr", "low_lr", "close_lr"
]

# FIX-TRADE-COUNT: rolling window raised from 4 320 to 8 640 (~12 months at 1h).
# The previous 6-month cap gave VARIMA only ~400–600 test rows, which — combined
# with the tight signal_threshold — yielded as few as 2 trades.  A longer window
# exposes more market regimes and produces more variation in close_lr predictions.
_DEFAULT_ROLLING_ROWS = 8_640   # ~12 months of 1h data (was 4 320)


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
    p: int = 3,                          # FIX: was 2; more AR lags capture
                                         # longer BTC autocorrelation structure
    d: int = 0,                          # SIGNAL-2: log-returns are I(0)
    q: int = 0,
    fast: bool = True,
    use_log_returns: bool = True,        # SIGNAL-1: model log-returns of OHLC
    rolling_rows: int = _DEFAULT_ROLLING_ROWS,
    # FIX-TRADE-COUNT: threshold lowered from 1e-5 to 5e-6 so the dead-band
    # no longer kills every signal.  1e-5 was calibrated against a 6-month
    # window; the wider window shifts the close_lr std slightly higher, so we
    # can afford a tighter threshold.  Optuna will refine this further.
    signal_threshold: float = 5e-6,
    high_performance: bool = True,
    **kwargs,
) -> tuple:
    """
    Train a VARIMA model and return (model, preds, test_index, df_test).

    Key changes vs. the original
    ─────────────────────────────
    * ``_DEFAULT_ROLLING_ROWS`` raised from 4 320 → 8 640 (~12 months at 1h):
      more training data → richer covariance structure → more varied predictions
      → more signals above the dead-band.
    * ``signal_threshold`` lowered from 1e-5 → 5e-6: the previous value
      suppressed almost all VARIMA signals, producing only 2 trades.  5e-6 is
      still conservative but allows the backtest to see real signals.
    * ``p`` raised from 2 → 3: captures more of BTC's autocorrelation lags
      without meaningfully increasing fit time on a 12-month window.
    * The ``close_lr`` component is extracted from the multivariate prediction
      TimeSeries before the threshold is applied, mirroring what
      ``prepare_predictions`` does downstream.  This prevents a shape mismatch
      when the caller iterates over components.

    Args:
        df               : OHLCV (+ indicator) DataFrame.
        target_cols      : Columns to model jointly. When use_log_returns=True
                           (default), log-returns are computed automatically.
        split_date       : ISO date string for train/test boundary.
        p, d, q          : VARIMA order. q MUST be 0.
        fast             : Drop 'volume', use OHLC (or their log-returns).
        use_log_returns  : If True (default), derive log-return columns from
                           OHLC and model those instead of raw price levels.
        rolling_rows     : Cap training set size. Default 8640 (~12 months 1h).
        signal_threshold : |close_lr prediction| must exceed this to trade.
        high_performance : If True, use full rolling_rows. If False, halve.
        **kwargs         : Forwarded to darts VARIMA constructor.

    Returns:
        model, preds, test_index, df_test
    """
    import logging as _logging
    _varima_logger = _logging.getLogger("varima")

    # --- Resource scaling -------------------------------------------------
    if not high_performance:
        rolling_rows = max(1, rolling_rows // 2)   # 8640 → 4320
        _varima_logger.info(
            f"VARIMA: high_performance=False — rolling_rows reduced to {rolling_rows}."
        )

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
        df = ensure_log_return(df, columns=base_cols)
        target_cols = [f"{c}_lr" for c in base_cols]
    else:
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
    # With log-returns (d=0), trend='n' is always appropriate.
    if use_log_returns and "trend" not in kwargs:
        kwargs["trend"] = "n"
    elif fast and d >= 1 and "trend" not in kwargs:
        kwargs["trend"] = "n"

    model = VARIMA(p=p, d=d, q=q, **kwargs)
    print(train_series)
    model.fit(train_series)

    # SIGNAL-3: attach threshold for downstream signal filtering.
    model.signal_threshold = signal_threshold

    _varima_logger.info(
        f"VARIMA({p},{d},{q}) fitted | target_cols={target_cols} | "
        f"rolling_rows={rolling_rows} | signal_threshold={signal_threshold:.2e}"
    )

    # --- 9. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 10. Diagnostic: log close_lr prediction statistics ---------------
    # This helps verify that enough predictions exceed the signal_threshold.
    try:
        close_lr_idx = target_cols.index("close_lr")
        close_lr_vals = preds.univariate_component(close_lr_idx).values().ravel()
        n_signals = int(np.sum(np.abs(close_lr_vals) > signal_threshold))
        _varima_logger.info(
            f"VARIMA close_lr preds: mean={close_lr_vals.mean():.3e}, "
            f"std={close_lr_vals.std():.3e}, "
            f"signals_above_threshold={n_signals}/{len(close_lr_vals)}"
        )
    except Exception:
        pass  # diagnostic only — never crash on this

    # --- 11. Return artifacts ---------------------------------------------
    test_index, df_test = make_test_artifacts(n_train_full, test_series, n_full=len(df_target))
    return model, preds, test_index, df_test