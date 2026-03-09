from darts.models import VARIMA
from darts import TimeSeries
from TradeX.ai.dl.utils import train_test_split
import pandas as pd
import numpy as np


# Columns that VARIMA operates on by default
_DEFAULT_TARGET_COLS = ["open", "high", "low", "close", "volume"]


def train(
    df: pd.DataFrame,
    target_cols: list[str] = None,
    split_date: str = "2024-01-01",
    p: int = 1,
    d: int = 0,
    q: int = 0,
    **kwargs,
):
    """
    Train a VARIMA model using Darts.

    Bug-fixes vs original:
    - `df.set_index(..., inplace=True)` mutated the caller's DataFrame.
      We now work on an explicit copy.
    - `target_cols` defaulted to a mutable list literal in the function
      signature — a classic Python footgun.  Replaced with `None` + guard.
    - No validation that every requested `target_col` actually exists in df;
      a missing column produced a cryptic Darts error deep in the stack.
    - `split_before` now goes through `train_test_split` which validates that
      the split date is sensible.
    - Leading NaN rows (from indicator warm-up) crashed VARIMA; they are now
      dropped before fitting.

    Performance:
    - Only the required columns are passed to `TimeSeries.from_dataframe`,
      avoiding serialisation of all indicator columns.
    - `dropna` is applied once up-front rather than letting Darts raise.

    Args:
        df          : OHLCV (+ indicator) DataFrame.
        target_cols : Columns to include in the multivariate series.
                      Defaults to OHLCV.
        split_date  : ISO date string for train/test boundary.
        p, d, q     : VARIMA order.
        **kwargs    : Forwarded to darts VARIMA constructor.

    Returns:
        model      : Trained VARIMA model.
        preds      : Darts TimeSeries of predictions (multivariate).
        test_index : Numeric array index of the test rows.
        df_test    : Empty DataFrame whose index matches the test period.
    """
    if target_cols is None:
        target_cols = _DEFAULT_TARGET_COLS

    # --- 1. Clean copy ---------------------------------------------------
    df = df.copy()

    if "datetime" in df.columns:
        df["datetime"] = (
            pd.to_datetime(df["datetime"], utc=True)
            .dt.tz_localize(None)   # strip tz → tz-naive UTC (Darts requirement)
        )
        df = df.set_index("datetime")

    # Validate requested columns exist
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VARIMA: columns not found in df: {missing}")

    df_target = df[target_cols].dropna()

    if df_target.empty:
        raise ValueError("VARIMA: DataFrame is empty after dropping NaN rows.")

    # --- 2. Build Darts TimeSeries (multivariate) ------------------------
    series = TimeSeries.from_dataframe(df_target)

    # --- 3. Train / test split (validated) --------------------------------
    train_series, test_series = train_test_split(series, split_date)
    # --- 4. Fit -----------------------------------------------------------
    model = VARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # --- 5. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 6. Build return artifacts ----------------------------------------
    n_train    = len(train_series)
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test