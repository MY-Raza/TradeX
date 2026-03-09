from darts.models import ARIMA
from TradeX.ai.dl.utils import prepare_series, train_test_split
import pandas as pd
import numpy as np


def train(
    df: pd.DataFrame,
    target_col: str = "close",
    split_date: str = "2024-01-01",
    p: int = 5,
    d: int = 1,
    q: int = 0,
    lookback: int = None,   # kept for interface parity; unused by ARIMA
    **kwargs,
):
    """
    Train an ARIMA model using Darts.

    Bug-fixes vs original:
    - `df.set_index(..., inplace=True)` on a caller-owned DataFrame mutates
      the object the caller still holds.  We work on an explicit copy instead.
    - `prepare_series` was called on the full df (including all indicator
      columns); only `target_col` is needed and passing the slim slice avoids
      building a large TimeSeries object unnecessarily.
    - ARIMA() with no arguments defaults to (1,1,0); explicit p/d/q params
      are now exposed so callers can tune without monkey-patching.
    - `split_before` now goes through `train_test_split` which validates that
      the split date is sensible (guards against silent empty-train errors).

    Performance:
    - Slicing to [target_col] before TimeSeries construction avoids serialising
      100+ indicator columns that ARIMA never reads.

    Args:
        df         : OHLCV (+ indicator) DataFrame.
        target_col : Column to forecast.
        split_date : ISO date string for the train/test boundary.
        p, d, q    : ARIMA order parameters.
        lookback   : Ignored; present for DL interface consistency.
        **kwargs   : Forwarded to darts ARIMA constructor.

    Returns:
        model      : Trained ARIMA model.
        preds      : Darts TimeSeries of predictions.
        test_index : Numeric array index of the test rows.
        df_test    : Empty DataFrame whose index matches the test period.
    """
    # --- 1. Prepare a clean, minimal copy ---------------------------------
    df = df.copy()

    if "datetime" in df.columns:
        df["datetime"] = (
            pd.to_datetime(df["datetime"], utc=True)
            .dt.tz_localize(None)   # strip tz → tz-naive UTC (Darts requirement)
        )
        df = df.set_index("datetime")

    # Keep only the target; everything else is irrelevant for ARIMA
    df_target = df[[target_col]]

    # Drop rows with NaN in target (indicator warm-up produces leading NaNs)
    df_target = df_target.dropna(subset=[target_col])

    # --- 2. Build Darts TimeSeries ----------------------------------------
    # reset_index() restores "datetime" as a plain tz-naive column; prepare_series handles it
    series = prepare_series(df_target.reset_index(), target_col)

    # --- 3. Train / test split (validated) --------------------------------
    train_series, test_series = train_test_split(series, split_date)

    # --- 4. Fit -----------------------------------------------------------
    model = ARIMA(p=p, d=d, q=q, **kwargs)
    model.fit(train_series)

    # --- 5. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 6. Build return artifacts ----------------------------------------
    n_train = len(train_series)
    n_test  = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test