from __future__ import annotations

import numpy as np
import pandas as pd
from darts.models import ARIMA

from TradeX.ai.dl.utils import prepare_series, train_test_split
from TradeX.ai.dl.models.trainer_utils import normalise_datetime, check_min_rows, make_test_artifacts


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
    # --- 1. Datetime normalisation ----------------------------------------
    df = normalise_datetime(df)

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    # --- 2. Slice target column and drop NaN warm-up rows -----------------
    df_target = df[[target_col]].dropna()

    # --- 3. Minimum-length guard ------------------------------------------
    check_min_rows(
        df_target,
        min_rows=p + d + 1,
        context=f"ARIMA({p},{d},{q})",
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
    test_index, df_test = make_test_artifacts(len(train_series), test_series)
    return model, preds, test_index, df_test