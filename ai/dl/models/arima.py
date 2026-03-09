from darts.models import ARIMA
from TradeX.ai.dl.utils import prepare_series, train_test_split
import pandas as pd
import numpy as np


def train(
    df: pd.DataFrame,
    target_col: str = "close",
    split_date: str = "2024-01-01",
    lookback: int = None,  # optional for DL interface consistency
):
    """
    Train an ARIMA model using Darts.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with datetime index or column
        target_col (str): Target column to forecast
        split_date (str): Date to split train/test sets
        lookback (int, optional): Ignored for ARIMA but kept for DL interface

    Returns:
        model: Trained ARIMA model
        preds: Predictions (Darts TimeSeries)
        test_index: Index of test set
        df_test: Test DataFrame (empty, for interface consistency)
    """
    # Ensure datetime column is tz-aware
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)

    # Prepare Darts TimeSeries
    series = prepare_series(df, target_col)

    # Split train/test
    train_series, test_series = train_test_split(series, split_date)

    # Initialize and train ARIMA
    model = ARIMA()
    model.fit(train_series)

    # Make predictions
    preds = model.predict(len(test_series))

    # Return index of test set and empty df_test for consistency
    test_index = np.arange(len(train_series), len(train_series) + len(test_series))
    df_test = pd.DataFrame(index=test_series.time_index)  # empty covariates

    return model, preds, test_index, df_test