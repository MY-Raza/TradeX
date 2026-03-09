from darts.models import VARIMA
from darts import TimeSeries
import pandas as pd
import numpy as np


def train(
    df: pd.DataFrame,
    target_cols: list = ["open", "high", "low", "close", "volume"],
    split_date: str = "2024-01-01",
    **kwargs
):
    """
    Train a VARIMA model using Darts.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with datetime column
        target_cols (list): Columns to include in multivariate forecast
        split_date (str): Date to split train/test sets
        **kwargs: Extra args for VARIMA

    Returns:
        model: Trained VARIMA model
        preds: Predictions (Darts TimeSeries)
        test_index: Numeric index of test set
        df_test: Test DataFrame (empty, for interface consistency)
    """
    # Ensure datetime column is UTC-aware
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)

    # Convert to Darts TimeSeries
    series = TimeSeries.from_dataframe(df, value_cols=target_cols)

    # Split train/test
    train_series, test_series = series.split_before(pd.Timestamp(split_date))

    # Initialize and train VARIMA
    model = VARIMA(**kwargs)
    model.fit(train_series)

    # Make predictions
    preds = model.predict(len(test_series))

    # Create numeric test index for backtesting
    test_index = np.arange(len(train_series), len(train_series) + len(test_series))
    df_test = pd.DataFrame(index=test_series.time_index)  # empty covariates

    return model, preds, test_index, df_test