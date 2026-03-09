import pandas as pd
from darts import TimeSeries


def prepare_series(df, target_col="close"):
    """
    Convert dataframe to Darts TimeSeries
    """
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    series = TimeSeries.from_dataframe(
        df,
        time_col="datetime",
        value_cols=target_col
    )

    return series


def train_test_split(series, split_date):
    """
    Split TimeSeries into train/test
    """
    return series.split_before(pd.Timestamp(split_date))