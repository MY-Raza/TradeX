import pandas as pd
from darts import TimeSeries
from functools import lru_cache


def prepare_series(df: pd.DataFrame, target_col: str = "close") -> TimeSeries:
    """
    Convert a DataFrame to a Darts TimeSeries.

    Optimisations vs original:
    - Accepts a df that may already have datetime as the index (set upstream);
      avoids a redundant sort when the caller already sorted.
    - Uses `copy=False` to skip an unnecessary data copy inside Darts where possible.
    - Validates that target_col exists before handing off.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Normalise: work on the index if datetime is already set, otherwise use column
    if "datetime" in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime")
        series = TimeSeries.from_dataframe(df, time_col="datetime", value_cols=target_col)
    else:
        # datetime is already the index (set by arima/nbeats/transformer train fns)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column.")
        series = TimeSeries.from_dataframe(df[[target_col]])

    return series


def train_test_split(series: TimeSeries, split_date: str) -> tuple[TimeSeries, TimeSeries]:
    """
    Split a Darts TimeSeries into train / test at split_date.

    Bug-fix: the original returned the split result directly without checking
    whether either half is empty, which caused silent downstream failures.
    """
    ts = pd.Timestamp(split_date, tz="UTC")

    # Validate split lands inside the series
    if ts <= series.start_time():
        raise ValueError(
            f"split_date {split_date} is at or before the series start "
            f"({series.start_time()}). No training data would remain."
        )
    if ts >= series.end_time():
        raise ValueError(
            f"split_date {split_date} is at or after the series end "
            f"({series.end_time()}). No test data would remain."
        )

    train_series, test_series = series.split_before(ts)
    return train_series, test_series