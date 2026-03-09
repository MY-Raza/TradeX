import pandas as pd
from darts import TimeSeries


def _to_naive_utc(dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Darts silently strips timezone info from any tz-aware DatetimeIndex when
    building a TimeSeries (it warns but continues).  This helper converts a
    tz-aware index to tz-naive UTC so our DataFrames always match what Darts
    produces internally, preventing tz-naive vs tz-aware comparison errors.
    """
    if dt_index.tz is not None:
        return dt_index.tz_convert("UTC").tz_localize(None)
    return dt_index


def _naive_utc_timestamp(split_date: str) -> pd.Timestamp:
    """
    Return a tz-naive UTC Timestamp for split_date so it can be compared
    against the tz-naive index that Darts produces internally.
    """
    ts = pd.Timestamp(split_date)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def prepare_series(df: pd.DataFrame, target_col: str = "close") -> TimeSeries:
    """
    Convert a DataFrame to a Darts TimeSeries.

    Root-cause fix (tz error):
      Darts strips timezone info when constructing a TimeSeries and emits a
      warning.  If we pass a tz-aware DatetimeIndex, Darts silently drops the
      tz — but any subsequent split_before() call with a tz-aware Timestamp
      then raises "Cannot compare tz-naive and tz-aware timestamps".
      Solution: convert the datetime column / index to tz-naive UTC *before*
      handing it to Darts so everything is consistently tz-naive.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if "datetime" in df.columns:
        df = df.copy()
        df["datetime"] = (
            pd.to_datetime(df["datetime"], utc=True)   # ensure UTC
            .dt.tz_localize(None)                       # strip tz → naive UTC
        )
        df = df.sort_values("datetime")
        series = TimeSeries.from_dataframe(df, time_col="datetime", value_cols=target_col)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column.")
        df = df.copy()
        df.index = _to_naive_utc(df.index)
        series = TimeSeries.from_dataframe(df[[target_col]])

    return series


def train_test_split(series: TimeSeries, split_date: str) -> tuple[TimeSeries, TimeSeries]:
    """
    Split a Darts TimeSeries into train / test at split_date.

    Root-cause fix (tz error):
      series.start_time() / end_time() are tz-naive (Darts strips tz on
      construction).  We therefore compare against a tz-naive Timestamp.

    Additional fix:
      Validates that the split date is strictly inside the series so neither
      half is empty (the original gave a cryptic Darts error in that case).
    """
    # tz-naive UTC Timestamp — matches what Darts stores internally
    ts = _naive_utc_timestamp(split_date)

    if ts <= series.start_time():
        raise ValueError(
            f"split_date '{split_date}' is at or before the series start "
            f"({series.start_time()}).  No training data would remain."
        )
    if ts >= series.end_time():
        raise ValueError(
            f"split_date '{split_date}' is at or after the series end "
            f"({series.end_time()}).  No test data would remain."
        )

    train_series, test_series = series.split_before(ts)
    return train_series, test_series