from __future__ import annotations

import logging

import pandas as pd
from darts import TimeSeries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_naive_utc(dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert any DatetimeIndex to tz-naive UTC."""
    if dt_index.tz is not None:
        return dt_index.tz_convert("UTC").tz_localize(None)
    return dt_index


def _naive_utc_timestamp(split_date: str) -> pd.Timestamp:
    """Return a tz-naive UTC Timestamp for *split_date*."""
    ts = pd.Timestamp(split_date)
    if ts.tz is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_series(df: pd.DataFrame, target_col: str = "close") -> TimeSeries:
    """
    Convert a DataFrame to a univariate Darts TimeSeries.

    Accepts DataFrames with either:
    - a 'datetime' column (any tz, or tz-naive), or
    - a DatetimeIndex (any tz, or tz-naive).

    The resulting TimeSeries is always tz-naive UTC.
    """
    # BUG-1 FIX: only check df.columns; index.name is NOT a data column.
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame columns. "
            f"Available columns: {list(df.columns)}"
        )

    if "datetime" in df.columns:
         
        dt = pd.to_datetime(df["datetime"])
        # BUG-3 FIX: always go through tz_localize("UTC") first so the series
        # is guaranteed tz-aware before tz_convert, then strip with tz_localize(None).
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        df["datetime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.sort_values("datetime")
        series = TimeSeries.from_dataframe(
            df, time_col="datetime", value_cols=target_col
        )
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have a DatetimeIndex or a 'datetime' column."
            )
         
        df.index = _to_naive_utc(df.index)
        # BUG-2 FIX: Darts requires a named index for from_dataframe.
        df.index.name = "datetime"
        series = TimeSeries.from_dataframe(df[[target_col]])

    return series


def train_test_split(
    series: TimeSeries, split_date: str
) -> tuple[TimeSeries, TimeSeries]:
    """
    Split a Darts TimeSeries at *split_date* into (train, test).

    split_before(ts) semantics: train = [start, ts), test = [ts, end].
    So split_date must be strictly inside the series.
    """
    ts = _naive_utc_timestamp(split_date)

    start = series.start_time()
    end   = series.end_time()

    if ts <= start:
        raise ValueError(
            f"split_date '{split_date}' ({ts}) is at or before the series "
            f"start ({start}).  No training data would remain."
        )
    if ts >= end:
        raise ValueError(
            f"split_date '{split_date}' ({ts}) is at or after the series "
            f"end ({end}).  No test data would remain."
        )

    train_series, test_series = series.split_before(ts)

    if len(train_series) == 0:
        raise ValueError(
            f"Train series is empty after split at '{split_date}'. "
            f"Series range: {start} → {end}"
        )
    if len(test_series) == 0:
        raise ValueError(
            f"Test series is empty after split at '{split_date}'. "
            f"Series range: {start} → {end}"
        )

    # BUG-4 FIX: log actual boundaries so split mis-configs surface instantly.
    logger.debug(
        "Split at %s: train=%d steps (%s → %s), test=%d steps (%s → %s).",
        ts,
        len(train_series), start, train_series.end_time(),
        len(test_series),  test_series.start_time(), end,
    )

    return train_series, test_series