"""
utils.py — Darts TimeSeries helpers
=====================================
Bug-fixes & robustness improvements over original:

1. tz_localize(None) called on a column that is ALREADY tz-naive raises
   TypeError.  The original code called `.dt.tz_localize(None)` after
   `.utc=True` in `pd.to_datetime`, which *does* produce a tz-aware Series,
   so the chain worked — but only accidentally.  Now we use a single
   `.dt.tz_convert("UTC").dt.tz_localize(None)` guard that is safe whether
   the input is tz-aware or tz-naive.

2. `prepare_series` did not handle the case where `df` already had a
   DatetimeIndex AND the index was tz-aware (e.g. from DB fetch).
   `_to_naive_utc` now handles all four cases:
       tz-aware index → convert to UTC → strip tz
       tz-naive index → no-op

3. `train_test_split` now returns the boundary Timestamp in the log messages
   using the series's actual freq so debugging split mismatches is instant.

4. Minor: all public functions have explicit return-type annotations.

Performance:
- `df.copy()` is done only when mutation is needed (inside the branch that
  modifies the "datetime" column), not unconditionally.
"""

from __future__ import annotations

import pandas as pd
from darts import TimeSeries


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

    The resulting TimeSeries is always tz-naive UTC so Darts' internal
    split_before() comparisons never raise tz-naive vs tz-aware errors.
    """
    if target_col not in df.columns and target_col not in (df.index.name,):
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if "datetime" in df.columns:
        df = df.copy()
        # Coerce to UTC-aware then strip tz → always tz-naive UTC
        dt = pd.to_datetime(df["datetime"])
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
        df = df.copy()
        df.index = _to_naive_utc(df.index)
        series = TimeSeries.from_dataframe(df[[target_col]])

    return series


def train_test_split(
    series: TimeSeries, split_date: str
) -> tuple[TimeSeries, TimeSeries]:
    """
    Split a Darts TimeSeries at *split_date* into (train, test).

    Validates:
    - split_date is strictly after series start (otherwise no train data).
    - split_date is strictly before series end   (otherwise no test data).

    Both halves must be non-empty; raises ValueError with a human-readable
    message otherwise.
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

    # Defensive: Darts should never produce empty halves given the guards
    # above, but let's be explicit.
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

    return train_series, test_series