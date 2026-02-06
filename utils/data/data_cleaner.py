# data_cleaner.py

import pandas as pd
from TradeX.utils.common.config_loader import get_logger
from TradeX.utils.common.constants import INTERVAL_MS_MAP
import numpy as np

# ---------------------------
# Initialize logger
# ---------------------------
logger = get_logger("data_cleaner")


# -------------------------------------------------
# Helper: Convert OHLCV columns to float
# -------------------------------------------------
def convert_ohlcv_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that OHLCV columns are of float type for calculations.

    Args:
        df (pd.DataFrame): DataFrame containing 'open', 'high', 'low', 'close', 'volume'.

    Returns:
        pd.DataFrame: DataFrame with OHLCV columns as float.
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df[ohlcv_cols] = df[ohlcv_cols].astype(float)
    return df


# -------------------------------------------------
# Clean OHLCV Data (epoch ms only)
# -------------------------------------------------
def clean_df(df: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    """
    Clean raw OHLCV data with Unix epoch timestamps (milliseconds).

    Steps:
        1. Validate required columns.
        2. Convert types to numeric.
        3. Sort and remove duplicates.
        4. Drop last (potentially incomplete) candle.
        5. Reindex to fill missing timestamps.
        6. Forward/backward fill OHLCV columns.
        7. Restore timestamp column if needed.

    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame.
        interval (str): OHLCV interval; used to calculate missing timestamps.

    Returns:
        pd.DataFrame: Cleaned OHLCV DataFrame.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    if interval not in INTERVAL_MS_MAP:
        raise ValueError(f"Unsupported interval: {interval}")

    interval_ms = INTERVAL_MS_MAP[interval]
    df = df.copy()

    # ---------------------------
    # Ensure required columns
    # ---------------------------
    if "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols]

    # ---------------------------
    # Convert numeric columns
    # ---------------------------
    df["timestamp"] = df["timestamp"]  # keep as int64
    df = convert_ohlcv_to_float(df)

    # ---------------------------
    # Sort by timestamp & remove duplicates
    # ---------------------------
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # ---------------------------
    # Drop last candle (often incomplete)
    # ---------------------------
    if len(df) > 1:
        df = df.iloc[:-1]

    # ---------------------------
    # Fill missing timestamps
    # ---------------------------
    start_ts = df["timestamp"].iloc[0]
    end_ts = df["timestamp"].iloc[-1]
    full_range = pd.date_range(
    start=start_ts,
    end=end_ts,
    freq=pd.Timedelta(milliseconds=interval_ms)
    )


    if len(full_range) > 10_000_000:  # safety threshold
        logger.warning("Skipping full reindexing due to huge number of rows")
    else:
        df = df.set_index("timestamp").reindex(full_range)

    # ---------------------------
    # Forward/backward fill OHLCV values
    # ---------------------------
    df = convert_ohlcv_to_float(df)
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].ffill().bfill()

    # ---------------------------
    # Restore timestamp column
    # ---------------------------
    df = df.reset_index().rename(columns={"index": "datetime"})

    logger.info(f"Cleaned OHLCV data | rows: {len(df)}")
    return df

def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()

    # Ensure timestamp column exists
    if "datetime" not in df.columns:
        if df.index.name == "datetime":
            df["datetime"] = df.index
            df = df.reset_index(drop=True)
        else:
            raise KeyError("No timestamp column or index found")

    # Convert to datetime
    df["datetime"] = ensure_datetime(df["datetime"])

    # Ensure numeric
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])

    # Interval → pandas freq
    def to_pandas_freq(interval: str) -> str:
        interval = interval.lower()
        if interval.endswith("m"):
            return interval[:-1] + "min"
        elif interval.endswith("h"):
            return interval[:-1] + "h"
        elif interval.endswith("d"):
            return interval[:-1] + "d"
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    freq = to_pandas_freq(interval)

    # --------------------------------------------------
    # Find FIRST aligned boundary AFTER first timestamp
    # --------------------------------------------------
    first_ts = df["datetime"].iloc[0]
    offset = pd.tseries.frequencies.to_offset(freq)

    floored = first_ts.floor(freq)
    start_boundary = floored if floored == first_ts else floored + offset

    # Drop partial leading candles
    df = df[df["datetime"] >= start_boundary]
    if df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    # --------------------------------------------------
    # Resample using anchored origin
    # --------------------------------------------------
    resampled = (
        df.set_index("datetime")
          .groupby(
              pd.Grouper(
                  freq=freq,
                  origin=start_boundary,
                  label="left",
                  closed="left"
              )
          )
          .agg(
              open=("open", "first"),
              high=("high", "max"),
              low=("low", "min"),
              close=("close", "last"),
              volume=("volume", "sum"),
          )
          .dropna(subset=["open"])  # remove empty buckets
          .reset_index()
    )

    return resampled

def ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, unit="ms", utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(series, utc=True, errors="coerce")


