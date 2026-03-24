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
    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    if interval not in INTERVAL_MS_MAP:
        raise ValueError(f"Unsupported interval: {interval}")

    interval_ms = INTERVAL_MS_MAP[interval]
     

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[required_cols]

    # Convert numeric columns
    df = convert_ohlcv_to_float(df)

    # Sort + dedupe
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # 🔥 DO NOT DROP LAST CANDLE — Binance already sends closed candles

    # ✅ Convert to proper UTC datetime BEFORE reindexing
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp"]).set_index("datetime")

    # Build full expected timeline
    full_range = pd.date_range(
        start=df.index[0],
        end=df.index[-1],
        freq=pd.Timedelta(milliseconds=interval_ms),
        tz="UTC"
    )

    if len(full_range) <= 10_000_000:
        df = df.reindex(full_range)

    # Fill gaps safely
    df[["open", "high", "low", "close", "volume"]] = (
        df[["open", "high", "low", "close", "volume"]]
        .ffill()
        .bfill()
    )

    df = df.reset_index().rename(columns={"index": "datetime"})

    logger.info(f"Cleaned OHLCV data | rows: {len(df)}")
    return df


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

     

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


