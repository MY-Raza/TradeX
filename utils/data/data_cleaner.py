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
    full_range = range(start_ts, end_ts + interval_ms, interval_ms)

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
    if "timestamp" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "timestamp"}, inplace=True)

    logger.info(f"Cleaned OHLCV data | rows: {len(df)}")
    return df


# -------------------------------------------------
# Resample OHLCV Data (integer timestamp)
# -------------------------------------------------
# def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
#     if df.empty:
#         return df

#     if interval not in INTERVAL_MS_MAP:
#         raise ValueError(f"Unsupported interval: {interval}")

#     df = df.copy()
#     interval_ms = INTERVAL_MS_MAP[interval]

#     # --------------------------------------------------
#     # 🔧 NORMALIZE TIMESTAMP (column OR index)
#     # --------------------------------------------------
#     if "timestamp" not in df.columns:
#         if df.index.name == "timestamp":
#             # DatetimeIndex → UNIX ms column
#             df["timestamp"] = df.index.view("int64") // 1_000_000
#             df = df.reset_index(drop=True)
#         else:
#             raise KeyError("No timestamp column or index found")

#     # Ensure int64 UNIX ms
#     df["timestamp"] = df["timestamp"].astype("int64")

#     # --------------------------------------------------
#     # Ensure numeric OHLCV
#     # --------------------------------------------------
#     df = convert_ohlcv_to_float(df)

#     # --------------------------------------------------
#     # Sort & deduplicate
#     # --------------------------------------------------
#     df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

#     # --------------------------------------------------
#     # Bucket timestamps
#     # --------------------------------------------------
#     df["bucket"] = (df["timestamp"] // interval_ms) * interval_ms

#     # --------------------------------------------------
#     # Aggregate OHLCV
#     # --------------------------------------------------
#     resampled = (
#         df.groupby("bucket", sort=True)
#         .agg(
#             open=("open", "first"),
#             high=("high", "max"),
#             low=("low", "min"),
#             close=("close", "last"),
#             volume=("volume", "sum"),
#         )
#         .reset_index()
#         .rename(columns={"bucket": "timestamp"})
#     )

#     return resampled

def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df

    if interval not in INTERVAL_MS_MAP:
        raise ValueError(f"Unsupported interval: {interval}")

    df = df.copy()
    interval_ms = INTERVAL_MS_MAP[interval]

    # --------------------------------------------------
    # Ensure timestamp column exists
    # --------------------------------------------------
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df["timestamp"] = df.index
            df = df.reset_index(drop=True)
        else:
            raise KeyError("No timestamp column or index found")

    # --------------------------------------------------
    # Determine if timestamp is datetime or numeric
    # --------------------------------------------------
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # Already datetime → use as-is
        df_ts = df.copy()
    else:
        # Convert numeric to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df_ts = df.copy()

    # --------------------------------------------------
    # Ensure numeric OHLCV
    # --------------------------------------------------
    df_ts = convert_ohlcv_to_float(df_ts)

    # --------------------------------------------------
    # Sort & deduplicate
    # --------------------------------------------------
    df_ts = df_ts.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # --------------------------------------------------
    # Bucket timestamps
    # --------------------------------------------------
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # For datetime, create bucket using pandas offset
        df_ts["bucket"] = df_ts["timestamp"].dt.floor(interval)
    else:
        # For numeric UNIX ms
        df_ts["bucket"] = (df_ts["timestamp"] // interval_ms) * interval_ms

    # --------------------------------------------------
    # Aggregate OHLCV
    # --------------------------------------------------
    resampled = (
        df_ts.groupby("bucket", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"bucket": "timestamp"})
    )

    return resampled


