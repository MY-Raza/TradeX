import pandas as pd
from TradeX.utils.common.config_loader import get_logger
from TradeX.utils.common.constants import INTERVAL_MS_MAP

logger = get_logger("data_cleaner")



# -------------------------------------------------
# Helper: convert OHLCV columns to float
# -------------------------------------------------
def convert_ohlcv_to_float(df: pd.DataFrame) -> pd.DataFrame:
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df[ohlcv_cols] = df[ohlcv_cols].astype(float)
    return df


# -------------------------------------------------
# Clean OHLCV Data (epoch ms only)
# -------------------------------------------------
def clean_df(df: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    """
    OHLCV cleaning pipeline using raw Unix epoch timestamps (milliseconds).
    No datetime conversion.
    """

    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df
    

    if interval not in INTERVAL_MS_MAP:
        raise ValueError(f"Unsupported interval: {interval}")

    interval_ms = INTERVAL_MS_MAP[interval]

    df = df.copy()

    # ---------------------------
    # Required columns
    # ---------------------------
    if "time" in df.columns:
      df = df.rename(columns={"time": "timestamp"}, inplace=True)
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols]

    # ---------------------------
    # Ensure correct dtypes
    # ---------------------------
    df["timestamp"] = df["timestamp"].astype("int64")
    df = convert_ohlcv_to_float(df)

    # ---------------------------
    # Sort & deduplicate
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
    if len(full_range) > 10_000_000:  # threshold, adjust based on memory
     logger.warning("Skipping full reindexing due to huge number of rows")
    else:
     df = df.set_index("timestamp").reindex(full_range)

    # ---------------------------
    # Forward fill OHLCV
    # ---------------------------
    df = convert_ohlcv_to_float(df)
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].fillna(method='ffill').fillna(method='bfill')

    # ---------------------------
    # Restore timestamp column
    # ---------------------------
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    logger.info(f"Cleaned OHLCV data | rows: {len(df)}")
    return df


# -------------------------------------------------
# Resample OHLCV Data (INT-only, code-based)
# -------------------------------------------------
def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample OHLCV data using integer timestamp bucketing.
    No datetime, no database.
    """

    if df.empty:
        return df

    if interval not in INTERVAL_MS_MAP:
        raise ValueError(f"Unsupported interval: {interval}")

    interval_ms = INTERVAL_MS_MAP[interval]

    df = df.copy()

    # ---------------------------
    # Ensure correct dtypes
    # ---------------------------
    df["timestamp"] = df["timestamp"].astype("int64")
    df = convert_ohlcv_to_float(df)

    # ---------------------------
    # Sort & deduplicate
    # ---------------------------
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # ---------------------------
    # Create time buckets
    # ---------------------------
    df["bucket"] = (df["timestamp"] // interval_ms) * interval_ms

    # ---------------------------
    # Aggregate OHLCV
    # ---------------------------
    resampled = (
        df.groupby("bucket", sort=True)
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

    logger.info(f"Resampled OHLCV to {interval} | rows: {len(resampled)}")
    return resampled
