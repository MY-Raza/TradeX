import pandas as pd
from TradeX.utils.common.utils_common import get_logger

logger = get_logger(__name__)


# -------------------------------------------------
# Helper: detect if timestamp column is in ms
# -------------------------------------------------
def is_timestamp_ms(series: pd.Series) -> bool:
    """
    Detect whether a timestamp Series is in milliseconds.
    """
    return pd.api.types.is_integer_dtype(series) or pd.api.types.is_numeric_dtype(series)


# -------------------------------------------------
# Clean OHLCV Data
# -------------------------------------------------
def clean_df(df: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    """
    Comprehensive OHLCV data cleaning pipeline with smart timestamp handling.
    """

    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    df = df.copy()

    # ---------------------------
    # Keep essential columns
    # ---------------------------
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[required_cols]

    # ---------------------------
    # Convert OHLCV to float
    # ---------------------------
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    # ---------------------------
    # Detect timestamp format
    # ---------------------------
    timestamp_is_ms = is_timestamp_ms(df["timestamp"])

    # Normalize timestamp → datetime for processing
    if timestamp_is_ms:
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ---------------------------
    # Sort and drop duplicates
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
    df.set_index("timestamp", inplace=True)

    freq_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1H",
        "1d": "1D",
    }
    freq = freq_map.get(interval, "1min")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)

    # ---------------------------
    # Fill missing OHLCV values
    # ---------------------------
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].ffill()

    # ---------------------------
    # Reset index
    # ---------------------------
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    # Convert back to ms ONLY if original input was ms
    if timestamp_is_ms:
        df["timestamp"] = df["timestamp"].astype("int64") // 10**6

    logger.info(f"Cleaned OHLCV data | total rows: {len(df)}")
    return df


# -------------------------------------------------
# Resample OHLCV Data
# -------------------------------------------------
def resample_ohlcv(df: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe with smart timestamp handling.
    """

    if df.empty:
        return df

    df = df.copy()

    # ---------------------------
    # Detect timestamp format
    # ---------------------------
    timestamp_is_ms = is_timestamp_ms(df["timestamp"])

    # Normalize timestamp → datetime
    if timestamp_is_ms:
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.set_index("timestamp", inplace=True)

    # ---------------------------
    # Resample
    # ---------------------------
    ohlc_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_resampled = (
        df.resample(interval)
        .agg(ohlc_agg)
        .dropna()
        .reset_index()
    )

    # Convert back to ms ONLY if original input was ms
    if timestamp_is_ms:
        df_resampled["timestamp"] = (
            df_resampled["timestamp"].astype("int64") // 10**6
        )

    logger.info(f"Resampled OHLCV to '{interval}' | rows: {len(df_resampled)}")
    return df_resampled
