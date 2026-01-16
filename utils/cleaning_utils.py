import pandas as pd
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


def clean_klines_df(df: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    """
    Comprehensive OHLCV data cleaning pipeline.

    Steps performed:
        1. Keep only essential columns: timestamp, open, high, low, close, volume
        2. Convert OHLCV columns to float
        3. Ensure timestamp is integer milliseconds
        4. Sort by timestamp
        5. Remove duplicate timestamps
        6. Drop the last candle (often incomplete)
        7. Fill missing timestamps to ensure a continuous series
        8. Fill missing OHLCV values using forward-fill (default)

    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame.
        interval (str, optional): Kline interval for filling missing timestamps.
                                  Options: "1m", "5m", "15m", "1h", "1d". Default is "1m".

    Returns:
        pd.DataFrame: Cleaned OHLCV DataFrame with continuous timestamps and filled values.
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
    # Convert OHLCV to float and timestamp to int
    # ---------------------------
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)

    # -----
    # Sort 
    # -----
    df = df.sort_values("timestamp")

    # ---------------------------
    # Drop last candle (often incomplete)
    # ---------------------------
    if len(df) > 1:
        df = df.iloc[:-1]

    # ---------------------------
    # Fill missing timestamps
    # ---------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1H",
        "1d": "1D",
    }
    freq = interval_map.get(interval, "1min")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)

    # ---------------------------
    # Fill missing values
    # ---------------------------
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].fillna(method="ffill")

    # ---------------------------
    # Reset index and convert timestamp to ms
    # ---------------------------
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    df["timestamp"] = df["timestamp"].astype("int64") // 10**6

    logger.info(f"Cleaned OHLCV data | total rows: {len(df)}")
    return df


def resample_ohlcv(df: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.

    Aggregation rules:
        - open: first value
        - high: max value
        - low: min value
        - close: last value
        - volume: sum

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'timestamp' in ms.
        interval (str, optional): New resampling interval (e.g., "5min", "15min", "1H"). Defaults to "5min".

    Returns:
        pd.DataFrame: Resampled OHLCV DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    ohlc_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_resampled = df.resample(interval).agg(ohlc_agg).dropna().reset_index()
    df_resampled["timestamp"] = df_resampled["timestamp"].astype("int64") // 10**6

    logger.info(f"Resampled OHLCV to '{interval}' | rows: {len(df_resampled)}")
    return df_resampled
