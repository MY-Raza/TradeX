import pandas as pd
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


def clean_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic OHLCV data cleaning for Binance or other exchanges.

    Cleaning steps:
        1. Keep only essential columns: timestamp, open, high, low, close, volume
        2. Convert OHLCV fields to numeric types (float)
        3. Convert timestamp to integer milliseconds
        4. Sort by timestamp
        5. Remove duplicate timestamps
        6. Drop the last candle (often incomplete)

    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame.

    Returns:
        pd.DataFrame: Cleaned OHLCV DataFrame.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    # Keep only necessary columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[required_cols].copy()

    # Convert OHLCV to float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Ensure timestamp is int
    df["timestamp"] = df["timestamp"].astype(int)

    # Sort by timestamp
    df = df.sort_values("timestamp")
    

    # Drop the last candle (usually incomplete)
    if len(df) > 1:
        df = df.iloc[:-1]

    logger.info(f"Cleaned OHLCV data: {len(df)} rows")
    return df.reset_index(drop=True)


def fill_missing_timestamps(df: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    """
    Fill missing timestamps to ensure a continuous time series.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'timestamp' column in ms.
        interval (str, optional): Kline interval, e.g., "1m", "5m", "1h", "1d". Defaults to "1m".

    Returns:
        pd.DataFrame: OHLCV DataFrame with continuous timestamps.
    """
    if df.empty:
        return df

    df = df.copy()

    # Convert timestamps to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Map interval string to pandas frequency
    interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1H",
        "1d": "1D",
    }
    freq = interval_map.get(interval, "1min")

    # Create a full timestamp index
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)

    # Reindex to fill missing timestamps
    df = df.reindex(full_index)

    # Reset index and convert back to ms
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    df["timestamp"] = df["timestamp"].astype("int64") // 10**6

    logger.info(f"Missing timestamps filled. Total rows: {len(df)}")
    return df


def fill_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Fill missing OHLCV values using forward or backward fill.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with possible NaNs.
        method (str, optional): Fill method: 'ffill' (forward) or 'bfill' (backward). Defaults to "ffill".

    Returns:
        pd.DataFrame: OHLCV DataFrame with missing values filled.
    """
    if df.empty:
        return df

    df = df.copy()

    # Fill missing values
    df.fillna(method=method, inplace=True)

    logger.info(f"Missing values filled using method='{method}'")
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

    # Convert timestamp to datetime and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Define OHLCV aggregation rules
    ohlc_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Resample and drop incomplete periods
    df_resampled = df.resample(interval).agg(ohlc_agg).dropna().reset_index()

    # Convert timestamp back to milliseconds
    df_resampled["timestamp"] = df_resampled["timestamp"].astype("int64") // 10**6

    logger.info(f"Resampled OHLCV to '{interval}'. Rows: {len(df_resampled)}")
    return df_resampled
