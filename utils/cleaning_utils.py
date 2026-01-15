import pandas as pd
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)

def clean_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Binance Futures kline DataFrame for further analysis or database insertion.

    Steps performed:
    1. Keep only relevant columns: 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    2. Convert numeric columns to float type for consistency.
    3. Drop duplicate rows based on the 'timestamp' column to avoid repeated data.
    4. Sort the DataFrame by 'timestamp' in ascending order.
    5. Drop the last row (incomplete candle) since it may not represent a full interval.

    Args:
        df (pd.DataFrame): Raw Binance kline DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for further processing or saving.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df.loc[:, numeric_cols] = df[numeric_cols].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    if not df.empty:
        df = df.iloc[:-1]

    logger.info(f"Cleaned DataFrame: {len(df)} rows remaining.")
    return df


def fill_missing_timestamps(df: pd.DataFrame, interval: str = "1min") -> pd.DataFrame:
    """
    Fill missing timestamps in the DataFrame with NaNs for OHLCV values.
    Useful when you want continuous time series for analysis.

    Args:
        df (pd.DataFrame): Cleaned kline DataFrame.
        interval (str): Time interval, e.g., '1min', '5min', '1h'.

    Returns:
        pd.DataFrame: DataFrame with missing timestamps inserted.
    """
    if df.empty:
        return df

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    # Map string intervals to pandas offset aliases
    interval_map = {"1m": "1min", "5m": "5min", "1h": "1H", "1d": "1D"}
    freq = interval_map.get(interval, "1min")  

    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(idx)

    # Reset index and convert back timestamp to ms
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df['timestamp'] = (df['timestamp'].astype('int64') // 10**6)

    logger.info(f"Inserted missing timestamps. Total rows: {len(df)}")
    return df


def remove_outliers(df: pd.DataFrame, columns=None, z_thresh=3.0) -> pd.DataFrame:
    """
    Remove rows with outliers based on z-score method.

    Args:
        df (pd.DataFrame): DataFrame containing numeric columns.
        columns (list): List of columns to check for outliers. Defaults to OHLCV.
        z_thresh (float): Z-score threshold for defining outliers.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if df.empty:
        return df

    df = df.copy()
    columns = columns or ["open", "high", "low", "close", "volume"]

    # Ensure all columns are float
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove any rows that are now NaN after conversion
    df = df.dropna(subset=columns)

    from scipy.stats import zscore
    z = df[columns].apply(zscore, nan_policy='omit')
    mask = (z.abs() < z_thresh).all(axis=1)
    df = df[mask]

    logger.info(f"Removed outliers. Total rows remaining: {len(df)}")
    return df



def fill_missing_values(df: pd.DataFrame, method='ffill') -> pd.DataFrame:
    """
    Fill missing OHLCV values in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with NaNs.
        method (str): 'ffill' for forward fill, 'bfill' for backward fill.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    if df.empty:
        return df

    df = df.copy()
    df.fillna(method=method, inplace=True)
    logger.info("Missing values filled using method: %s", method)
    return df


def resample_ohlcv(df: pd.DataFrame, interval: str = '5min') -> pd.DataFrame:
    """
    Resample OHLCV data to a higher interval.
    Example: 1-minute data -> 5-minute candles.

    Args:
        df (pd.DataFrame): DataFrame with timestamp as ms and OHLCV columns.
        interval (str): Resampling interval, e.g., '5min', '1H', '1D'.

    Returns:
        pd.DataFrame: Resampled OHLCV DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    df_resampled = df.resample(interval).agg(ohlc_dict).dropna().reset_index()
    df_resampled['timestamp'] = (df_resampled['timestamp'].astype('int64') // 10**6)
    
    logger.info(f"Resampled OHLCV to interval '{interval}'. Total rows: {len(df_resampled)}")
    return df_resampled
