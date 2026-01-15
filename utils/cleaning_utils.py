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
    # Check if the DataFrame is empty
    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    # Keep only relevant columns and create a new copy to avoid SettingWithCopyWarning
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # Convert numeric columns to float using .loc to avoid SettingWithCopyWarning
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df.loc[:, numeric_cols] = df[numeric_cols].astype(float)

    # Drop duplicate timestamps and sort by timestamp ascending
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    # Drop the last row, which may be an incomplete candle
    if not df.empty:
        df = df.iloc[:-1]

    logger.info(f"Cleaned DataFrame: {len(df)} rows remaining.")
    return df
