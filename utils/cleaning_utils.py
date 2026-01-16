import pandas as pd
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


def clean_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic OHLCV cleaning:
    - Select required columns
    - Convert numeric fields
    - Sort by timestamp
    - Drop duplicates
    - Drop last incomplete candle
    """

    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[required_cols].copy()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp")

    # Drop last candle (often incomplete)
    if len(df) > 1:
        df = df.iloc[:-1]

    logger.info(f"Cleaned OHLCV data: {len(df)} rows")
    return df.reset_index(drop=True)


def fill_missing_timestamps(df: pd.DataFrame, interval: str = "1m") -> pd.DataFrame:
    """
    Insert missing timestamps for a continuous time series.
    """

    if df.empty:
        return df

    df = df.copy()
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

    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    df["timestamp"] = df["timestamp"].astype("int64") // 10**6

    logger.info(f"Missing timestamps filled. Total rows: {len(df)}")
    return df


def fill_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Fill missing OHLCV values using forward or backward fill.
    """

    if df.empty:
        return df

    df = df.copy()
    df.fillna(method=method, inplace=True)

    logger.info(f"Missing values filled using method='{method}'")
    return df


def resample_ohlcv(df: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
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

    df_resampled = (
        df.resample(interval)
        .agg(ohlc_agg)
        .dropna()
        .reset_index()
    )

    df_resampled["timestamp"] = (
        df_resampled["timestamp"].astype("int64") // 10**6
    )

    logger.info(
        f"Resampled OHLCV to '{interval}'. Rows: {len(df_resampled)}"
    )
    return df_resampled
