# kraken_fetcher.py

import requests
import pandas as pd
from datetime import datetime, timezone
from TradeX.utils.common.logs import get_logger

logger = get_logger("kraken_fetcher")


class KrakenFuturesFetcher:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data from Kraken Futures.

    Usage:
        fetcher = KrakenFuturesFetcher(symbol="PF_XBTUSD", interval="1m")
        df = fetcher.fetch(start_date="2024-01-01", end_date="now")

    Attributes:
        symbol (str): Trading pair symbol (e.g., "PF_XBTUSD").
        interval (str): Kline interval (e.g., "1m", "5m", "1h").
        base_url (str): Full API URL for fetching OHLCV data.
    """

    BASE_URL_TEMPLATE = "https://futures.kraken.com/api/charts/v1/trade/{symbol}/{interval}"

    def __init__(self, symbol: str, interval: str = "1m"):
        """
        Initialize the KrakenFuturesFetcher.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Kline interval; defaults to "1m".
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.base_url = self.BASE_URL_TEMPLATE.format(symbol=self.symbol, interval=self.interval)

    @staticmethod
    def to_unix(date_str: str, end: bool = False) -> int:
        """
        Convert a date string to a UNIX timestamp in seconds.

        Supports formats:
            - "YYYY-MM-DD"
            - "YYYY-MM-DD HH:MM:SS"
            - "now"

        Args:
            date_str (str): Date string to convert.
            end (bool): If True and only a date is provided, sets time to 23:59:59.

        Returns:
            int: UNIX timestamp in seconds.
        """
        if date_str.lower() == "now":
            return int(datetime.now(tz=timezone.utc).timestamp())

        # Try parsing full datetime first
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Fallback to date only
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if end:
                dt = dt.replace(hour=23, minute=59, second=59)

        # Ensure timestamp is in UTC
        return int(dt.replace(tzinfo=timezone.utc).timestamp())

    def fetch_chunk(self, from_ts: int) -> dict:
        """
        Fetch a single chunk of OHLCV data starting from `from_ts`.

        Args:
            from_ts (int): UNIX timestamp (seconds) to start fetching from.

        Returns:
            dict: Parsed JSON response from Kraken Futures API.
        """
        params = {"from": from_ts}
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_data(self, start_date: str, end_date: str = "now") -> pd.DataFrame:
        """
        Fetch all OHLCV data between start_date and end_date.

        Returns a DataFrame with columns:
            ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Timestamp is in milliseconds for compatibility with other pipelines.

        Args:
            start_date (str): Start date string.
            end_date (str): End date string; defaults to "now".

        Returns:
            pd.DataFrame: Historical OHLCV data.
        """
        start_ts = self.to_unix(start_date)
        end_ts = self.to_unix(end_date)

        all_candles = []
        current_from = start_ts

        while True:
            print(f"Fetching candles from {datetime.utcfromtimestamp(current_from)}")
            raw = self.fetch_chunk(current_from)
            candles = raw.get("candles", [])

            if not candles:
                print("No more candles returned.")
                break

            all_candles.extend(candles)

            # Stop if API indicates no more candles
            if not raw.get("more_candles", False):
                print("Reached last candle.")
                break

            # Move to next timestamp (last candle + interval)
            last_ts = candles[-1]["time"] // 1000  # Convert seconds
            current_from = last_ts + 60  # Increment by 1 minute for 1m interval

            # Stop if passed end timestamp
            if current_from > end_ts:
                break

        # Convert list of candles to DataFrame
        df = pd.DataFrame(all_candles)
        logger.info(f"First 15 Rows Before Performing any Operation: {df.head(15)}")
        if df.empty:
            print("⚠️ No data fetched.")
            return df

        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        # Keep timestamp in milliseconds
        df["timestamp"] = df["time"].astype(int)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Filter for end date (converted to milliseconds)
        df = df[df["timestamp"] <= end_ts * 1000]

        # Debug info
        logger.info(f"✅ Total rows fetched: {len(df)}")
        logger.info(f"First 15 Rows:{df.head(15)}")
        logger.info(f"Start: {datetime.utcfromtimestamp(df['timestamp'].min() / 1000)}")
        logger.info(f"End  : {datetime.utcfromtimestamp(df['timestamp'].max() / 1000)}")

        return df
