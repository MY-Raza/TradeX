# kraken_fetcher.py

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from TradeX.utils.common.logs import get_logger

logger = get_logger("kraken_fetcher")


class KrakenFuturesFetcher:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data from Kraken Futures.

    Returns a DataFrame with:
        ['datetime', 'open', 'high', 'low', 'close', 'volume']
    All timestamps are UTC-aware datetime objects.
    """

    BASE_URL_TEMPLATE = "https://futures.kraken.com/api/charts/v1/trade/{symbol}/{interval}"

    def __init__(self, symbol: str, interval: str = "1m"):
        self.symbol = symbol.upper()
        self.interval = interval
        self.base_url = self.BASE_URL_TEMPLATE.format(symbol=self.symbol, interval=self.interval)

        # Map interval to seconds for internal increment
        if interval.endswith("m"):
            self.interval_sec = int(interval[:-1]) * 60
        elif interval.endswith("h"):
            self.interval_sec = int(interval[:-1]) * 3600
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    @staticmethod
    def to_unix(date_str: str, end: bool = False) -> int:
        """Convert date string to UNIX timestamp in seconds."""
        if date_str.lower() == "now":
            return int(datetime.now(tz=timezone.utc).timestamp())

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if end:
                dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())

    def fetch_chunk(self, from_ts: int) -> dict:
        """Fetch a single chunk of OHLCV data from Kraken API."""
        params = {"from": from_ts}
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_data(self, start_date: str, end_date: str = "now") -> pd.DataFrame:
        """
        Fetch all OHLCV data between start_date and end_date.

        Returns:
            pd.DataFrame: ['datetime', 'open', 'high', 'low', 'close', 'volume']
        """
        start_ts = self.to_unix(start_date)
        end_ts = self.to_unix(end_date)

        all_candles = []
        current_from = start_ts

        while True:
            logger.info(f"Fetching candles from {datetime.utcfromtimestamp(current_from)} UTC")
            raw = self.fetch_chunk(current_from)
            candles = raw.get("candles", [])

            if not candles:
                logger.info("No more candles returned.")
                break

            all_candles.extend(candles)

            # Stop if API indicates no more candles
            if not raw.get("more_candles", False):
                break

            # Move to next timestamp
            last_ts = candles[-1]["time"]  # in seconds
            current_from = last_ts + self.interval_sec

            # Stop if passed end timestamp
            if current_from > end_ts:
                break

        if not all_candles:
            logger.warning("⚠️ No data fetched from Kraken.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_candles)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert timestamp -> UTC datetime
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Filter by end_date
        df = df[df["timestamp"] <= pd.to_datetime(end_date, utc=True)]

        logger.info(f"✅ Total rows fetched: {len(df)}")
        logger.info(f"Start: {df['timestamp'].min()}")
        logger.info(f"End  : {df['timestamp'].max()}")

        return df
