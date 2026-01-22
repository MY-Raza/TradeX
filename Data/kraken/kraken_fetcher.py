import requests
import pandas as pd
from datetime import datetime, timezone


class KrakenFuturesFetcher:
    """
    Fetch OHLCV data from Kraken Futures.

    Usage:
        fetcher = KrakenFuturesFetcher(symbol="PF_XBTUSD", interval="1m")
        df = fetcher.fetch(start_date="2024-01-01", end_date="now")
    """

    BASE_URL_TEMPLATE = "https://futures.kraken.com/api/charts/v1/trade/{symbol}/{interval}"

    def __init__(self, symbol: str, interval: str = "1m"):
        self.symbol = symbol
        self.interval = interval
        self.base_url = self.BASE_URL_TEMPLATE.format(symbol=self.symbol, interval=self.interval)

    @staticmethod
    def to_unix(date_str: str, end: bool = False) -> int:
        """
        Convert date string to UNIX timestamp (seconds).

        Supports:
          - "YYYY-MM-DD"
          - "YYYY-MM-DD HH:MM:SS"
          - "now"

        If end=True, sets to 23:59:59 if time is not provided.
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

        return int(dt.replace(tzinfo=timezone.utc).timestamp())

    def fetch_chunk(self, from_ts: int) -> dict:
        """
        Fetch a chunk of OHLCV data starting from `from_ts`.
        """
        params = {"from": from_ts}
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch(self, start_date: str, end_date: str = "now") -> pd.DataFrame:
        """
        Fetch all OHLCV data between start_date and end_date.
        Returns a DataFrame with columns: timestamp, open, high, low, close, volume
        Timestamp is in Unix milliseconds (ms) for compatibility with clean_df.
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

            # Stop if no more candles
            if not raw.get("more_candles", False):
                print("Reached last candle.")
                break

            # Move to next timestamp (last candle + interval)
            last_ts = candles[-1]["time"] // 1000  # seconds
            current_from = last_ts + 60  # 1-minute increment

            # Stop if passed end timestamp
            if current_from > end_ts:
                break

        # Convert to DataFrame
        df = pd.DataFrame(all_candles)
        if df.empty:
            print("⚠️ No data fetched.")
            return df

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        # --------------------------
        # Keep timestamp in milliseconds
        # --------------------------
        df["timestamp"] = df["time"].astype(int)  # milliseconds
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Filter for end date (convert end_ts to ms)
        df = df[df["timestamp"] <= end_ts * 1000]

        print(f"✅ Total rows fetched: {len(df)}")
        print(f"Start: {datetime.utcfromtimestamp(df['timestamp'].min() / 1000)}")
        print(f"End  : {datetime.utcfromtimestamp(df['timestamp'].max() / 1000)}")

        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV."""
        df.to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")
