# binance_fetcher.py

from binance.client import Client
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()


class BinanceFuturesFetcher:
    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize Binance client using API keys from environment or provided arguments.
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.api_secret = api_secret or os.getenv("API_SECRET_KEY")
        self.client = Client(self.api_key, self.api_secret)

    def fetch_futures_data(self, symbol: str, start_ts: int, end_ts: int, output_path: str, interval: str = "1m"):
        """
        Fetch futures data for a given symbol between start_ts and end_ts with the specified interval.
        Save the result to output_path as CSV.
        
        interval: Binance interval string (e.g., '1m', '5m', '1h', '1d', etc.)
        """
        all_klines = []

        while start_ts < end_ts:
            print(f"Fetching FUTURES {symbol} from {datetime.utcfromtimestamp(start_ts / 1000)} at interval {interval}")

            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=1000
            )

            if not klines:
                break

            all_klines.extend(klines)
            start_ts = klines[-1][0] + 1
            time.sleep(0.5)

        raw_columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore"
        ]

        df = pd.DataFrame(all_klines, columns=raw_columns)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        df["timestamp"] = df["timestamp"].astype("int64")
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        # Remove duplicates & sort
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

        # Drop the last row
        if not df.empty:
            df = df.iloc[:-1]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"FUTURES saved: {output_path} (last row dropped)")
