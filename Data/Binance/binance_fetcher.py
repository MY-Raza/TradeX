# binance_fetcher.py

import os
import time
from datetime import datetime
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
from TradeX.utils.common.logs import get_logger

# ---------------------------
# Initialize logger
# ---------------------------
logger = get_logger("binance_fetcher")

# ---------------------------
# Load API keys from environment
# ---------------------------
# Construct the path to the .env file located at the project root
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)

# Retrieve Binance API credentials from environment variables
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Binance API credentials not found in environment variables.")

# ---------------------------
# Create global Binance client
# ---------------------------
BINANCE_CLIENT = Client(API_KEY, API_SECRET)
logger.info("Binance Client initialized.")


class BinanceFuturesFetcher:
    """
    A class to fetch historical Binance Futures klines (candlestick) data.

    Attributes:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        start_date (str): Start date for historical data ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS').
        end_date (str): End date for historical data. Defaults to 'now' (current UTC time).
        interval (str): Kline interval (e.g., '1m', '5m', '1h', '1d').
        limit (int): Maximum number of klines returned per request (Binance API limit is 1000).
        sleep_seconds (float): Pause duration between requests to avoid rate limits.
        start_ts (int): Start timestamp in milliseconds since epoch.
        end_ts (int): End timestamp in milliseconds since epoch.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1m",
        limit: int = 1000,
        sleep_seconds: float = 2,
    ):
        """
        Initialize the BinanceFuturesFetcher instance.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            start_date (str): Start date of data.
            end_date (str): End date of data; defaults to "now".
            interval (str): Kline interval.
            limit (int): Max number of klines per request.
            sleep_seconds (float): Delay between API calls.
        """
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.sleep_seconds = sleep_seconds

        # Convert start and end dates to timestamps in milliseconds
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BinanceFuturesFetcher initialized | symbol={self.symbol} | interval={self.interval} "
            f"| start={self.start_date} | end={self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """
        Convert start_date and end_date strings to epoch timestamps in milliseconds.

        Supports the following formats:
            - 'YYYY-MM-DD'
            - 'YYYY-MM-DD HH:MM:SS'

        Returns:
            tuple[int, int]: (start_ts, end_ts) in milliseconds.
        
        Raises:
            ValueError: If start_date is later than end_date.
        """
        # Parse start_date
        try:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        start_ts = int(start_dt.timestamp() * 1000)

        # Parse end_date
        if self.end_date.lower() == "now":
            end_ts = int(datetime.utcnow().timestamp() * 1000)
        else:
            try:
                end_dt = datetime.strptime(self.end_date, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            end_ts = int(end_dt.timestamp() * 1000)

        # Ensure start date is before end date
        if start_ts >= end_ts:
            raise ValueError("start_date must be earlier than end_date")

        return start_ts, end_ts
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical klines data from Binance Futures API.

        Binance API returns a maximum of 'limit' klines per request.
        This method loops over the date range in chunks to fetch all data.

        Returns:
            pd.DataFrame: DataFrame containing historical klines with columns:
                ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                 'close_time', 'quote_asset_volume', 'number_of_trades',
                 'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']

        Notes:
            - Sleeps for `sleep_seconds` between requests to avoid hitting rate limits.
            - Automatically handles advancing the start timestamp to prevent duplicate data.
        """
        all_klines = []
        start_ts = self.start_ts

        while start_ts < self.end_ts:
            logger.info(
                f"Fetching {self.symbol} | {datetime.utcfromtimestamp(start_ts / 1000)} | interval={self.interval}"
            )

            # Fetch klines from Binance Futures API
            klines = BINANCE_CLIENT.futures_klines(
                symbol=self.symbol,
                interval=self.interval,
                startTime=start_ts,
                endTime=self.end_ts,
                limit=self.limit
            )

            if not klines:
                logger.warning("No more data returned from Binance.")
                break

            all_klines.extend(klines)

            # Advance start_ts to the timestamp of last kline + 1 ms to avoid duplicates
            start_ts = klines[-1][0] + 1

            # Polite sleep to prevent API rate limit errors
            time.sleep(self.sleep_seconds)

        if not all_klines:
            logger.warning("No data fetched from Binance.")
            return pd.DataFrame()

        # Convert list of klines to DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
