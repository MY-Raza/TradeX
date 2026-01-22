# bybit_fetcher.py

import os
import time
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from TradeX.utils.common.logs import get_logger

# ---------------------------
# Initialize logger
# ---------------------------
logger = get_logger("bybit_fetcher")

# ---------------------------
# Load API keys from environment
# ---------------------------
# Construct path to .env file located at project root
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)

# Retrieve Bybit API credentials from environment variables
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Bybit API credentials not found in environment variables.")

# ---------------------------
# Create global HTTP client
# ---------------------------
BYBIT_CLIENT = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    demo=False,  # Set to True for demo mode if needed
)

logger.info("Bybit HTTP client initialized.")


class BybitFuturesFetcher:
    """
    A class to fetch historical USDT Perpetual Futures Klines from Bybit.

    Only fetches raw Kline data; does not clean, sort, or resample.

    Attributes:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        start_date (str): Start date for historical data ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS').
        end_date (str): End date for historical data; defaults to 'now' (current UTC time).
        interval (str): Kline interval (e.g., '1' for 1-minute klines).
        interval_ms (int): Interval duration in milliseconds (derived from `interval`).
        limit (int): Max number of klines per API request.
        max_loops (int): Maximum number of loops to prevent infinite fetching.
        start_ts (int): Start timestamp in milliseconds since epoch.
        end_ts (int): End timestamp in milliseconds since epoch.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1",
        limit: int = 1000,
        max_loops: int = 20_000,
    ):
        """
        Initialize the BybitFuturesFetcher instance.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            start_date (str): Start date of historical data.
            end_date (str): End date of historical data; defaults to 'now'.
            interval (str): Kline interval. Currently only supports '1' (1-minute).
            limit (int): Max number of klines per request (default 1000).
            max_loops (int): Max loop iterations to avoid infinite loops.
        """
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.max_loops = max_loops

        # Validate interval and calculate interval in milliseconds
        if interval == "1":
            self.interval_ms = 60_000  # 1 minute = 60,000 ms
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Convert start and end dates to timestamps in milliseconds
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BybitFuturesFetcher initialized | symbol={self.symbol} | "
            f"start={self.start_date} → end={self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """
        Convert start_date and end_date strings to epoch timestamps in milliseconds.

        Supports formats:
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

        # Ensure start_date is before end_date
        if start_ts >= end_ts:
            raise ValueError("start_date must be earlier than end_date")

        return start_ts, end_ts

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical Kline data from Bybit API.

        Bybit API returns a maximum of 'limit' klines per request.
        This method loops over the date range in chunks to fetch all data.

        Returns:
            pd.DataFrame: DataFrame containing historical klines with columns:
                ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']

        Notes:
            - Sleeps 0.25 seconds between requests to avoid hitting rate limits.
            - Tracks last timestamp to prevent duplicate fetches.
            - Stops fetching if max_loops is reached to avoid infinite loops.
        """
        all_klines = []
        start_ts = self.start_ts
        last_start_ts = None
        loops = 0

        while start_ts < self.end_ts:
            loops += 1
            # Stop loop if maximum iteration count reached
            if loops > self.max_loops:
                logger.error("Max loop count reached. Breaking loop.")
                break

            # Stop loop if timestamp did not advance
            if start_ts == last_start_ts:
                logger.error("Timestamp did not advance. Breaking loop.")
                break
            last_start_ts = start_ts

            logger.info(
                f"Fetching {self.symbol} from {datetime.utcfromtimestamp(start_ts / 1000)}"
            )

            # Fetch klines from Bybit API
            response = BYBIT_CLIENT.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.interval,
                start=start_ts,
                limit=self.limit,
            )

            klines = response.get("result", {}).get("list", [])
            if not klines:
                logger.info("No more data returned.")
                break

            # Reverse klines to chronological order (oldest first)
            klines.reverse()
            all_klines.extend(klines)

            # Advance start timestamp to last kline + interval
            start_ts = int(klines[-1][0]) + self.interval_ms

            # Polite sleep to avoid rate limits
            time.sleep(0.25)

        if not all_klines:
            logger.warning("No data fetched from Bybit.")
            return pd.DataFrame()

        # Convert list of klines to DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        return df
