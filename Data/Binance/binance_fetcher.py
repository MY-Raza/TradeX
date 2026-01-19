# binance_fetcher.py
import os
import time
from datetime import datetime
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
from TradeX.utils.common.logs import get_logger

logger = get_logger("binance_fetcher")

# ---------------------------
# Load API keys from environment
# ---------------------------
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)

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
    Fetch historical Binance Futures klines (candlestick) data.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1m",
        limit: int = 1000,
        sleep_seconds: float = 0.5,
    ):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.sleep_seconds = sleep_seconds

        # Convert start and end dates to timestamps
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BinanceFuturesFetcher initialized | symbol={self.symbol} | interval={self.interval} "
            f"| start={self.start_date} | end={self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        try:
            start_ts = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = (
                int(datetime.utcnow().timestamp() * 1000)
                if self.end_date.lower() == "now"
                else int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
            )
            return start_ts, end_ts
        except Exception:
            logger.exception("Failed to convert dates to timestamps.")
            raise

    def fetch_data(self) -> pd.DataFrame:
        all_klines = []
        start_ts = self.start_ts

        while start_ts < self.end_ts:
            logger.info(
                f"Fetching {self.symbol} | {datetime.utcfromtimestamp(start_ts / 1000)} | interval={self.interval}"
            )

            klines = BINANCE_CLIENT.futures_klines(
                symbol=self.symbol,
                interval=self.interval,
                startTime=start_ts,
                endTime=self.end_ts,
                limit=self.limit
            )

            if not klines:
                logger.warning("No more klines returned from Binance.")
                break

            all_klines.extend(klines)

            # Advance start_ts to avoid duplicates
            start_ts = klines[-1][0] + 1

            # Polite sleep
            time.sleep(self.sleep_seconds)

        if not all_klines:
            logger.warning("No data fetched from Binance.")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_klines,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ]
        )

        logger.info(f"Fetched {len(df)} rows for {self.symbol}.")
        return df
