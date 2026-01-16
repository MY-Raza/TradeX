import time
from datetime import datetime
import pandas as pd
from binance.client import Client
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)

class BinanceFuturesFetcher:
    """
    Fetches raw Binance Futures klines data.
    All fetch parameters are configured at initialization.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1m",
        limit: int = 1000,
        sleep_seconds: float = 0.5
    ):
        self.client = Client(api_key, api_secret)

        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.sleep_seconds = sleep_seconds

        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BinanceFuturesFetcher initialized | "
            f"symbol={self.symbol} | interval={self.interval} | "
            f"start={self.start_date} | end={self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """
        Convert start and end dates (YYYY-MM-DD) to milliseconds timestamps.
        """
        try:
            start_ts = int(
                datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000
            )

            end_ts = (
                int(datetime.utcnow().timestamp() * 1000)
                if self.end_date.lower() == "now"
                else int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
            )

            return start_ts, end_ts

        except Exception:
            logger.exception("Failed to convert dates to timestamps.")
            raise

    def fetch_klines(self) -> pd.DataFrame:
        """
        Fetch Binance Futures klines using initialized parameters.
        """
        all_klines = []
        start_ts = self.start_ts

        while start_ts < self.end_ts:
            logger.info(
                f"Fetching {self.symbol} | "
                f"{datetime.utcfromtimestamp(start_ts / 1000)} | interval={self.interval}"
            )

            klines = self.client.futures_klines(
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
            start_ts = klines[-1][0] + 1
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
