import time
from datetime import datetime
import pandas as pd
from binance.client import Client
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)

class BinanceFuturesFetcher:
    """
    Fetches raw Binance Futures klines data.
    Handles conversion from start/end dates to timestamps internally.
    """
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        logger.info("BinanceFuturesFetcher initialized.")

    def _convert_to_timestamp(self, start_date: str, end_date: str) -> tuple[int, int]:
        """
        Convert start and end dates (YYYY-MM-DD) to milliseconds timestamps.
        If end_date is "now", use current UTC time.

        Returns:
            tuple[int, int]: (start_ts, end_ts) in milliseconds
        """
        try:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = (
                int(datetime.utcnow().timestamp() * 1000)
                if end_date.lower() == "now"
                else int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            )
            logger.info(f"Timestamps resolved | start={start_date} | end={end_date}")
            return start_ts, end_ts
        except Exception as e:
            logger.exception("Failed to convert dates to timestamps.")
            raise e

    def fetch_klines(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch Binance Futures klines for a symbol between given dates.

        Args:
            symbol (str): Trading pair, e.g., 'BTCUSDT'
            start_date (str): Start date in 'YYYY-MM-DD'
            end_date (str): End date in 'YYYY-MM-DD' or 'now'
            interval (str): Kline interval, default '1m'

        Returns:
            pd.DataFrame: Raw klines DataFrame
        """
        # Convert dates to timestamps internally
        start_ts, end_ts = self._convert_to_timestamp(start_date, end_date)

        all_klines = []

        while start_ts < end_ts:
            logger.info(f"Fetching {symbol} from {datetime.utcfromtimestamp(start_ts / 1000)} interval={interval}")

            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=1000
            )

            if not klines:
                logger.warning("No more klines returned from Binance.")
                break

            all_klines.extend(klines)
            start_ts = klines[-1][0] + 1
            time.sleep(0.5)

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

        logger.info(f"Fetched {len(df)} rows for {symbol}.")
        return df
