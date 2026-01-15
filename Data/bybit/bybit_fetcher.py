import time
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


class BybitFuturesFetcher:
    """
    Fetches raw Bybit USDT Perpetual Futures klines data.
    Handles conversion from start/end dates to timestamps internally
    and paginates safely to avoid infinite loops.
    """

    INTERVAL_MAP = {
        "1": 60_000,
    }

    def __init__(self, api_key: str, api_secret: str, demo: bool = False):
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            demo=demo
        )
        logger.info("BybitFuturesFetcher initialized.")

    def _convert_to_timestamp(self, start_date: str, end_date: str) -> tuple[int, int]:
        """
        Convert start and end dates (YYYY-MM-DD) to milliseconds timestamps.
        If end_date is "now", use current UTC time.
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
        interval: str = "1"
    ) -> pd.DataFrame:
        """
        Fetch Bybit Futures klines for a symbol between given dates.
        Paginates safely to avoid infinite loops.

        Args:
            symbol (str): Trading pair, e.g., 'BTCUSDT'
            start_date (str): Start date in 'YYYY-MM-DD'
            end_date (str): End date in 'YYYY-MM-DD' or 'now'
            interval (str): Kline interval in minutes (Bybit uses string values)

        Returns:
            pd.DataFrame: Raw klines DataFrame
        """
        start_ts, end_ts = self._convert_to_timestamp(start_date, end_date)

        all_klines = []
        while start_ts < end_ts:
            
            logger.info(
                f"Fetching {symbol} from {datetime.utcfromtimestamp(start_ts / 1000)} interval={interval}"
            )

            response = self.client.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=interval,
                        start=start_ts,
                        end=end_ts,
                         limit=1000
                        )

               # Extract the list of klines safely
            klines_list = response.get("result", {}).get("list", [])

            if not klines_list:
                logger.info("No more klines returned. Ending loop.")
                break

            all_klines.extend(klines_list)

            # Update start_ts using the last kline's timestamp
            start_ts = int(klines_list[-1][0]) + 1  # ensure int
            time.sleep(0.5)

        if not all_klines:
            logger.warning("No data fetched from Bybit.")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_klines,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ]
        )

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["timestamp"] = df["timestamp"].astype(int)

        logger.info(f"Fetched {len(df)} rows for {symbol}.")
        return df
