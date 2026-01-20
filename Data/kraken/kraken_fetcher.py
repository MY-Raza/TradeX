# kraken_fetcher.py
import time
from datetime import datetime
import pandas as pd
import ccxt
from TradeX.utils.common.logs import get_logger

logger = get_logger("kraken_fetcher")


class KrakenFuturesFetcher:
    """
    Fetch historical Kraken Futures OHLCV (candlestick) data using CCXT.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        timeframe: str = "1m",
        limit: int = 100,
        sleep_seconds: float = 0.5,
    ):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.limit = limit
        self.sleep_seconds = sleep_seconds

        # Initialize Kraken exchange for futures
        self.exchange = ccxt.kraken({
            'options': {'defaultType': 'future'}
        })

        # Convert dates to timestamps
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"KrakenFuturesFetcher initialized | symbol={self.symbol} | timeframe={self.timeframe} "
            f"| start={self.start_date} | end={self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """Convert start_date and end_date to epoch milliseconds."""
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

        if start_ts >= end_ts:
            raise ValueError("start_date must be earlier than end_date")

        return start_ts, end_ts

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data in batches until the end timestamp is reached.
        Returns a pandas DataFrame.
        """
        all_ohlcv = []
        since = self.start_ts

        # Kraken fetch_ohlcv uses milliseconds for 'since'
        while since < self.end_ts:
            logger.info(
                f"Fetching {self.symbol} | {datetime.utcfromtimestamp(since / 1000)} | timeframe={self.timeframe}"
            )

            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=self.limit
                )
            except ccxt.NetworkError as e:
                logger.warning(f"Network error: {e}. Retrying in {self.sleep_seconds} seconds...")
                time.sleep(self.sleep_seconds)
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                break

            if not ohlcv:
                logger.warning("No more data returned from Kraken.")
                break

            all_ohlcv.extend(ohlcv)

            # Advance timestamp to last fetched + 1ms to avoid duplicates
            since = ohlcv[-1][0] + 1
            time.sleep(self.sleep_seconds)

        if not all_ohlcv:
            logger.warning("No data fetched from Kraken.")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        return df
