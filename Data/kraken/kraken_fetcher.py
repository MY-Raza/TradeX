# kraken_fetcher.py
import time
from datetime import datetime
import pandas as pd
import ccxt
from TradeX.utils.common.logs import get_logger

logger = get_logger("kraken_fetcher")


class KrakenFuturesFetcher:
    """
    Fetch historical Kraken Futures OHLCV data in batches.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        timeframe: str = "1m",
        limit: int = 1000,
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
            "options": {"defaultType": "future"}
        })

        # Convert dates to timestamps (ms)
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"KrakenFuturesFetcher initialized | symbol={self.symbol} | timeframe={self.timeframe} "
            f"| start={self.start_date} | end={self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """Convert start_date and end_date to epoch milliseconds."""
        # Start date
        try:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        start_ts = int(start_dt.timestamp() * 1000)

        # End date
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
        Fetch OHLCV data in batches until end_ts.
        Returns a DataFrame with epoch ms timestamps.
        """
        all_ohlcv = []
        since = self.start_ts

        while since < self.end_ts:
            logger.info(
                f"Fetching {self.symbol} | from {datetime.utcfromtimestamp(since/1000)}"
            )

            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=self.limit
                )
            except ccxt.NetworkError as e:
                logger.warning(f"Network error: {e}, retrying in {self.sleep_seconds}s")
                time.sleep(self.sleep_seconds)
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                break

            if not ohlcv:
                logger.info("No more data returned from Kraken.")
                break

            all_ohlcv.extend(ohlcv)
            #since = ohlcv[-1][0] + 1  # Advance 1 ms to avoid duplicates
            #time.sleep(self.sleep_seconds)
            last_candle_ts = ohlcv[-1][0]
            if last_candle_ts == since:
            # Safety to avoid infinite loop
             since += 60000  # skip 1 minute
            else:
             since = last_candle_ts + 1
            
            time.sleep(self.sleep_seconds)

        if not all_ohlcv:
            logger.warning("No data fetched from Kraken.")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Keep timestamp as epoch ms (do not convert to datetime)
        return df
