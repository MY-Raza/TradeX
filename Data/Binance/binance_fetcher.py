import time
from datetime import datetime
import pandas as pd
from binance.client import Client
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)

class BinanceFuturesFetcher:
    """
    A class to fetch historical Binance Futures klines (candlestick) data.  

    This class handles:
        - Conversion of human-readable dates to timestamps in milliseconds.
        - Automatic fetching of multiple batches of klines to respect Binance API limits.
        - Optional pause between requests to avoid rate limits.
    
    Attributes:
        client (Client): Binance API client.
        symbol (str): Trading pair symbol, e.g., "BTCUSDT".
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format or "now".
        interval (str): Kline interval, e.g., "1m", "5m", "1h".
        limit (int): Maximum number of klines per API call (default 1000).
        sleep_seconds (float): Pause between API calls to avoid rate limits.
        start_ts (int): Start timestamp in milliseconds.
        end_ts (int): End timestamp in milliseconds.
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
        """
        Initialize the BinanceFuturesFetcher with API keys and fetch parameters.

        Args:
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
            symbol (str): Trading pair symbol (e.g., "BTCUSDT").
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str, optional): End date (YYYY-MM-DD) or "now". Defaults to "now".
            interval (str, optional): Kline interval. Defaults to "1m".
            limit (int, optional): Maximum klines per request. Defaults to 1000.
            sleep_seconds (float, optional): Pause between API calls. Defaults to 0.5.
        """
        # Initialize Binance API client
        self.client = Client(api_key, api_secret)

        # Store configuration parameters
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.sleep_seconds = sleep_seconds

        # Convert start and end dates to timestamps
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BinanceFuturesFetcher initialized | "
            f"symbol={self.symbol} | interval={self.interval} | "
            f"start={self.start_date} | end={self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """
        Convert start and end dates to milliseconds timestamps.

        Returns:
            tuple[int, int]: start_ts, end_ts in milliseconds.

        Raises:
            ValueError: If date format is invalid.
        """
        try:
            # Convert start date to timestamp
            start_ts = int(
                datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000
            )

            # Convert end date or use current UTC time
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
        Fetch Binance Futures klines for the configured symbol and date range.

        Handles fetching in batches due to API limits.  
        Automatically sleeps between requests to avoid hitting rate limits.

        Returns:
            pd.DataFrame: DataFrame containing OHLCV data with columns:
                - timestamp
                - open
                - high
                - low
                - close
                - volume
                - close_time
                - quote_asset_volume
                - number_of_trades
                - taker_buy_base_volume
                - taker_buy_quote_volume
                - ignore
        """
        all_klines = []
        start_ts = self.start_ts

        # Fetch data in batches until reaching the end timestamp
        while start_ts < self.end_ts:
            logger.info(
                f"Fetching {self.symbol} | "
                f"{datetime.utcfromtimestamp(start_ts / 1000)} | interval={self.interval}"
            )

            # Fetch a batch of klines
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=self.interval,
                startTime=start_ts,
                endTime=self.end_ts,
                limit=self.limit
            )

            # Break if no more data is returned
            if not klines:
                logger.warning("No more klines returned from Binance.")
                break

            # Add batch to the overall results
            all_klines.extend(klines)

            # Update start_ts to the last timestamp + 1 ms to avoid duplicates
            start_ts = klines[-1][0] + 1

            # Sleep to avoid hitting Binance rate limits
            time.sleep(self.sleep_seconds)

        if not all_klines:
            logger.warning("No data fetched from Binance.")
            return pd.DataFrame()

        # Convert raw data to DataFrame
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
