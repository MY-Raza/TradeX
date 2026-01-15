import time
from datetime import datetime
import pandas as pd
from binance.client import Client
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)

class BinanceFuturesFetcher:
    """
    BinanceFuturesFetcher is responsible for fetching raw kline (OHLCV) data 
    from Binance Futures for a given symbol and time range.

    This class **only fetches data**. It does not clean or save it.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the Binance client with API credentials.

        Args:
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
        """
        self.client = Client(api_key, api_secret)
        logger.info("BinanceFuturesFetcher initialized.")

    def fetch_klines(self, symbol: str, start_ts: int, end_ts: int, interval: str = "1m") -> pd.DataFrame:
        """
        Fetch Binance Futures kline data for a given symbol and time range.

        The function fetches data in chunks of 1000 records (Binance limit)
        until the entire requested time range is covered.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT").
            start_ts (int): Start timestamp in milliseconds.
            end_ts (int): End timestamp in milliseconds.
            interval (str, optional): Kline interval (default: "1m").

        Returns:
            pd.DataFrame: Raw OHLCV DataFrame with the following columns:
                ["timestamp", "open", "high", "low", "close", "volume",
                 "close_time", "quote_asset_volume", "number_of_trades",
                 "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]
        """
        all_klines = []  # List to store all fetched klines

        # Loop until we cover the entire time range
        while start_ts < end_ts:
            logger.info(f"Fetching {symbol} from {datetime.utcfromtimestamp(start_ts / 1000)} interval={interval}")
            
            # Fetch up to 1000 klines from Binance API
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=1000
            )

            # Break the loop if no more data is returned
            if not klines:
                logger.warning("No more klines returned from Binance.")
                break

            # Add fetched klines to the list
            all_klines.extend(klines)

            # Update start_ts to fetch the next batch
            start_ts = klines[-1][0] + 1

            # Sleep briefly to avoid hitting Binance rate limits
            time.sleep(0.5)

        # Return empty DataFrame if no data was fetched
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

        logger.info(f"Fetched {len(df)} rows for {symbol}.")
        return df
