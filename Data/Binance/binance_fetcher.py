import time
from datetime import datetime

import pandas as pd
from binance.client import Client

from TradeX.utils.db.utils import save_df_to_db
from TradeX.logs.logging import get_logger


logger = get_logger(__name__)


class BinanceFuturesFetcher:
    """
    Fetch Binance Futures kline data and store it in PostgreSQL / TimescaleDB.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        engine,
        schema
    ):
        """
        Initialize Binance client and database configuration.

        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
            engine: SQLAlchemy engine
            schema (str | None): Database schema (prompted at runtime if None)
        """
        self.client = Client(api_key, api_secret)
        self.engine = engine
        self.schema = schema

        logger.info("BinanceFuturesFetcher initialized.")

    def fetch_klines(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch Binance futures klines and return them as a pandas DataFrame.

        Args:
            symbol (str): Trading pair (e.g., BTCUSDT)
            start_ts (int): Start timestamp in milliseconds
            end_ts (int): End timestamp in milliseconds
            interval (str): Kline interval (default: 1m)

        Returns:
            pd.DataFrame: Cleaned OHLCV data
        """
        all_klines = []

        while start_ts < end_ts:
            logger.info(
                f"Fetching {symbol} | "
                f"from {datetime.utcfromtimestamp(start_ts / 1000)} | "
                f"interval={interval}"
            )

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
                "close_time", "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore"
            ]
        )

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        # Clean data
        df = (
            df.drop_duplicates(subset="timestamp")
              .sort_values("timestamp")
        )

        # Drop last (possibly incomplete) candle
        if not df.empty:
            df = df.iloc[:-1]

        logger.info(f"Fetched {len(df)} rows for {symbol}.")
        return df

    def fetch_and_save(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        interval: str = "1m"
    ):
        """
        Fetch Binance Futures data and store it in the database.

        Args:
            symbol (str): Trading pair
            start_ts (int): Start timestamp (ms)
            end_ts (int): End timestamp (ms)
            interval (str): Kline interval
        """
        df = self.fetch_klines(symbol, start_ts, end_ts, interval)

        if df.empty:
            logger.warning("No data to save. Skipping database insert.")
            return

        symbol_clean = symbol.upper().replace("USDT", "")
        table_name = f"{symbol_clean.lower()}_{interval}"

        logger.info(
            f"Saving data to table '{table_name}' "
            f"(schema will be resolved at runtime)."
        )

        save_df_to_db(
            df=df,
            table_name=table_name,
            engine=self.engine,
            schema=self.schema,          # ← prompts if None
            time_column="timestamp",
            is_timeseries=True
        )

        logger.info(
            f"Data saved successfully for {symbol} "
            f"({len(df)} rows)."
        )
