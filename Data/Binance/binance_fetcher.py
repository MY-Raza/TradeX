from binance.client import Client
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

class BinanceFuturesFetcher:
    def __init__(self, api_key=None, api_secret=None, db_url=None):
        """
        Initialize Binance client and SQLAlchemy engine
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.api_secret = api_secret or os.getenv("API_SECRET_KEY")
        self.client = Client(self.api_key, self.api_secret)

        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.engine = create_engine(self.db_url)

        # Create schema 'data_binance' if it does not exist
        with self.engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS data_binance;"))
            conn.commit()
        self.schema = "data_binance"

    def fetch_and_save(self, symbol: str, start_ts: int, end_ts: int, interval: str = "1m"):
        """
        Fetch Binance futures data and save to PostgreSQL under schema 'data_binance'.
        Each symbol gets its own table dynamically (symbol_interval).
        """
        symbol_clean = symbol.upper().replace("USDT", "")
        table_name = f"{symbol_clean.lower()}_{interval}"

        all_klines = []
        while start_ts < end_ts:
            print(f"Fetching {symbol} from {datetime.utcfromtimestamp(start_ts / 1000)} at interval {interval}")

            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=1000
            )

            if not klines:
                break

            all_klines.extend(klines)
            start_ts = klines[-1][0] + 1
            time.sleep(0.5)

        if not all_klines:
            print("No data fetched.")
            return

        # Prepare DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # Remove duplicates & sort
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        if not df.empty:
            df = df.iloc[:-1]

        # Save to PostgreSQL using SQLAlchemy with schema
        df.to_sql(
            table_name,
            self.engine,
            schema=self.schema,       # save table in data_binance schema
            if_exists="append",
            index=False,
            method='multi'
        )
        print(f"Inserted {len(df)} rows into table '{self.schema}.{table_name}'.")
