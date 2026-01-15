from binance.client import Client
import pandas as pd
from datetime import datetime
import time
from TradeX.utils.db.utils import save_df_to_db  

class BinanceFuturesFetcher:
    def __init__(self, api_key, api_secret, engine, schema="data_binance"):
        """
        Initialize Binance client and DB engine info.
        """
        self.client = Client(api_key, api_secret)
        self.engine = engine
        self.schema = schema

    def fetch_klines(self, symbol: str, start_ts: int, end_ts: int, interval: str = "1m") -> pd.DataFrame:
        """
        Fetch Binance futures data and return as a pandas DataFrame.
        """
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
            return pd.DataFrame()

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

        # Clean data
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
        if not df.empty:
            df = df.iloc[:-1]

        return df

    def fetch_and_save(self, symbol: str, start_ts: int, end_ts: int, interval: str = "1m"):
        """
        Fetch data and save it to PostgreSQL using functional db_utils.
        """
        df = self.fetch_klines(symbol, start_ts, end_ts, interval)
        if df.empty:
            return

        symbol_clean = symbol.upper().replace("USDT", "")
        table_name = f"{symbol_clean.lower()}_{interval}"

        # Save DataFrame using the function from db_utils.py
        save_df_to_db(df, table_name, engine=self.engine, schema=self.schema,time_column='timestamp',is_timeseries = True)
