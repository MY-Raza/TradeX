from binance.client import Client
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET_KEY")

client = Client(API_KEY, API_SECRET)


# Function for Fetching Future Data

def fetch_1m_futures_data(symbol, start_ts, end_ts, output_path):
    interval = Client.KLINE_INTERVAL_1MINUTE
    all_klines = []

    while start_ts < end_ts:
        print(f"Fetching FUTURES {symbol} from {datetime.utcfromtimestamp(start_ts/1000)}")

        klines = client.futures_klines(
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

    raw_columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(all_klines, columns=raw_columns)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df["timestamp"] = df["timestamp"].astype("int64")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    # Safety: remove duplicates & sort
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"FUTURES saved: {output_path}")
