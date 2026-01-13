from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Get the keys
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET_KEY')

client = Client(API_KEY, API_SECRET)

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE

end_time = datetime.utcnow()
start_time = end_time - timedelta(days=365)

start_ts = int(start_time.timestamp() * 1000)
end_ts = int(end_time.timestamp() * 1000)

all_klines = []

while start_ts < end_ts:
    print(f"Fetching data from {datetime.fromtimestamp(start_ts/1000)}")

    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ts,
        limit=1000
    )

    if not klines:
        break

    all_klines.extend(klines)
    start_ts = klines[-1][0] + 1

    time.sleep(0.5)

    columns = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore"
]

df = pd.DataFrame(all_klines, columns=columns)

df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

price_cols = ["open", "high", "low", "close", "volume"]
df[price_cols] = df[price_cols].astype(float)

df.to_csv("BTCUSDT_1m_1year.csv", index=False)

print("Data saved successfully!")