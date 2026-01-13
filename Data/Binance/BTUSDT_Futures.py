from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET_KEY')

client = Client(API_KEY, API_SECRET)

# -------------------------------
# 2. PARAMETERS
# -------------------------------
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE

end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

start_ts = int(start_time.timestamp() * 1000)
end_ts = int(end_time.timestamp() * 1000)

all_klines = []

# -------------------------------
# 3. FETCH FUTURES KLINES
# -------------------------------
while start_ts < end_ts:
    print(f"Fetching futures data from {datetime.fromtimestamp(start_ts/1000)}")

    klines = client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ts,
        limit=1000
    )

    if not klines:
        break

    all_klines.extend(klines)
    start_ts = klines[-1][0] + 1

    time.sleep(0.3)

# -------------------------------
# 4. CREATE DATAFRAME
# -------------------------------
df = pd.DataFrame(all_klines, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore"
])

# -------------------------------
# 5. KEEP REQUIRED COLUMNS ONLY
# -------------------------------
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# Convert timestamp to UNIX (seconds)
df["timestamp"] = df["timestamp"] // 1000

# Convert values to float
df[["open", "high", "low", "close", "volume"]] = \
    df[["open", "high", "low", "close", "volume"]].astype(float)

# -------------------------------
# 6. SAVE CSV
# -------------------------------
df.to_csv("BTCUSDT_FUTURES_1m_1week.csv", index=False)

print("Futures data saved successfully!")
