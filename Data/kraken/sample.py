import requests
import pandas as pd
from datetime import datetime, timezone

# -------------------------
SYMBOL = "PF_XBTUSD"
BASE_URL = f"https://futures.kraken.com/api/charts/v1/trade/{SYMBOL}/1m"

START_DATE = "2024-01-01"
END_DATE   = "2024-01-05"

# Convert to UNIX timestamps
def to_unix(date_str, end=False):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

start_ts = to_unix(START_DATE)
end_ts = to_unix(END_DATE, end=True)

# -------------------------
# FETCH CHUNK
# -------------------------
def fetch_chunk(from_ts):
    params = {"from": from_ts}
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# -------------------------
# FETCH ALL CANDLES
# -------------------------
all_candles = []
current_from = start_ts

while True:
    print(f"Fetching candles from {datetime.utcfromtimestamp(current_from)}")
    raw = fetch_chunk(current_from)
    candles = raw.get("candles", [])
    
    if not candles:
        print("No more candles returned.")
        break

    all_candles.extend(candles)

    # If no more candles, break
    if not raw.get("more_candles", False):
        print("Reached last candle.")
        break

    # Move to next timestamp (last candle + 60 seconds)
    last_ts = candles[-1]["time"] // 1000  # ms → s
    current_from = last_ts + 60

    # Stop if we passed requested end_ts
    if current_from > end_ts:
        break

# -------------------------
# CONVERT TO DATAFRAME
# -------------------------
df = pd.DataFrame(all_candles)
for col in ["open","high","low","close","volume"]:
    df[col] = pd.to_numeric(df[col])
df["timestamp"] = (df["time"] // 1000).astype(int)
df = df[["timestamp","open","high","low","close","volume"]]

# Filter for end date just in case
df = df[df["timestamp"] <= end_ts]

print("✅ Total rows fetched:", len(df))
print("Start:", datetime.utcfromtimestamp(df['timestamp'].min()))
print("End  :", datetime.utcfromtimestamp(df['timestamp'].max()))

df.to_csv(f"kraken_{SYMBOL}_ohlcv_full.csv", index=False)
print("💾 Saved to CSV")
