import MetaTrader5 as mt5
from datetime import datetime
from TradeX.utils.common.config_loader import read_config
from TradeX.data.mt5.metatrader5_fetcher import MetaTrader5FutureFetcher
from dotenv import load_dotenv
import os

# =========================================
# Load Environment Variables
# =========================================
load_dotenv()  # reads .env

MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")

# =========================================
# LOAD CONFIG
# =========================================
config = read_config("config.yml")
raw_symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

utc_from = datetime.fromisoformat(start_date)
utc_to = datetime.now() if end_date == "now" else datetime.fromisoformat(end_date)

# -------------------------------
# Create MT5 fetcher instance
# -------------------------------
fetcher = MetaTrader5FutureFetcher(
    login=MT5_LOGIN,
    password=MT5_PASSWORD,
    server=MT5_SERVER,
    symbols=raw_symbols,
    utc_from=utc_from,
    utc_to=utc_to,
    timeframe=None
)

# -------------------------------
# Fetch data for all symbols
# -------------------------------
for symbol in raw_symbols:
    df = fetcher.fetch(symbol)
    if df is not None:
        print(f"\nData for {symbol}:")
        print(df.head())
        print(f"✅ Rows fetched: {len(df)}\n")

# -------------------------------
# Shutdown MT5
# -------------------------------
fetcher.shutdown()
