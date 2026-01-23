import MetaTrader5 as mt5
from datetime import datetime
from TradeX.utils.common.config_loader import read_config
from TradeX.data.mt5.metatrader5_fetcher import MetaTrader5FutureFetcher
from dotenv import load_dotenv
import os

# =========================================
# MT5 CONNECTION DETAILS
# =========================================
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)

MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER") 

# =========================================
# INITIALIZE MT5
# =========================================
if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    raise RuntimeError(f"❌ MT5 init failed: {mt5.last_error()}")
print("✅ MT5 initialized")

# =========================================
# LOAD CONFIG
# =========================================
config = read_config("config.yml")
raw_symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

utc_from = datetime.fromisoformat(start_date)
utc_to = datetime.now() if end_date == "now" else datetime.fromisoformat(end_date)

# =========================================
# CREATE FETCHER INSTANCE
# =========================================
fetcher = MetaTrader5FutureFetcher(raw_symbols, utc_from, utc_to, mt5.TIMEFRAME_M1)

# =========================================
# FETCH DATA
# =========================================
for symbol in raw_symbols:
    df = fetcher.fetch(symbol)
    if df is not None:
        print(f"\nData for {symbol}:")
        print(df.head())
        print(f"✅ Rows fetched: {len(df)}\n")

# =========================================
# SHUTDOWN MT5
# =========================================
mt5.shutdown()
print("🔌 MT5 shutdown complete")
