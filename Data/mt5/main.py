import MetaTrader5 as mt5
from TradeX.utils.common.config_loader import read_config



# Initialize connection
login = 261947
password = "784iOm&y9B"
server = "FusionMarkets-Demo"  

if not mt5.initialize(login=login, password=password, server=server):
    print("initialize() failed, error code:", mt5.last_error())
else:
    print("MT5 initialized successfully")

symbols = mt5.symbols_get()
for s in symbols:
    if "FUT" in s.name or "CFD" in s.name:
        print(s.name)

from datetime import datetime, timedelta
import pandas as pd

symbol = "US30"  # Replace with your chosen symbol
timeframe = mt5.TIMEFRAME_M1  # 1-minute candles

# Define date range
utc_to = datetime.now()
utc_from = utc_to - timedelta(days=7)

# Fetch historical rates
rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)

# Convert to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]

print(df.head())

all_symbols = mt5.symbols_get()

# Print total number of symbols
print(f"Total symbols available: {len(all_symbols)}")

# Print first 20 symbols with details
for symbol in all_symbols[:20]:
    print(symbol.name, symbol.description, symbol.path)
