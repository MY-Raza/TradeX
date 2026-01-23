import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from TradeX.utils.common.config_loader import read_config

# =========================================
# MT5 CONNECTION DETAILS
# =========================================
LOGIN = 261947
PASSWORD = "784iOm&y9B"
SERVER = "FusionMarkets-Demo"

TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute candles

# =========================================
# SYMBOL SUFFIX RULES
# =========================================
SPECIAL_SUFFIX = {
    "neth": "25"   # neth -> NETH25
}

def resolve_mt5_symbol(symbol: str) -> str:
    """
    Converts config symbol to MT5 symbol.
    Example:
      btc  -> BTCUSD
      eth  -> ETHUSD
      neth -> NETH25
    """
    symbol = symbol.lower()

    if symbol in SPECIAL_SUFFIX:
        return f"{symbol.upper()}{SPECIAL_SUFFIX[symbol]}"

    return f"{symbol.upper()}USD"


# =========================================
# INITIALIZE MT5
# =========================================
if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
    raise RuntimeError(f"❌ MT5 init failed: {mt5.last_error()}")

print("✅ MT5 initialized")

# =========================================
# LOAD CONFIG
# =========================================
config = read_config("config.yml")

raw_symbols = config["symbols"]          # ['btc', 'eth', 'neth', ...]
start_date = config["start_date"]
end_date = config["end_date"]

utc_from = datetime.fromisoformat(start_date)
utc_to = datetime.now() if end_date == "now" else datetime.fromisoformat(end_date)

# =========================================
# RESOLVE & VALIDATE SYMBOLS
# =========================================
mt5_symbols = []
print("\n📌 Resolving symbols:")

for sym in raw_symbols:
    mt5_sym = resolve_mt5_symbol(sym)

    if mt5.symbol_info(mt5_sym):
        mt5.symbol_select(mt5_sym, True)
        mt5_symbols.append(mt5_sym)
        print(f"  ✔ {sym} -> {mt5_sym}")
    else:
        print(f"  ⚠ {sym} -> {mt5_sym} (NOT FOUND)")

if not mt5_symbols:
    raise RuntimeError("❌ No valid symbols found on MT5")

# =========================================
# FETCH HISTORICAL DATA
# =========================================
def fetch_data(symbol):
    rates = mt5.copy_rates_range(symbol, TIMEFRAME, utc_from, utc_to)

    if rates is None or len(rates) == 0:
        print(f"❌ No data for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df[["time", "open", "high", "low", "close", "tick_volume"]]


# =========================================
# MAIN LOOP
# =========================================
print("\n📥 Fetching historical data...\n")

for symbol in mt5_symbols:
    print(f"➡ Fetching {symbol}")

    df = fetch_data(symbol)
    if df is not None:
        print(df.head())
        print(f"✅ Rows fetched: {len(df)}\n")

# =========================================
# SHUTDOWN MT5
# =========================================
mt5.shutdown()
print("🔌 MT5 shutdown complete")
