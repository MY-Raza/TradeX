import MetaTrader5 as mt5
from datetime import datetime
from TradeX.utils.common.config_loader import read_config
from TradeX.data.mt5.metatrader5_fetcher import MetaTrader5FutureFetcher


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
    "neth": "25"
}

def resolve_mt5_symbol(symbol: str) -> str:
    """
    Converts config symbol to MT5 symbol.
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
raw_symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

utc_from = datetime.fromisoformat(start_date)
utc_to = datetime.now() if end_date == "now" else datetime.fromisoformat(end_date)

# =========================================
# RESOLVE SYMBOLS
# =========================================
resolved_symbols = []
print("\n📌 Resolving symbols:")

for s in raw_symbols:
    mt5_sym = resolve_mt5_symbol(s)
    if mt5.symbol_info(mt5_sym):
        mt5.symbol_select(mt5_sym, True)
        resolved_symbols.append(mt5_sym)
        print(f"  ✔ {s} -> {mt5_sym}")
    else:
        print(f"  ⚠ {s} -> {mt5_sym} (NOT FOUND)")

if not resolved_symbols:
    raise RuntimeError("❌ No valid symbols found on MT5")

# =========================================
# CREATE FETCHER INSTANCE
# =========================================
fetcher = MetaTrader5FutureFetcher(
    symbols=resolved_symbols,
    utc_from=utc_from,
    utc_to=utc_to,
    timeframe=TIMEFRAME
)

# =========================================
# FETCH DATA
# =========================================
for symbol in resolved_symbols:
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
