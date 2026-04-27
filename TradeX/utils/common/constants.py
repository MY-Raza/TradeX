# Name of the Database Schemas to be created
EXCHANGE_SCHEMA_MAP = {
    "bybit": "data_bybit",
    "binance": "data_binance",
    "kraken": "data_kraken",
    "metatrader5": "data_metatrader5",
    "signals": "talib_signals"
}

# Interval That Can be used for resampling
INTERVAL_MS_MAP = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
}