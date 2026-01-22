# Name of the Database Schemas to be created
EXCHANGE_SCHEMA_MAP = {
    "bybit": "data_bybit",
    "binance": "data_binance",
    "kraken": "data_kraken"
}

# Interval That Can be used for resampling
INTERVAL_MS_MAP = {
    "1m": 60_000,
}