EXCHANGE_SCHEMA_MAP = {
    "bybit": "data_bybit",
    "binance": "data_binance",
}


from types import MappingProxyType

INTERVAL_MS_MAP = MappingProxyType({
    "1m": 60_000,
})