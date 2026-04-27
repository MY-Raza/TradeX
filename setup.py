from setuptools import setup, find_packages

setup(
    name="TradeX",
    version="0.1.0",
    packages=[
        "TradeX",
        "TradeX.backtest",
        "TradeX.data",
        "TradeX.data.binance",
        "TradeX.data.bybit",
        "TradeX.data.kraken",
        "TradeX.data.mt5",
        "TradeX.sentiments",
        "TradeX.sentiments.data",
        "TradeX.utils",
        "TradeX.utils.db",
        "TradeX.utils.data",
    ],
    package_dir={
        "TradeX": ".",           # repo root maps to TradeX
        "TradeX.backtest": "backtest",
        "TradeX.data": "data",
        "TradeX.data.binance": "data/binance",
        "TradeX.data.bybit": "data/bybit",
        "TradeX.data.kraken": "data/kraken",
        "TradeX.data.mt5": "data/mt5",
        "TradeX.sentiments": "sentiments",
        "TradeX.sentiments.data": "sentiments/data",
        "TradeX.utils": "utils",
        "TradeX.utils.db": "utils/db",
        "TradeX.utils.data": "utils/data",
    },
)