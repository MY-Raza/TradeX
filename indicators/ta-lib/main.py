# main.py
import numpy as np
from indicators import *
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.db.utils import fetch_ohlcv_df
import pandas as pd
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from signals import generate_signals

logger = get_logger("indicators_main")

SCHEMA = EXCHANGE_SCHEMA_MAP["binance"]

# ---------------------------
#  OHLCV data
# ---------------------------
df_1m = fetch_ohlcv_df(
    table_name="btc",
    schema=SCHEMA,
    time_column="timestamp"
)

if df_1m.empty:
    logger.error(f"No Data Fetched")
    exit()

else:
    df_1h = resample_ohlcv(df_1m,interval="1h")
    open_ = df_1h["open"].values
    high = df_1h["high"].values
    low = df_1h["low"].values
    close = df_1h["close"].values
    volume = df_1h["volume"].values
    # Reference series for statistical indicators
    ref = np.random.uniform(50, 200,len(df_1h)).astype(np.float64)
    # Variable periods for MAVP
    periods = np.random.uniform(5, 30,len(df_1h)).astype(np.float64)

macd_val, macd_signal, _ = macd(close)

indicators = {
    'close': close,
    'sma': sma(close),
    'ema': ema(close),
    'adx': adx(high, low, close),
    'plus_di': plus_di(high, low, close),
    'minus_di': minus_di(high, low, close),
    'macd': macd(close)[0],          # macd value
    'macd_signal': macd(close)[1],   # macd signal
    'rsi': rsi(close),
    'mfi': mfi(high, low, close, volume),
    'stoch_k': stoch(high, low, close)[0],
    'stoch_d': stoch(high, low, close)[1],
    'atr': atr(high, low, close)
}

signals = generate_signals(indicators)

logger.info(f"Signals (last 10): {signals[-10:]}")
