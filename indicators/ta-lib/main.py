# main.py

import numpy as np
import pandas as pd
from indicators import *
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.db.utils import fetch_ohlcv_df,save_df_to_db
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from signals import generate_signals

# ---------------------------
# Logger Initialization
# ---------------------------
logger = get_logger("indicators_main")

# ---------------------------
# Exchange schema configuration
# ---------------------------
SCHEMA = EXCHANGE_SCHEMA_MAP["binance"]
schema = EXCHANGE_SCHEMA_MAP["signals"]

# ---------------------------
# Fetch OHLCV data (1-minute interval)
# ---------------------------
df_1m = fetch_ohlcv_df(
    table_name="btc",      # Table name in database
    schema=SCHEMA,         # Exchange schema
    time_column="timestamp"  # Column storing timestamp
)

if df_1m.empty:
    logger.error("No Data Fetched from database. Exiting.")
    exit()
else:
    # ---------------------------
    # Resample 1-minute data to 1-hour data
    # ---------------------------
    df_1h = resample_ohlcv(df_1m, interval="1h")
    
    # Extract OHLCV series as numpy arrays
    open_ = df_1h["open"].values
    high = df_1h["high"].values
    low = df_1h["low"].values
    close = df_1h["close"].values
    volume = df_1h["volume"].values
    
    # Optional: Reference series and variable periods (used in some indicators)
    ref = np.random.uniform(50, 200, len(df_1h)).astype(np.float64)
    periods = np.random.uniform(5, 30, len(df_1h)).astype(np.float64)

# ---------------------------
# Compute MACD indicator
# ---------------------------
macd_val, macd_signal, _ = macd(close)

# ---------------------------
# Compute all indicators and store in a dictionary
# This dictionary will be passed to the signal generator
# ---------------------------
indicators = {
    'close': close,
    'sma': sma(close),                     # Simple Moving Average
    'ema': ema(close),                     # Exponential Moving Average
    'adx': adx(high, low, close),          # Average Directional Index
    'plus_di': plus_di(high, low, close),  # +DI
    'minus_di': minus_di(high, low, close),# -DI
    'macd': macd(close)[0],                # MACD line
    'macd_signal': macd(close)[1],         # MACD signal line
    'rsi': rsi(close),                     # Relative Strength Index
    'mfi': mfi(high, low, close, volume),  # Money Flow Index
    'stoch_k': stoch(high, low, close)[0],# Stochastic %K
    'stoch_d': stoch(high, low, close)[1],# Stochastic %D
    'atr': atr(high, low, close)           # Average True Range
}

# ---------------------------
# Generate trading signals
# 1 = Buy, -1 = Sell, 0 = Hold
# ---------------------------
signals = generate_signals(indicators)

# ---------------------------
# Log the last 10 signals for review
# ---------------------------
logger.info(f"Signals (last 50): {signals[-1:]}")

signals_df = pd.DataFrame({
    "timestamp": df_1h["timestamp"],
    "signal": signals                 # 1 = Buy, -1 = Sell, 0 = Hold
})

save_df_to_db(
    df=signals_df,
    table_name="btc_signals_1h",  
    schema=schema,             
    time_column="timestamp",
    is_timeseries=True         
)
