# main.py

import numpy as np
import pandas as pd
import inspect
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.db.utils import fetch_ohlcv_df, save_df_to_db
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from signals import *
from TradeX.backtest.backtester import Backtester

# ---------------------------
# Logger
# ---------------------------
logger = get_logger("indicators_main")

# ---------------------------
# Schemas
# ---------------------------
OHLCV_SCHEMA = EXCHANGE_SCHEMA_MAP["bybit"]
SIGNALS_SCHEMA = EXCHANGE_SCHEMA_MAP["signals"]

# ---------------------------
# Fetch OHLCV Data
# ---------------------------
df_1m = fetch_ohlcv_df("btc_1m", OHLCV_SCHEMA, "timestamp")
if df_1m.empty:
    logger.error("No OHLCV data. Exiting.")
    exit()

df_1h = resample_ohlcv(df_1m, "1h")

# ---------------------------
# Prepare arrays for automatic argument detection
# ---------------------------
open_ = df_1h["open"].values
high = df_1h["high"].values
low = df_1h["low"].values
close = df_1h["close"].values
volume = df_1h["volume"].values
ref = np.random.uniform(50, 200, len(df_1h)).astype(np.float64)
periods = np.random.randint(5, 30, len(df_1h)).astype(np.float64)

# Mapping common names to arrays
AUTO_ARGS = {
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
    "ref": ref,
    "periods": periods,
    "pattern_name": "CDLDOJI"
}

# ---------------------------
# Automatically detect all functions ending with _signal
# ---------------------------
signal_funcs = {name: func for name, func in globals().items() if callable(func) and name.endswith("_signal")}

# ---------------------------
# Compute & save all signals automatically
# ---------------------------
for func_name, func in signal_funcs.items():
    try:
        # Automatically get function arguments
        sig = inspect.signature(func)
        args_to_pass = []
        for param in sig.parameters.values():
            if param.name in AUTO_ARGS:
                args_to_pass.append(AUTO_ARGS[param.name])
            elif param.default != inspect.Parameter.empty:
                # use default value
                pass
            else:
                raise ValueError(f"Cannot automatically provide argument '{param.name}' for {func_name}")

        # Call the signal function
        result = func(*args_to_pass)

        # Handle candlestick special case
        if func_name == "candlestick_signal":
            result, pattern = result
            table_name = f"btc_{pattern.lower()}_1h"
        else:
            table_name = f"btc_{func_name.replace('_signal','')}_1h"

        # Convert result to DataFrame
        if isinstance(result, tuple):
            df_signal = pd.DataFrame({f"{func_name}_{i}": r for i, r in enumerate(result)})
        elif isinstance(result, np.ndarray) and result.ndim > 1 and result.shape[1] > 1:
            df_signal = pd.DataFrame(result, columns=[f"{func_name}_{i}" for i in range(result.shape[1])])
        else:
            df_signal = pd.DataFrame({func_name: result})

        df_signal.insert(0, "timestamp", df_1h["timestamp"])

        # Save to database
        save_df_to_db(df_signal, table_name, SIGNALS_SCHEMA, "timestamp", is_timeseries=True)
        logger.info(f"Saved {func_name} → {table_name}")

    except Exception as e:
        logger.error(f"Error computing {func_name}: {e}")


# ---------------------------
# Backtesting example
# ---------------------------
signals_df = fetch_ohlcv_df("btc_cdldoji_1h", SIGNALS_SCHEMA, "timestamp")
bt = Backtester(price_df=df_1m, signal_df=signals_df, tp=3, sl=1)
bt.run_backtest()
print(bt.get_results())