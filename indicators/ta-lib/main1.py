# main.py

import numpy as np
import pandas as pd
import inspect
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from signals import *
from TradeX.backtest.backtester import Backtester
import os

# ---------------------------
# Logger
# ---------------------------
logger = get_logger("indicators_main")

# ---------------------------
# CSV Input/Output Config
# ---------------------------
INPUT_CSV = r"C:\Users\Yasir Raza Attari\Desktop\trading\TradeX\indicators\ta-lib\btc_1m_data.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
SIGNALS_FOLDER = os.path.join(BASE_DIR, "signals_csv")  
os.makedirs(SIGNALS_FOLDER, exist_ok=True)

# ---------------------------
# Fetch OHLCV Data from CSV
# ---------------------------
df_1m = pd.read_csv(INPUT_CSV)
if df_1m.empty:
    logger.error("No OHLCV data. Exiting.")
    exit()
df_1m["timestamp"] = pd.to_datetime(df_1m["datetime"])
# Optional: drop the old datetime column if you no longer need it
df_1m = df_1m.drop(columns=["datetime"])

# Resample 1m -> 1h
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
# Automatically detect all *_signal functions
# ---------------------------
signal_funcs = {name: func for name, func in globals().items() if callable(func) and name.endswith("_signal")}

# ---------------------------
# Compute & save all signals automatically to separate CSVs
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

        # Determine CSV file name
        if func_name == "candlestick_signal":
            result, pattern = result
            csv_name = f"{func_name}_{pattern.lower()}.csv"
        else:
            csv_name = f"{func_name}.csv"

        # Convert result to DataFrame
        if isinstance(result, tuple):
            df_signal = pd.DataFrame({f"{func_name}_{i}": r for i, r in enumerate(result)})
        elif isinstance(result, np.ndarray) and result.ndim > 1 and result.shape[1] > 1:
            df_signal = pd.DataFrame(result, columns=[f"{func_name}_{i}" for i in range(result.shape[1])])
        else:
            df_signal = pd.DataFrame({func_name: result})

        df_signal.insert(0, "timestamp", df_1h["timestamp"])

        # Save to CSV
        output_csv = os.path.join(SIGNALS_FOLDER, csv_name)
        df_signal.to_csv(output_csv, index=False)
        logger.info(f"Saved {func_name} → {output_csv}")

    except Exception as e:
        logger.error(f"Error computing {func_name}: {e}")

# ---------------------------
# Backtesting example
# ---------------------------
os.makedirs(SIGNALS_FOLDER, exist_ok=True)
signals_csv = os.path.join(SIGNALS_FOLDER, "candlestick_signal_cdldoji.csv")  
if not os.path.exists(signals_csv):
    logger.error(f"Signal CSV {signals_csv} not found. Exiting backtest.")
    exit()

signals_df = pd.read_csv(signals_csv, parse_dates=["timestamp"])

bt = Backtester(price_df=df_1m, signal_df=signals_df, tp=3, sl=1)
bt.run_backtest()
trades_df = bt.get_results()
print(trades_df)
