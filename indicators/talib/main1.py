# main.py

import numpy as np
import pandas as pd
import inspect
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.indicators.talib.signals import *
from TradeX.backtest.backtest1 import Backtest
from TradeX.backtest.backtester import Backtester,BacktestConfig
import os

# ---------------------------
# Logger
# ---------------------------
logger = get_logger("indicators_main")

# ---------------------------
# CSV Input/Output Config
# ---------------------------
INPUT_CSV = r"D:\trading\TradeX\indicators\talib\btc_1m_data.csv"
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

# # Resample 1m -> 1h
# df_1h = resample_ohlcv(df_1m, "1h")

# # # ---------------------------
# # # Prepare arrays for automatic argument detection
# # # ---------------------------
# # open_ = df_1h["open"].values
# # high = df_1h["high"].values
# # low = df_1h["low"].values
# # close = df_1h["close"].values
# # volume = df_1h["volume"].values
# # ref = np.random.uniform(50, 200, len(df_1h)).astype(np.float64)
# # periods = np.random.randint(5, 30, len(df_1h)).astype(np.float64)

# # # Mapping common names to arrays
# # AUTO_ARGS = {
# #     "open": open_,
# #     "high": high,
# #     "low": low,
# #     "close": close,
# #     "volume": volume,
# #     "ref": ref,
# #     "periods": periods,
# #     "pattern_name": "CDLDOJI"
# # }

# # # ---------------------------
# # # Automatically detect all *_signal functions
# # # ---------------------------
# # signal_funcs = {name: func for name, func in globals().items() if callable(func) and name.endswith("_signal")}

# # # ---------------------------
# # # Compute & save all signals automatically to separate CSVs
# # # ---------------------------
# # for func_name, func in signal_funcs.items():
# #     try:
# #         # Automatically get function arguments
# #         sig = inspect.signature(func)
# #         args_to_pass = []
# #         for param in sig.parameters.values():
# #             if param.name in AUTO_ARGS:
# #                 args_to_pass.append(AUTO_ARGS[param.name])
# #             elif param.default != inspect.Parameter.empty:
# #                 # use default value
# #                 pass
# #             else:
# #                 raise ValueError(f"Cannot automatically provide argument '{param.name}' for {func_name}")

# #         # Call the signal function
# #         result = func(*args_to_pass)

# #         # Determine CSV file name
# #         if func_name == "candlestick_signal":
# #             result, pattern = result
# #             csv_name = f"{func_name}_{pattern.lower()}.csv"
# #         else:
# #             csv_name = f"{func_name}.csv"

# #         # Convert result to DataFrame
# #         if isinstance(result, tuple):
# #             df_signal = pd.DataFrame({"signals": r for i, r in enumerate(result)})
# #         elif isinstance(result, np.ndarray) and result.ndim > 1 and result.shape[1] > 1:
# #             df_signal = pd.DataFrame(result, columns=["signals" for i in range(result.shape[1])])
# #         else:
# #             df_signal = pd.DataFrame({"signals": result})

# #         df_signal.insert(0, "timestamp", df_1h["timestamp"])

# #         # Save to CSV
# #         output_csv = os.path.join(SIGNALS_FOLDER, csv_name)
# #         df_signal.to_csv(output_csv, index=False)
# #         logger.info(f"Saved {func_name} → {output_csv}")

# #     except Exception as e:
# #         logger.error(f"Error computing {func_name}: {e}")

# # ---------------------------
# # Backtesting example
# # ---------------------------
os.makedirs(SIGNALS_FOLDER, exist_ok=True)
signals_csv = os.path.join(SIGNALS_FOLDER, "bbands_signal.csv")  
if not os.path.exists(signals_csv):
    logger.error(f"Signal CSV {signals_csv} not found. Exiting backtest.")
    exit()

# signals_df = pd.read_csv(signals_csv)
# # Convert both to datetime FIRST
# df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True)
# signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"], utc=True)

# # OPTIONAL (if you don’t want timezone-aware data)
# df_1m["timestamp"] = df_1m["timestamp"].dt.tz_convert(None)
# signals_df["timestamp"] = signals_df["timestamp"].dt.tz_convert(None)

# bt = Backtester(
#         price_df=df_1m,
#         signal_df=signals_df,
#     )
# trades_df,final_balance,total_return =bt.run_backtest()

# print("\n===== BACKTEST RESULTS =====")
# print(trades_df.head())
# print(f"\nFinal Balance: ${final_balance}")
# print(f"Total Return: {total_return}%")

#     # Save trade history
# csv_name = "ledger.csv"
# output_csv = os.path.join(SIGNALS_FOLDER, csv_name)
# trades_df.to_csv(output_csv, index=False)

# # Load CSV files

    # Run backtest
df_predictions = pd.read_csv(signals_csv)
df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp']).dt.tz_localize(None)
df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp']).dt.tz_localize(None)
bt = Backtest(df_1m, df_predictions,take_profit=3,stop_loss=1)
df_ledger, final_balance, pnl_percent = bt.run()
print("Final Balance:", final_balance)
print("PnL %:", pnl_percent)
df_ledger.to_csv("ledger1.csv", index=False)
print("Results saved to ledger1.csv")

# signals_df = pd.read_csv(signals_csv)

# price_df = df_1m[["timestamp", "open", "high", "low"]].copy()
# price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
# signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])

# bt = Backtester(
#     BacktestConfig(
#         starting_balance=1000,
#         leverage=1,
#         take_profit_pct=0.03,
#         stop_loss_pct=0.01
#     )
# )

# ledger, final_balance, pnl_pct = bt.run(price_df, signals_df)

# print("Final Balance:", final_balance)
# print("PnL %:", pnl_pct)

# ledger.to_csv(os.path.join(SIGNALS_FOLDER, "ledger.csv"), index=False)



