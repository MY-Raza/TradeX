# main.py

import numpy as np
import pandas as pd
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.db.utils import fetch_ohlcv_df, save_df_to_db,drop_table
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from signals import *
from indicators import *
from TradeX.backtest.backtester import Backtester

# ---------------------------
# Logger Initialization
# ---------------------------
logger = get_logger("indicators_main")

# ---------------------------
# Exchange schema configuration
# ---------------------------
SCHEMA = EXCHANGE_SCHEMA_MAP["signals"]
# ---------------------------
# Fetch OHLCV data (1-minute interval)
# ---------------------------
df_1m = fetch_ohlcv_df(
    table_name="btc_1m",               # Table name in database
    schema=EXCHANGE_SCHEMA_MAP["binance"],  # Exchange schema
    time_column="timestamp"         # Column storing timestamp
)

if df_1m.empty:
    logger.error("No Data Fetched from database. Exiting.")
    exit()

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
# Automatically detect all functions ending with '_signal' in signals.py
# ---------------------------
signal_funcs = {name: func for name, func in globals().items() if callable(func) and name.endswith("_signal")}

# ---------------------------
# Define required arguments for each signal
# ---------------------------
signal_args = {
    "sma_signal": (close,),
    "ema_signal": (close,),
    "dema_signal": (close,),
    "tema_signal": (close,),
    "trima_signal": (close,),
    "wma_signal": (close,),
    "t3_signal": (close,),
    "kama_signal": (close,),
    "ma_signal": (close,),

    "ht_trendline_signal": (close,),
    "mama_signal": (close,),
    "mavp_signal": (close, periods),

    "bbands_signal": (close,),
    "midpoint_signal": (close,),
    "midprice_signal": (high, low),

    "sar_signal": (close, high, low),
    "sarext_signal": (close, high, low),

    "adx_signal": (high, low, close),
    "adxr_signal": (high, low, close),

    "apo_signal": (close,),
    "ppo_signal": (close,),
    "macd_signal": (close,),
    "macdext_signal": (close,),
    "macdfix_signal": (close,),

    "cci_signal": (high, low, close),
    "mom_signal": (close,),
    "roc_signal": (close,),
    "rocp_signal": (close,),
    "rocr_signal": (close,),
    "rocr100_signal": (close,),
    "trix_signal": (close,),
    "cmo_signal": (close,),

    "mfi_signal": (high, low, close, volume),
    "bop_signal": (open_, high, low, close),

    "aroon_signal": (high, low),
    "aroonosc_signal": (high, low),

    "rsi_signal": (close,),
    "stoch_signal": (high, low, close),
    "stochf_signal": (high, low, close),
    "stochrsi_signal": (close,),
    "ultosc_signal": (high, low, close),
    "willr_signal": (high, low, close),

    "ad_signal": (high, low, close, volume),
    "adosc_signal": (high, low, close, volume),
    "obv_signal": (close, volume),

    "ht_trendmode_signal": (close,),
    "ht_phasor_signal": (close,),
    "ht_sine_signal": (close,),

    "avgprice_signal": (open_, high, low, close),
    "medprice_signal": (high, low, close),
    "typprice_signal": (high, low, close),
    "wclprice_signal": (high, low, close),

    "atr_signal": (high, low, close),
    "natr_signal": (high, low, close),
    "trange_signal": (high, low, close),

    "candlestick_signal":(open_, high, low, close, "CDLDOJI")
        ,

    "beta_signal": (close, ref),
    "correl_signal": (close, ref),
    "linearreg_angle_signal": (close,),
    "linearreg_slope_signal": (close,),
    "stddev_signal": (close,),
    "tsf_signal": (close,),
    "var_signal": (close,),
}

# ---------------------------
# Compute all signals and save each to its own table
# ---------------------------
for func_name, func in signal_funcs.items():
    args_config = signal_args.get(func_name, ())

    if not args_config:
        logger.warning(f"No arguments provided for {func_name}, skipping")
        continue

    # ---------------------------
    # Handle functions with MULTIPLE argument sets (like candlestick patterns)
    # ---------------------------
    if isinstance(args_config, list):
        for args in args_config:
            try:
                result = func(*args)

                # Special handling for candlestick_signal
                if func_name == "candlestick_signal":
                    signals_array, pattern_name = result
                    table_name = f"btc_{pattern_name.lower().replace('_signal','')}_1h"
                    col_prefix = func_name
                    result = signals_array
                    print(table_name)
                else:
                    table_name = f"btc_{func_name.replace('_signal','')}_1h"
                    col_prefix = func_name

                # Convert result to DataFrame
                if isinstance(result, tuple):
                    result_df = pd.DataFrame({f"{col_prefix}_{i}": r for i, r in enumerate(result)})
                elif isinstance(result, np.ndarray) and result.ndim > 1 and result.shape[1] > 1:
                    result_df = pd.DataFrame(result, columns=[f"{col_prefix}_{i}" for i in range(result.shape[1])])
                else:
                    result_df = pd.DataFrame({f"{col_prefix}": result})

                result_df.insert(0, "timestamp", df_1h["timestamp"])

                save_df_to_db(
                    result_df,
                    table_name=table_name,
                    schema=SCHEMA,
                    time_column="timestamp",
                    is_timeseries=True
                )
                logger.info(f"Saved {func_name} -> {table_name}")

            except Exception as e:
                logger.error(f"Error computing {func_name} with args {args}: {e}")

    # ---------------------------
    # Normal single-run signals
    # ---------------------------
    else:
        try:
            result = func(*args_config)
            if func_name == "candlestick_signal":
                signals_array, pattern_name = result   # unpack tuple
                result = signals_array                 # keep only numeric signal array
                table_name = f"btc_{pattern_name.lower().replace('_signal','')}_1h"  # dynamic table name
                col_prefix = func_name   
            else:
                table_name = f"btc_{func_name.replace('_signal','')}_1h"
                col_prefix = func_name  

            if isinstance(result, tuple):
                result_df = pd.DataFrame({f"{col_prefix}_{i}": r for i, r in enumerate(result)})
            elif isinstance(result, np.ndarray) and result.ndim > 1 and result.shape[1] > 1:
                result_df = pd.DataFrame(result, columns=[f"{col_prefix}_{i}" for i in range(result.shape[1])])
            else:
                result_df = pd.DataFrame({f"{col_prefix}": result})

            result_df.insert(0, "timestamp", df_1h["timestamp"])

            save_df_to_db(
                result_df,
                table_name=table_name,
                schema=SCHEMA,
                time_column="timestamp",
                is_timeseries=True
            )
            logger.info(f"Saved {func_name} to table {table_name}")

        except Exception as e:
            logger.error(f"Error computing or saving {func_name}: {e}")

signals_df = fetch_ohlcv_df(
    table_name="btc_cdldoji_1h",               # Table name in database
    schema=EXCHANGE_SCHEMA_MAP["signals"],  # Exchange schema
    time_column="timestamp"         # Column storing timestamp
)

bt = Backtester(
    price_1m_df=df_1m,      # from OHLCV schema
    signal_1h_df=signals_df,    # from signals schema
    tp=3,
    sl=1
)

