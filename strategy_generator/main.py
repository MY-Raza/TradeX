import os
import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.newbacktest import HighPerfBacktest
from TradeX.utils.db.utils import save_df_to_db,fetch_ohlcv_df
from TradeX.indicators.talib.indicators import ALL_INDICATORS

from strategy_counter import generate_strategy_id
from signals_combiner import randomize_indicators, run_active_signals_with_voting

# ============================
# Logger
# ============================
logger = get_logger("strategy_main")
# strategies = get_profitable_strategies(100)

# for strategy in strategies:
#     # iterate all dynamic columns
#     for col, value in strategy.__dict__.items():
#         print(f"{col} → {value}")



# ============================
# Strategy configuration
# ============================
TIMEFRAMES = [
     "5m"]
RUNS_PER_TIMEFRAME = 50

# ============================
# Load 1-minute OHLCV
# ============================
df_1m = fetch_ohlcv_df(
    table_name="btc_1m",
    schema="data_binance",
    time_column="datetime",
)

if df_1m.empty:
    logger.error("OHLCV data empty. Exiting.")
    raise SystemExit
# ============================
# Multi-timeframe strategy loop
# ============================
for timeframe in TIMEFRAMES:
    logger.info(f"\n⏱ Processing timeframe: {timeframe}")

    # ----------------------------
    # Resample OHLCV
    # ----------------------------
    df_tf = resample_ohlcv(df_1m, timeframe)
    open_ = df_tf["open"].values
    high = df_tf["high"].values
    low = df_tf["low"].values
    close_ = df_tf["close"].values
    volume = df_tf["volume"].values
    timestamps = df_tf["datetime"]

    # ----------------------------
    # Run strategies for timeframe
    # ----------------------------
    for run_idx in range(1, RUNS_PER_TIMEFRAME + 1):
        logger.info(f" {timeframe} | Strategy {run_idx}/{RUNS_PER_TIMEFRAME}")

        # Random indicators
        flags = randomize_indicators(ALL_INDICATORS)

        # Generate signals
        signals, windows_dict = run_active_signals_with_voting(
            flags,
            open_,
            high,
            low,
            close_,
            volume,
            timestamps,
        )

        if signals.empty:
            logger.warning("Empty signals — skipping.")
            continue

        # Strategy ID (timeframe-aware)
        strategy_id = generate_strategy_id(flags, timeframe=timeframe)
        # Save signals
        save_df_to_db(
            df=signals,
            schema="strategy_signals",
            table_name=strategy_id,
            time_column="datetime",
            is_timeseries=True
        )

        # Backtest (still uses 1m execution)
        bt = HighPerfBacktest(
            df_price=df_1m,
            df_predictions=signals,
            starting_balance=1000,
            take_profit=3,
            stop_loss=1,
            fee=0.05,
            leverage=1,
            slippage=0
        )

        ledger, final_balance, total_pnl_percent = bt.run()

        # Log result
        logger.info(
            f" {strategy_id} | Balance={final_balance:.2f} | "
            f"PnL={total_pnl_percent:.2f}% | Trades={len(ledger)}"
        )
        # Flatten windows_dict into row_data
        row_data = {**flags}
        for ind_name, params in windows_dict.items():
             for param_name, value in params.items():
        # e.g., MACD_window.fastperiod → 'MACD_fastperiod'
                row_data[f"{ind_name}_{param_name}"] = value
        strategy_df = pd.DataFrame([row_data])
        strategy_df.insert(0, "pnl_sum", total_pnl_percent)
        strategy_df.insert(0, "timehorizon", timeframe)
        strategy_df.insert(0, "symbol", "btc")
        strategy_df.insert(0, "sl", "1")
        strategy_df.insert(0, "tp", "3")
        strategy_df.insert(0, "strategy", strategy_id)
        strategy_df.columns = strategy_df.columns.str.lower()
        


        save_df_to_db(
            df=strategy_df,
            table_name="strategy_registry",
            schema="strategies",
            time_column=None,
            is_timeseries=False
        )

logger.info("\n All timeframes completed successfully.")
