import os
import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.backtest import BackTest
from TradeX.utils.db.utils import save_df_to_db,fetch_ohlcv_df
from TradeX.indicators.talib.indicators import ALL_INDICATORS

from strategy_counter import generate_strategy_id
from signals_combiner import randomize_indicators, run_active_signals_with_voting
from TradeX.utils.common.config_loader import read_config


# -------------------------------------------------
# Load Configuration
# -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
binance_config_path = os.path.join(current_dir, "config.yml")
config = read_config(binance_config_path)
symbols = config["symbols"]
default_start_date = config["start_date"]
end_date = config["end_date"]


# ============================
# Logger
# ============================
logger = get_logger("strategy_main")

# ============================
# Strategy configuration
# ============================
TIMEFRAMES = [
     "1h","15m","5m"]
RUNS_PER_TIMEFRAME = 7

# ============================
# Multi-Symbol Loop
# ============================
for symbol in symbols:

    logger.info(f"\n🚀 Processing Symbol: {symbol}")

    # -----------------------------------
    # Load 1-minute OHLCV for this symbol
    # -----------------------------------
    df_1m = fetch_ohlcv_df(
        table_name=f"{symbol.lower()}_1m",
        schema="data_binance",
        time_column="datetime",
    )

    if df_1m.empty:
        logger.warning(f"{symbol} OHLCV data empty. Skipping.")
        continue

    # ============================
    # Multi-timeframe loop
    # ============================
    for timeframe in TIMEFRAMES:
        logger.info(f"⏱ {symbol} | Timeframe: {timeframe}")

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
        # Strategy Runs
        # ----------------------------
        for run_idx in range(1, RUNS_PER_TIMEFRAME + 1):
            logger.info(
                f"{symbol} | {timeframe} | Strategy {run_idx}/{RUNS_PER_TIMEFRAME}"
            )

            flags = randomize_indicators(ALL_INDICATORS)

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

            # Make strategy ID symbol-aware
            strategy_id = generate_strategy_id(
                symbol,
                flags,
                timeframe=timeframe,
            )

            # Save signals
            save_df_to_db(
                df=signals,
                schema="strategy_signals",
                table_name=strategy_id,
                time_column="datetime",
                is_timeseries=True,
            )

            # ----------------------------
            # Backtest (1m execution)
            # ----------------------------
            bt = BackTest(
                df_price=df_1m,
                df_predictions=signals,
                starting_balance=1000,
                take_profit=3,
                stop_loss=1,
                fee=0.05,
                leverage=1,
                slippage=0,
            )

            ledger, final_balance, total_pnl_percent = bt.run()

            logger.info(
                f"{symbol} | {strategy_id} | "
                f"Balance={final_balance:.2f} | "
                f"PnL={total_pnl_percent:.2f}% | "
                f"Trades={len(ledger)}"
            )

            # ----------------------------
            # Save strategy metadata
            # ----------------------------
            row_data = {**flags}

            for ind_name, params in windows_dict.items():
                for param_name, value in params.items():
                    row_data[f"{ind_name}_{param_name}"] = value

            strategy_df = pd.DataFrame([row_data])

            strategy_df.insert(0, "pnl_sum", total_pnl_percent)
            strategy_df.insert(0, "timehorizon", timeframe)
            strategy_df.insert(0, "symbol", symbol.lower())
            strategy_df.insert(0, "sl", "1")
            strategy_df.insert(0, "tp", "3")
            strategy_df.insert(0, "strategy", strategy_id)

            strategy_df.columns = strategy_df.columns.str.lower()

            save_df_to_db(
                df=strategy_df,
                table_name="strategy_registry",
                schema="strategies",
                time_column=None,
                is_timeseries=False,
            )

logger.info("\nAll symbols and timeframes completed successfully.")

