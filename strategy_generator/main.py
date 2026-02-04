import os
import pandas as pd

from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.newbacktest import HighPerfBacktest
from TradeX.utils.db.utils import save_df_to_db

from strategy_counter import generate_strategy_id
from signals_combiner import randomize_indicators, run_active_signals_with_voting

# ============================
# Logger
# ============================
logger = get_logger("strategy_main")

# ============================
# Strategy configuration
# ============================
TIMEFRAMES = ["1h", "15m", "5m"]
RUNS_PER_TIMEFRAME = 50

# ============================
# Indicators
# ============================
ALL_INDICATORS = (
    "BBANDS", "DEMA", "EMA", "HT_TRENDLINE", "KAMA",
    "MA", "MAMA", "MIDPOINT", "MIDPRICE", "SAR",
    "SAREXT", "SMA", "T3", "TEMA", "TRIMA", "WMA",
    "ADX", "ADXR", "APO", "AROON", "AROONOSC",
    "BOP", "CCI", "CMO", "DX", "MACD", "MACDEXT",
    "MACDFIX", "MFI", "MINUS_DI", "MINUS_DM",
    "MOM", "PLUS_DI", "PLUS_DM", "PPO", "ROC",
    "ROCP", "ROCR", "ROCR100", "RSI", "STOCH",
    "STOCHF", "STOCHRSI", "TRIX", "ULTOSC",
    "WILLR", "AD", "ADOSC", "OBV", "ATR", "NATR",
    "TRANGE", "AVGPRICE", "MEDPRICE", "TYPPRICE",
    "WCLPRICE", "HT_DCPERIOD", "HT_DCPHASE",
    "HT_PHASOR", "HT_SINE", "HT_TRENDMODE",
    "LINEARREG", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "LINEARREG_SLOPE",
    "STDDEV", "TSF", "VAR"
)

# ============================
# Load 1-minute OHLCV
# ============================
INPUT_CSV = r"D:\trading\TradeX\indicators\talib\btc_1m_data.csv"
df_1m = pd.read_csv(INPUT_CSV)

if df_1m.empty:
    logger.error("OHLCV data empty. Exiting.")
    raise SystemExit

df_1m["timestamp"] = pd.to_datetime(df_1m["datetime"])
df_1m.drop(columns=["datetime"], inplace=True)

# ============================
# Output directory
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FOLDER = os.path.join(BASE_DIR, "strategy_csv")
os.makedirs(SIGNALS_FOLDER, exist_ok=True)

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
    close = df_tf["close"].values
    volume = df_tf["volume"].values
    timestamps = df_tf["timestamp"]

    # ----------------------------
    # Run strategies for timeframe
    # ----------------------------
    for run_idx in range(1, RUNS_PER_TIMEFRAME + 1):
        logger.info(f" {timeframe} | Strategy {run_idx}/{RUNS_PER_TIMEFRAME}")

        # Random indicators
        flags = randomize_indicators(ALL_INDICATORS)

        # Generate signals
        signals = run_active_signals_with_voting(
            flags,
            open_,
            high,
            low,
            close,
            volume,
            timestamps
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
            time_column="timestamp",
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

        # Save ledger CSV
        ledger.to_csv(
            os.path.join(SIGNALS_FOLDER, f"{strategy_id}_ledger.csv"),
            index=False
        )

        # Save strategy metadata
        strategy_df = pd.DataFrame([flags])
        strategy_df.insert(0, "timehorizon", timeframe)
        strategy_df.insert(0, "symbol", "btc")
        strategy_df.insert(0, "s1", "1")
        strategy_df.insert(0, "tp", "3")
        strategy_df.insert(0, "strategy", strategy_id)
        strategy_df.columns = strategy_df.columns.str.lower()

        save_df_to_db(
            df=strategy_df,
            table_name="strategy_registry",
            schema="strategy_identifier",
            time_column="strategy",
            is_timeseries=False
        )

logger.info("\n All timeframes completed successfully.")
