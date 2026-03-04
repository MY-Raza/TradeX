import sys
import time
from datetime import datetime, timezone
from TradeX.utils.db.utils import get_profitable_strategies, fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.execution.binance.strategy_utils import (
    required_base_candles,
    analyze_strategy
)
from TradeX.execution.binance.strategy_signals_orchestrator import (
    execute_strategies_on_dataframe,
    get_latest_signals
)
import os
import subprocess
from TradeX.utils.common.config_loader import read_config
from TradeX.execution.binance.executor import FuturesTrader
from dotenv import load_dotenv
from TradeX.utils.db.utils import save_df_to_db
from TradeX.ai.ml.inference import run_inference

logger = get_logger("execution_binance_main")
SCHEMA = EXCHANGE_SCHEMA_MAP["binance"]

# Only run for BTC
symbols = ["btc"]

# -----------------------------
# Parse command-line argument
# -----------------------------
if len(sys.argv) < 2:
    logger.error("Usage: python main.py <timeframe> (e.g., 1h, 15m, 5m)")
    exit()

timeframe = sys.argv[1].lower()
logger.info(f"Running continuous strategy execution for timeframe: {timeframe}")

timeframe_minutes = {
    "1h": [0],
    "15m": [0, 15, 30, 45],
    "5m": list(range(0, 60, 5)),
    "1m": list(range(0, 60))
}

if timeframe not in timeframe_minutes:
    logger.error(f"Unsupported timeframe: {timeframe}")
    exit()

# -----------------------------
# Wait for next valid interval
# -----------------------------
def wait_for_next_interval(valid_minutes):
    while True:
        now = datetime.now(timezone.utc)
        minute = now.minute
        second = now.second

        if minute in valid_minutes:
            logger.info(f"Current time {now.strftime('%H:%M:%S')} is valid for {timeframe} execution.")
            break

        next_minute = min([m for m in valid_minutes if m > minute] + [valid_minutes[0] + 60])
        wait_seconds = (next_minute - minute) * 60 - second
        logger.info(f"Waiting {wait_seconds} seconds until next valid {timeframe} time...")
        time.sleep(wait_seconds)

# -----------------------------
# Load Binance credentials
# -----------------------------
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)
API_KEY = os.getenv("BINANCE_DEMO_API_KEY")
API_SECRET = os.getenv("BINANCE_DEMO_SECRET_KEY")
trader = FuturesTrader(API_KEY, API_SECRET, "BTCUSDT")

# Get symbol info once to handle precision
symbol_info = next(
    s for s in trader.client.futures_exchange_info()["symbols"]
    if s["symbol"] == "BTCUSDT"
)
step_size = float(symbol_info["filters"][2]["stepSize"])  # lot size step
def format_quantity(qty):
    return int(qty / step_size) * step_size  # round down to allowed precision

# -----------------------------
# Continuous loop
# -----------------------------
while True:
    try:
        wait_for_next_interval(timeframe_minutes[timeframe])

        # Optional: run the data preprocessing script
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "binance", "main.py")
        )
        subprocess.run([sys.executable, script_path])

        for symbol in symbols:
            logger.info(f"Processing symbol: {symbol}")

            USE_ML_MODEL = True
            MODEL_NAME = "xgboost_classifier_20260304_101010"
            model_predictions = None

            strategies = get_profitable_strategies(
                symbol="btc",
                timehorizon=timeframe,
                min_pnl=100,
                best="highest"
            )

            if not strategies:
                logger.warning(f"No profitable strategies found for {symbol}.")
                continue

            strategy_max_values = []
            active_strategies = {}

            for strategy in strategies:
                result = analyze_strategy(strategy)
                strategy_name = result["strategy_name"]
                max_window = result["max_window"]
                active_flags = result["active_flags"]

                active_strategies[strategy_name] = active_flags
                strategy_max_values.append(max_window)
                logger.info(f"{symbol} | {strategy_name} → Highest window: {max_window}")

            max_value = max(filter(None, strategy_max_values))

            required_1m = required_base_candles(
                target_tf=timeframe,
                base_tf="1m",
                window=max_value * 3
            )

            logger.info(f"{symbol} | Required 1m candles: {required_1m}")

            df_1m = fetch_ohlcv_df(
                table_name="btc_1m",
                schema=SCHEMA,
                time_column="datetime",
                limit=required_1m
            )
            if df_1m.empty:
                logger.warning(f"{symbol} | No 1m data fetched.")
                continue
            logger.info(f"{symbol} | Fetched {len(df_1m)} rows of 1m data.")

            df_resampled = resample_ohlcv(df=df_1m, interval=timeframe)
            if df_resampled.empty:
                logger.warning(f"{symbol} | Resampled {timeframe} dataframe is empty.")
                continue
            logger.info(f"{symbol} | Resampled to {len(df_resampled)} rows.")
            if USE_ML_MODEL:
                model_predictions = run_inference(model_name= MODEL_NAME, df_tf=df_resampled)
            results = execute_strategies_on_dataframe(df=df_resampled, strategies=strategies,use_model=USE_ML_MODEL,model_signals=model_predictions)
            if not results:
                logger.warning(f"{symbol} | No signals generated.")
                continue

            latest_signals = get_latest_signals(results)

            for strat, data in latest_signals.items():
                logger.info(f"{symbol} | Latest Signal: {strat} → {data['signal']} at {data['datetime']}")
                signal = data["signal"]
                if signal in [1, -1]:
                    qty = format_quantity(0.01)  # Adjust quantity to allowed precision
                    trader.process_signal(signal, quantity=qty)
        trade_df = trader.get_trade_log_df()
        if not trade_df.empty:
            save_df_to_db(
            df=trade_df,
            table_name="btc_trades",
            schema="execution",
            time_column="datetime",
            is_timeseries=True
        )
        # Wait 1 second before next iteration to avoid busy loop
        time.sleep(1)

    except Exception as e:
        logger.exception(f"Error in continuous execution loop: {e}")
        time.sleep(5)  # Wait a few seconds before retrying
