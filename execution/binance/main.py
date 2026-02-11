import sys
import time
from datetime import datetime, timedelta
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

logger = get_logger("execution_binance_main")
SCHEMA = EXCHANGE_SCHEMA_MAP["binance"]

# -----------------------------
# 0️⃣ Parse command-line argument
# -----------------------------
if len(sys.argv) < 2:
    logger.error("Usage: python main.py <timeframe> (e.g., 1h, 15m, 5m)")
    exit()

timeframe = sys.argv[1].lower()
logger.info(f"Running strategy execution for timeframe: {timeframe}")

# -----------------------------
# 1️⃣ Allowed minutes for each timeframe
# -----------------------------
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
# 2️⃣ Wait until current time is valid for timeframe
# -----------------------------
def wait_for_next_interval(valid_minutes):
    while True:
        now = datetime.utcnow()
        minute = now.minute
        second = now.second

        # If current minute is valid, break the loop
        if minute in valid_minutes:
            logger.info(f"Current time {now.strftime('%H:%M:%S')} is valid for {timeframe} execution.")
            break

        # Calculate seconds until next valid minute
        next_minute = min([m for m in valid_minutes if m > minute] + [valid_minutes[0] + 60])
        wait_seconds = (next_minute - minute) * 60 - second
        logger.info(f"Waiting {wait_seconds} seconds until next valid {timeframe} time...")
        time.sleep(wait_seconds)

wait_for_next_interval(timeframe_minutes[timeframe])

# ============================================================
# 3️⃣ Fetch profitable strategies
# ============================================================
strategies = get_profitable_strategies(
    timehorizon=timeframe,
    min_pnl=100,
    best="lowest"
)

if not strategies:
    logger.warning("No profitable strategies found.")
    exit()

# ============================================================
# 4️⃣ Analyze strategies → find max required window
# ============================================================
strategy_max_values = []
active_strategies = {}

for strategy in strategies:
    result = analyze_strategy(strategy)

    strategy_name = result["strategy_name"]
    max_window = result["max_window"]
    active_flags = result["active_flags"]

    active_strategies[strategy_name] = active_flags
    strategy_max_values.append(max_window)

    logger.info(f"Strategy {strategy_name} → Highest window value: {max_window}")

# ============================================================
# 5️⃣ Compute required 1m candles
# ============================================================
max_value = max(filter(None, strategy_max_values))
required_1m = required_base_candles(
    target_tf=timeframe,
    base_tf="1m",
    window=max_value
)
logger.info(f"Required 1m candles: {required_1m}")

# ============================================================
# 6️⃣ Fetch latest 1m candles
# ============================================================
df_1m = fetch_ohlcv_df(
    table_name="btc_1m",
    schema=SCHEMA,
    time_column="datetime",
    limit=required_1m
)

if df_1m.empty:
    logger.warning("No 1m data fetched.")
    exit()

logger.info(f"Fetched {len(df_1m)} rows of 1m data.")

# ============================================================
# 7️⃣ Resample to target timeframe
# ============================================================
df_resampled = resample_ohlcv(
    df=df_1m,
    interval=timeframe
)

if df_resampled.empty:
    logger.warning(f"Resampled {timeframe} dataframe is empty.")
    exit()

logger.info(f"Resampled to {len(df_resampled)} rows of {timeframe} data.")

# ============================================================
# 8️⃣ Execute strategies on resampled data
# ============================================================
results = execute_strategies_on_dataframe(
    df=df_resampled,
    strategies=strategies
)

if not results:
    logger.warning("No signals generated.")
    exit()

# ============================================================
# 9️⃣ Get latest live signals
# ============================================================
latest_signals = get_latest_signals(results)
logger.info(f"Latest Signals: {latest_signals}")
