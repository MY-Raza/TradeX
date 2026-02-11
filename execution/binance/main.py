import sys
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

# ------------------------------------------------------------
# 0️⃣ Parse command-line argument for timeframe
# ------------------------------------------------------------
if len(sys.argv) < 2:
    logger.error("Usage: python main.py <timeframe> (e.g., 1h, 15m)")
    exit()

timeframe = sys.argv[1]  # e.g., "1h"
logger.info(f"Running strategy execution for timeframe: {timeframe}")


# ============================================================
# 1️⃣ Fetch profitable strategies
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
# 2️⃣ Analyze strategies → find max required window
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

    logger.info(
        f"Strategy {strategy_name} → Highest window value: {max_window}"
    )


# ============================================================
# 3️⃣ Compute required 1m candles
# ============================================================
max_value = max(filter(None, strategy_max_values))
CHECK_PARAMETER = 60
required_1m = required_base_candles(
    target_tf=timeframe,
    base_tf="1m",
    window=max_value + CHECK_PARAMETER
)

logger.info(f"Required 1m candles: {required_1m}")


# ============================================================
# 4️⃣ Fetch latest 1m candles
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
# 5️⃣ Resample to target timeframe
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
# 6️⃣ Execute strategies on resampled data
# ============================================================
results = execute_strategies_on_dataframe(
    df=df_resampled,
    strategies=strategies
)

if not results:
    logger.warning("No signals generated.")
    exit()


# ============================================================
# 7️⃣ Get latest live signals
# ============================================================
latest_signals = get_latest_signals(results)
logger.info(f"Latest Signals: {latest_signals}")
