from TradeX.utils.db.utils import get_profitable_strategies, fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.execution.binance.strategy_utils import required_base_candles, analyze_strategy

import os
import subprocess
import sys

logger = get_logger("execution_binance_main")

SCHEMA = EXCHANGE_SCHEMA_MAP["binance"]

# Fetch strategies
strategies = get_profitable_strategies(timehorizon="1h", min_pnl=100)

strategy_max_values = []
active_strategies = {}

# 🔍 Analyze each strategy
for strategy in strategies:
    result = analyze_strategy(strategy)

    strategy_name = result["strategy_name"]
    max_window = result["max_window"]
    active_flags = result["active_flags"]

    active_strategies[strategy_name] = active_flags
    strategy_max_values.append(max_window)

    logger.info(
        f"Strategy {strategy_name} → Highest window value among specified patterns: {max_window}"
    )

# 🧠 Use the LARGEST window required across all strategies
max_value = max(filter(None, strategy_max_values))

required_1m = required_base_candles(
    target_tf="1h",
    base_tf="1m",
    window=max_value
)

logger.info(f"Required 1m candles: {required_1m}")

# 📊 Fetch base timeframe data
df_1m = fetch_ohlcv_df(
    table_name="btc_1m",
    schema=SCHEMA,
    time_column="datetime",
    limit=required_1m
)

print(df_1m.head())
print("Active strategies:", active_strategies)

# 🔁 Resample to 1h
df_1h = resample_ohlcv(
    df=df_1m,
    interval="1h"
)

print(df_1h.head())

# ▶ Run data pipeline script
script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "data", "binance", "main.py"
    )
)

subprocess.run([sys.executable, script_path])
