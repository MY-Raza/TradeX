from TradeX.utils.db.utils import get_profitable_strategies
from TradeX.utils.common.logs import get_logger
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from TradeX.utils.db.utils import fetch_ohlcv_df
from TradeX.utils.data.data_cleaner import resample_ohlcv
import os
import subprocess
import sys

logger = get_logger("execution_binance_main")

SCHEMA = EXCHANGE_SCHEMA_MAP["binance"]
strategies = get_profitable_strategies(timehorizon="1h",min_pnl=100)

# The ending patterns you care about
ending_patterns = [
    '_period', '_fastperiod', '_slowperiod', '_signalperiod',
    '_slowd_period', '_slowk_period', '_fastk_period'
]

strategy_max_values = []
active_strategies = {}
for i, strategy in enumerate(strategies, start=1):
    # Get all attributes of the strategy
    attr_dict = strategy.__dict__

    # Filter numeric values where the key ends with one of the patterns
    numeric_values = [
        v for k, v in attr_dict.items()
        if any(k.endswith(pat) for pat in ending_patterns) and isinstance(v, (int, float))
    ]
    true_columns = [
        col for col, value in strategy.__dict__.items()
        if value is True
    ]

    # Determine max value safely
    max_value = max(numeric_values) if numeric_values else None
    strategy_name = getattr(strategy, "strategy", "Unknown")
    active_strategies[strategy_name] = true_columns

      # replace "strategy" with actual column name
    logger.info(f"Strategy {strategy_name} → Highest window value among specified patterns: {max_value}")

def required_base_candles(
    target_tf: str,
    base_tf: str,
    window: int
) -> int:
    """
    Calculate required number of base timeframe candles
    to compute indicators after resampling.

    Example:
        target_tf = "1h"
        base_tf   = "1m"
        window    = 30
        → 1800 candles
    """

    TF_TO_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    if target_tf not in TF_TO_MINUTES or base_tf not in TF_TO_MINUTES:
        raise ValueError("Unsupported timeframe")

    target_minutes = TF_TO_MINUTES[target_tf]
    base_minutes = TF_TO_MINUTES[base_tf]

    if target_minutes % base_minutes != 0:
        raise ValueError("Base timeframe must divide target timeframe")

    candles_per_target = target_minutes // base_minutes

    return window * candles_per_target

required_1m = required_base_candles(
    target_tf="1h",
    base_tf="1m",
    window=max_value
)

logger.info(f"Required 1m candles: {required_1m}")

df_1m = fetch_ohlcv_df(
    table_name="btc_1m",
    schema=SCHEMA,
    time_column="datetime",
    limit=required_1m
)

print(df_1m.head())
print(active_strategies)
df_1h = resample_ohlcv(
    df=df_1m,
    interval="1h"
)
print(df_1h.head())
script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # execution/binance/
        "..", "..", "data", "binance", "main.py"  # relative path to data/binance/main.py
    )
)

subprocess.run([sys.executable, script_path])