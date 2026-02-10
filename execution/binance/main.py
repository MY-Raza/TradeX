from TradeX.utils.db.utils import get_profitable_strategies
from TradeX.utils.common.logs import get_logger
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
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

for i, strategy in enumerate(strategies, start=1):
    # Get all attributes of the strategy
    attr_dict = strategy.__dict__

    # Filter numeric values where the key ends with one of the patterns
    numeric_values = [
        v for k, v in attr_dict.items()
        if any(k.endswith(pat) for pat in ending_patterns) and isinstance(v, (int, float))
    ]

    # Determine max value safely
    max_value = max(numeric_values) if numeric_values else None

    strategy_name = getattr(strategy, "strategy", "Unknown")  # replace "strategy" with actual column name
    logger.info(f"Strategy {strategy_name} → Highest window value among specified patterns: {max_value}")

script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # execution/binance/
        "..", "..", "data", "binance", "main.py"  # relative path to data/binance/main.py
    )
)

subprocess.run([sys.executable, script_path])