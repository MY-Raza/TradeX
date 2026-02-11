from typing import Dict, List, Any


ENDING_PATTERNS = [
    '_period', '_fastperiod', '_slowperiod', '_signalperiod',
    '_slowd_period', '_slowk_period', '_fastk_period'
]


def analyze_strategy(strategy: Any) -> Dict[str, Any]:
    """
    Analyze a strategy object to extract:
    - Strategy name
    - Active boolean flags
    - Maximum indicator window value based on known parameter suffixes
    """

    attr_dict = strategy.__dict__

    # Get numeric indicator window values
    numeric_values = [
        v for k, v in attr_dict.items()
        if any(k.endswith(pat) for pat in ENDING_PATTERNS) and isinstance(v, (int, float))
    ]

    # Get active boolean strategy components
    true_columns = [
        col for col, value in attr_dict.items()
        if value is True
    ]

    max_value = max(numeric_values) if numeric_values else None
    strategy_name = getattr(strategy, "strategy", "Unknown")

    return {
        "strategy_name": strategy_name,
        "max_window": max_value,
        "active_flags": true_columns
    }


def required_base_candles(target_tf: str, base_tf: str, window: int) -> int:
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
