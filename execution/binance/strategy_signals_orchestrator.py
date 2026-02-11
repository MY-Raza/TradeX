"""
strategy_signal_orchestrator.py

Responsible for:
- Extracting active indicators from strategy rows
- Running signal voting
- Returning final signals per strategy
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from TradeX.strategy_generator.signals_combiner import run_active_signals_with_voting
from TradeX.indicators.talib.signals import SIGNAL_FUNCTIONS


# -------------------------------------------------------------
# Helper: Extract boolean indicator flags
# -------------------------------------------------------------
def extract_indicator_flags(strategy) -> Dict[str, bool]:
    """
    Extract only valid indicator boolean flags
    from a strategy row.
    """
    return {
        col: value
        for col, value in strategy.__dict__.items()
        if isinstance(value, bool)
        and col in SIGNAL_FUNCTIONS
    }


# -------------------------------------------------------------
# Core Engine
# -------------------------------------------------------------
def execute_strategies_on_dataframe(
    df: pd.DataFrame,
    strategies: List
) -> Dict[str, Dict]:
    """
    Executes all strategies on provided OHLCV dataframe.

    Args:
        df (pd.DataFrame): Resampled OHLCV dataframe (e.g., 1h)
        strategies (List): Strategy rows from DB

    Returns:
        dict:
        {
            strategy_name: {
                "signals_df": DataFrame,
                "windows": dict,
                "latest_signal": int
            }
        }
    """

    if df.empty:
        return {}

    required_cols = {"open", "high", "low", "close", "volume", "datetime"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Convert to numpy arrays once (performance optimization)
    open_ = df["open"].to_numpy(dtype=np.float32)
    high = df["high"].to_numpy(dtype=np.float32)
    low = df["low"].to_numpy(dtype=np.float32)
    close_ = df["close"].to_numpy(dtype=np.float32)
    volume = df["volume"].to_numpy(dtype=np.float32)
    timestamps = df["datetime"].to_numpy()

    results = {}

    for strategy in strategies:
        strategy_name = getattr(strategy, "strategy", "Unknown")

        flags = extract_indicator_flags(strategy)

        if not flags:
            continue

        final_df, windows_used = run_active_signals_with_voting(
            flags=flags,
            open_=open_,
            high=high,
            low=low,
            close_=close_,
            volume=volume,
            timestamps=timestamps
        )

        latest_signal = (
            int(final_df["signals"].iloc[-1])
            if not final_df.empty
            else 0
        )

        results[strategy_name] = {
            "signals_df": final_df,
            "windows": windows_used,
            "latest_signal": latest_signal
        }

    return results


# -------------------------------------------------------------
# Utility: Get only latest live signals
# -------------------------------------------------------------
def get_latest_signals(results: Dict[str, Dict]) -> Dict[str, int]:
    """
    Extract only the latest signal per strategy.

    Returns:
        {strategy_name: signal}
    """
    return {
        name: data["latest_signal"]
        for name, data in results.items()
    }
