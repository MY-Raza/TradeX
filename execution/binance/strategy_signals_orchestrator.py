"""
strategy_signal_orchestrator.py

Responsible for:
- Extracting active indicators from strategy rows
- Running signal voting
- Returning final signals per strategy
"""

from typing import List, Dict
import numpy as np
import pandas as pd

from TradeX.strategy_generator.signals_combiner import run_active_signals_with_voting
from TradeX.indicators.talib.signals import SIGNAL_FUNCTIONS


# -------------------------------------------------------------
# Helper: Extract boolean indicator flags
# -------------------------------------------------------------
def extract_indicator_flags(strategy) -> Dict[str, bool]:
    """
    Extract only valid indicator boolean flags from a strategy row.
    Maps lowercase strategy columns to SIGNAL_FUNCTIONS keys (case-insensitive).
    Preserves candlestick patterns separately.
    """
    flags = {}

    for col, value in strategy.__dict__.items():
        if not isinstance(value, bool) or not value:
            continue  # skip non-bool or False flags

        col_lower = col.lower()

        # Match against SIGNAL_FUNCTIONS (case-insensitive)
        matched_signal = next(
            (k for k in SIGNAL_FUNCTIONS if k.lower() == col_lower), None
        )
        if matched_signal:
            flags[matched_signal] = True
        # Special handling for candlestick patterns (assume column name starts with 'cdl')
        elif col_lower.startswith("cdl"):
            flags[col] = True

    print(f"[DEBUG] Strategy '{getattr(strategy, 'strategy', 'Unknown')}' flags extracted: {flags}")
    return flags


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
        print("[DEBUG] Input dataframe is empty.")
        return {}

    required_cols = {"open", "high", "low", "close", "volume", "datetime"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Convert OHLCV to numpy arrays
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    close_ = df["close"].values
    volume = df["volume"].values
    timestamps = df["datetime"]
    print(f"[DEBUG] OHLCV arrays prepared: {len(open_)} rows")

    results = {}

    for strategy in strategies:
        strategy_name = getattr(strategy, "strategy", "Unknown")
        print(f"\n[DEBUG] Processing strategy: {strategy_name}")

        flags = extract_indicator_flags(strategy)

        if not flags:
            print(f"[DEBUG] Strategy '{strategy_name}' has no valid active indicators. Skipping.")
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

        print(f"[DEBUG] Strategy '{strategy_name}' windows used: {windows_used}")
        print(f"[DEBUG] Strategy '{strategy_name}' signals dataframe length: {len(final_df)}")

        latest_signal = int(final_df["signals"].iloc[-1]) if not final_df.empty else 0
        print(f"[DEBUG] Strategy '{strategy_name}' latest signal: {latest_signal}")

        results[strategy_name] = {
            "signals_df": final_df,
            "windows": windows_used,
            "latest_signal": latest_signal
        }

    print(f"\n[DEBUG] Total strategies processed: {len(results)}")
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
    latest = {name: data["latest_signal"] for name, data in results.items()}
    print(f"[DEBUG] Latest signals extracted: {latest}")
    return latest
