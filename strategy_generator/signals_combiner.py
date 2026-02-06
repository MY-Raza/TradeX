import os
from concurrent.futures import ThreadPoolExecutor
from TradeX.indicators.talib.signals import candlestick_signal, SIGNAL_FUNCTIONS
from TradeX.utils.common.logs import get_logger
import pandas as pd
import numpy as np

logger = get_logger("signals_combiner")

# ============================
# Randomize indicators
# ============================
def randomize_indicators(all_indicators):
    """
    Randomly activate or deactivate a list of indicators.

    Args:
        all_indicators (tuple/list): List of all indicator names.

    Returns:
        dict: Mapping of indicator name -> True/False (active/inactive)
    """
    flags_array = np.random.choice([True, False], size=len(all_indicators))
    return dict(zip(all_indicators, flags_array))


# ============================
# Compute active signals with voting
# ============================
def run_active_signals_with_voting(flags, open_, high, low, close_, volume, timestamps):
    """
    Computes signals for all active indicators in parallel using ThreadPoolExecutor
    and combines them into a single final signal using majority voting.

    Returns:
        tuple: (DataFrame with final signals, windows dict)
    """
    signals_dict = {}  # Stores signals for each indicator
    windows_dict = {}  # Stores window/period per function
    data = {"open": open_, "high": high, "low": low, "close": close_, "volume": volume}

    def compute_signal(name):
        """
        Compute the signal for a single indicator.

        Returns:
            tuple: (indicator_name, function_name, signal_array, window)
        """
        try:
            # ---------------------------
            # Candlestick patterns
            # ---------------------------
            if name.startswith("CDL"):
                sig, _ = candlestick_signal(open_, high, low, close_, name)
                return name, "candlestick_signal", sig.astype(np.int8), None

            # ---------------------------
            # Regular indicators
            # ---------------------------
            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None, None, None

            # Select arguments dynamically based on function signature
            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]

            # Call the indicator function
            sig = func(*args)

            # If function returns (signal_array, window) tuple
            if isinstance(sig, tuple) and len(sig) == 2:
                signal_array, window = sig
            else:
                signal_array, window = sig, None

            return name, func.__name__, signal_array.astype(np.int8), window

        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None, None, None

    # ---------------------------
    # Parallel computation of active indicators
    # ---------------------------
    active_indicators = [name for name, active in flags.items() if active]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, active_indicators)

    # ---------------------------
    # Collect results
    # ---------------------------
    for name, func_name, sig, window in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig

            if window is not None:
                base_name = func_name.replace("_signal", "")
                windows_dict[f"{base_name}_window"] = window

    # ---------------------------
    # Voting to generate final signal
    # ---------------------------
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values()))
        buy_votes = np.sum(all_signals == 1, axis=1)
        sell_votes = np.sum(all_signals == -1, axis=1)
        final_signal = np.where(
            buy_votes > sell_votes, 1,
            np.where(sell_votes > buy_votes, -1, 0)
        ).astype(np.int8)
    else:
        final_signal = np.zeros(len(timestamps), dtype=np.int8)

    # Return final DataFrame and windows dict
    return pd.DataFrame({"datetime": timestamps, "signals": final_signal}), windows_dict
