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
    
    Each indicator is assigned either True (active) or False (inactive).

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
def run_active_signals_with_voting(flags, open_, high, low, close, volume, timestamps):
    """
    Computes signals for all active indicators in parallel using ThreadPoolExecutor
    and combines them into a single final signal using majority voting.

    Voting logic:
        - If majority of indicators give 'buy' (1) -> final signal = 1
        - If majority give 'sell' (-1) -> final signal = -1
        - If tie or no signal -> final signal = 0

    Args:
        flags (dict): Mapping of indicator name -> True/False (active/inactive)
        open_ (np.array): Open prices
        high (np.array): High prices
        low (np.array): Low prices
        close (np.array): Close prices
        volume (np.array): Volume data
        timestamps (pd.Series or list): Timestamps corresponding to OHLCV data

    Returns:
        pd.DataFrame: DataFrame with columns:
            - timestamp: Timestamps of data
            - signals: Final voting signal (-1, 0, 1)
    """
    signals_dict = {}  # Stores signals for each individual indicator
    data = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    def compute_signal(name):
        """
        Compute the signal for a single indicator.

        Candlestick indicators are handled separately via candlestick_signal().
        Other indicators use the SIGNAL_FUNCTIONS mapping.

        Returns:
            tuple: (indicator_name, signal_array) or (None, None) on error
        """
        try:
            # ---------------------------
            # Candlestick patterns
            # ---------------------------
            if name.startswith("CDL"):
                sig, _ = candlestick_signal(open_, high, low, close, name)
                return name, sig.astype(np.int8)

            # ---------------------------
            # Regular indicators
            # ---------------------------
            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None

            # Select arguments dynamically based on function signature
            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]

            # Call the indicator function
            sig = func(*args)
            return name, sig.astype(np.int8)

        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None

    # ---------------------------
    # Parallel computation of active indicators
    # ---------------------------
    active_indicators = [name for name, active in flags.items() if active]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, active_indicators)

    # Collect results
    for name, sig in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig

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
        # If no active signals, output zeros
        final_signal = np.zeros(len(timestamps), dtype=np.int8)

    # Return as DataFrame
    return pd.DataFrame({"timestamp": timestamps, "signals": final_signal})
