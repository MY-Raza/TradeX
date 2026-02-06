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
    flags_array = np.random.choice([True, False], size=len(all_indicators))
    return dict(zip(all_indicators, flags_array))


# ============================
# Compute active signals with voting
# ============================
def run_active_signals_with_voting(flags, open_, high, low, close_, volume, timestamps):
    """
    Computes signals for all active indicators in parallel and combines them
    into a final signal using majority voting.

    Returns:
        tuple: (DataFrame with final signals, windows dict)
    """
    signals_dict = {}
    windows_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close_, "volume": volume}

    # ---------------------------
    # Indicators that require special handling
    # ---------------------------
    def compute_signal(name):
        try:
            # ---------------------------
            # Candlestick
            # ---------------------------
            if name.startswith("CDL"):
                sig, window = candlestick_signal(open_, high, low, close_, name)
                return name, sig.astype(np.int8), window

            # ---------------------------
            # Regular indicators
            # ---------------------------
            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None, None

            # Select arguments dynamically
            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]
            result = func(*args)

            # If tuple returned: (signal_array, window)
            if isinstance(result, tuple):
                if len(result) == 2:
                    signal_array, window = result
                else:
                    signal_array, window = result[0], None
            else:
                signal_array, window = result, None

            return name, signal_array.astype(np.int8), window

        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None, None

    # ---------------------------
    # Run active indicators in parallel
    # ---------------------------
    active_indicators = [name for name, active in flags.items() if active]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, active_indicators)

    # ---------------------------
    # Collect results
    # ---------------------------
    for name, sig, window in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig
            if window is not None:
                # Convert rsi_signal -> rsi_window
                if name.endswith("_signal"):
                    base_name = name[:-7]
                else:
                    base_name = name
                windows_dict[f"{base_name}_window"] = window

    # ---------------------------
    # Voting
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

    return pd.DataFrame({"datetime": timestamps, "signals": final_signal}), windows_dict
