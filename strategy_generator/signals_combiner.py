import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from TradeX.indicators.talib.signals import candlestick_signal, SIGNAL_FUNCTIONS
from TradeX.utils.common.logs import get_logger

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
def run_active_signals_with_voting(flags, open_, high, low, close, volume, timestamps, window=14):
    """
    Compute signals for all active indicators using ThreadPoolExecutor
    and combine them with majority voting.

    Args:
        flags (dict): indicator_name -> True/False
        open_, high, low, close, volume: np.arrays
        timestamps: pd.Series or list
        window (int): timeperiod/period for indicators

    Returns:
        pd.DataFrame: timestamp + signals
    """
    signals_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    def compute_signal(name):
        try:
            # Candlestick patterns
            if name.startswith("CDL"):
                sig, _ = candlestick_signal(open_, high, low, close, name)
                return name, sig.astype(np.int8)

            # Regular indicators
            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None

            # Prepare arguments dynamically
            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]
            
            # Prepare kwargs with window override
            kwargs = {}
            if 'timeperiod' in func.__code__.co_varnames:
                kwargs['timeperiod'] = window
            elif 'period' in func.__code__.co_varnames:
                kwargs['period'] = window

            sig = func(*args, **kwargs)
            return name, sig.astype(np.int8)

        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None

    # Parallel execution
    active_indicators = [name for name, active in flags.items() if active]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, active_indicators)

    # Collect signals
    for name, sig in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig

    # Majority voting
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

    return pd.DataFrame({"datetime": timestamps, "signals": final_signal})
