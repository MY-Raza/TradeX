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
# Compute signals in parallel
# ============================
def run_active_signals_with_voting(flags, open_, high, low, close, volume, timestamps):
    signals_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    def compute_signal(name):
        try:
            if name.startswith("CDL"):
                sig, _ = candlestick_signal(open_, high, low, close, name)
                return name, sig.astype(np.int8)
            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                return None, None
            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]
            sig = func(*args)
            return name, sig.astype(np.int8)
        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, [n for n, a in flags.items() if a])

    for name, sig in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig

    # Voting
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values()))
        buy_votes = np.sum(all_signals == 1, axis=1)
        sell_votes = np.sum(all_signals == -1, axis=1)
        final_signal = np.where(buy_votes > sell_votes, 1,
                        np.where(sell_votes > buy_votes, -1, 0)).astype(np.int8)
    else:
        final_signal = np.zeros(len(timestamps), dtype=np.int8)

    return pd.DataFrame({"timestamp": timestamps, "signals": final_signal})