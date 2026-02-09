import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from TradeX.indicators.talib.signals import candlestick_signal, SIGNAL_FUNCTIONS
from TradeX.utils.common.logs import get_logger

logger = get_logger("signals_combiner")


def randomize_indicators(all_indicators):
    flags_array = np.random.choice([True, False], size=len(all_indicators))
    return dict(zip(all_indicators, flags_array))


def run_active_signals_with_voting(flags, open_, high, low, close_, volume, timestamps):
    """
    Computes signals for all active indicators in parallel and combines them
    into a final signal using majority voting.

    Returns:
        tuple: (DataFrame with final signals, windows dict with unpacked params)
    """
    signals_dict = {}
    windows_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close_, "volume": volume}

    # ---------------------------
    # Indicator computation
    # ---------------------------
    def compute_signal(name):
        try:
            # Candlestick patterns
            if name.startswith("CDL"):
                sig, window = candlestick_signal(open_, high, low, close_, name)
                return name, sig.astype(np.int8), {"window": window}

            # Regular indicators
            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None, None

            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]
            result = func(*args)

            # Extract signal and window
            if isinstance(result, tuple) and len(result) == 2:
                signal_array, window = result
            else:
                signal_array, window = result, None

            # ---------------------------
            # Unpack window parameters
            # ---------------------------
            unpacked_window = {}
            if window is not None:
                if isinstance(window, (list, tuple)):
                    # MACD / MACDEXT
                    if ("MACD" in name or "PPO" in name) and len(window) == 3:
                        unpacked_window = {"fastperiod": window[0], "slowperiod": window[1], "signalperiod": window[2]}
                    # ADOSC
                    elif "ADOSC" in name and len(window) >= 2:
                        unpacked_window = {"fastperiod": window[0], "slowperiod": window[1]}
                    # STOCH-style
                    elif any(x in name for x in ["STOCH", "STOCHF", "STOCHRSI"]) and len(window) >= 3:
                        unpacked_window = {"fastk_period": window[0], "slowk_period": window[1], "slowd_period": window[2]}
                    # Fallback for other multi-value windows
                    else:
                        unpacked_window = {f"param{i}": w for i, w in enumerate(window)}
                else:
                    # Single-period indicators
                    unpacked_window = {"period": window}

            return name, signal_array.astype(np.int8), unpacked_window

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
    for name, sig, window_params in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig

            # Only store meaningful window params
            if window_params:
                # Filter out None or 0 values
                filtered_window = {k: v for k, v in window_params.items() if v not in (None, 0)}
                if filtered_window:
                    windows_dict[name] = filtered_window

    # ---------------------------
    # Majority voting
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

