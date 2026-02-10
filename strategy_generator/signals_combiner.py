import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Signal generators
from TradeX.indicators.talib.signals import candlestick_signal, SIGNAL_FUNCTIONS

# Centralized logger
from TradeX.utils.common.logs import get_logger

logger = get_logger("signals_combiner")


def randomize_indicators(all_indicators):
    """
    Randomly activates or deactivates indicators.

    This is mainly used for:
    - Strategy randomization
    - Genetic / stochastic strategy generation
    - Monte Carlo style experimentation

    Args:
        all_indicators (list[str]): List of indicator names.

    Returns:
        dict[str, bool]: Mapping of indicator name → active flag.
    """
    # Random True/False flag for each indicator
    flags_array = np.random.choice([True, False], size=len(all_indicators))

    # Convert array into {indicator_name: is_active}
    return dict(zip(all_indicators, flags_array))


def run_active_signals_with_voting(
    flags,
    open_,
    high,
    low,
    close_,
    volume,
    timestamps
):
    """
    Executes all active indicators in parallel and combines their outputs
    using majority voting to produce a final trading signal.

    Signal convention:
        +1 → Buy
        -1 → Sell
         0 → Neutral / No consensus

    NaNs in indicator outputs are dropped before voting.

    Args:
        flags (dict[str, bool]): Indicator activation map.
        open_, high, low, close_, volume (np.ndarray): OHLCV market data.
        timestamps (array-like): Datetime index for alignment.

    Returns:
        tuple:
            - pd.DataFrame: Columns = ["datetime", "signals"]
            - dict: Indicator → window/parameter configuration
    """

    # Holds raw signals from each indicator
    signals_dict = {}
    windows_dict = {}

    # Centralized data dictionary so indicators can auto-pick arguments
    data = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close_,
        "volume": volume
    }

    def compute_signal(name):
        """Compute a single indicator signal."""
        try:
            if name.startswith("CDL"):
                sig, window = candlestick_signal(open_, high, low, close_, name)
                return name, sig.astype(np.float32), {"window": window}  # float for NaN support

            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None, None

            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]
            result = func(*args)

            if isinstance(result, tuple) and len(result) == 2:
                signal_array, window = result
            else:
                signal_array, window = result, None

            unpacked_window = {}
            if window is not None:
                if isinstance(window, (list, tuple)):
                    if "MACD" in name and len(window) == 3:
                        unpacked_window = {"fastperiod": window[0], "slowperiod": window[1], "signalperiod": window[2]}
                    elif any(x in name for x in ["ADOSC", "PPO", "STOCHF"]) and len(window) >= 2:
                        unpacked_window = {"fastperiod": window[0], "slowperiod": window[1]}
                    elif any(x in name for x in ["STOCH", "STOCHRSI"]) and len(window) >= 3:
                        unpacked_window = {"fastk_period": window[0], "slowk_period": window[1], "slowd_period": window[2]}
                    else:
                        unpacked_window = {f"param{i}": w for i, w in enumerate(window)}
                else:
                    unpacked_window = {"period": window}

            return name, signal_array.astype(np.float32), unpacked_window

        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None, None

    # Run active indicators in parallel
    active_indicators = [name for name, active in flags.items() if active]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, active_indicators)

    for name, sig, window_params in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig
            if window_params:
                filtered_window = {k: v for k, v in window_params.items() if v not in (None, 0)}
                if filtered_window:
                    windows_dict[name] = filtered_window

    # ---------------------------------------------------------
    # Majority voting with NaN handling
    # ---------------------------------------------------------
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values())).astype(float)  # allow NaN

        # Only consider rows with no NaN
        valid_mask = ~np.isnan(all_signals).any(axis=1)

        # Initialize final signal with neutral
        final_signal = np.zeros(len(timestamps), dtype=np.int8)

        if valid_mask.any():
            valid_signals = all_signals[valid_mask]
            buy_votes = np.sum(valid_signals == 1, axis=1)
            sell_votes = np.sum(valid_signals == -1, axis=1)
            final_signal[valid_mask] = np.where(
                buy_votes > sell_votes, 1,
                np.where(sell_votes > buy_votes, -1, 0)
            ).astype(np.int8)

    else:
        final_signal = np.zeros(len(timestamps), dtype=np.int8)

    return pd.DataFrame({
        "datetime": timestamps,
        "signals": final_signal
    }), windows_dict

