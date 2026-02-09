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

    Steps:
        1. Run all active indicators concurrently
        2. Collect individual indicator signals
        3. Store indicator window parameters (for strategy reproducibility)
        4. Apply majority voting to derive final signal

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

    # Holds indicator configuration (periods, fast/slow params, etc.)
    windows_dict = {}

    # Centralized data dictionary so indicators can auto-pick arguments
    data = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close_,
        "volume": volume
    }

    # ---------------------------------------------------------
    # Indicator execution wrapper
    # ---------------------------------------------------------
    def compute_signal(name):
        """
        Computes a single indicator signal.

        This function is executed in parallel for performance.

        Args:
            name (str): Indicator name.

        Returns:
            tuple:
                - indicator name
                - signal array (int8)
                - unpacked window/parameter dictionary
        """
        try:
            # -------------------------------------------------
            # Candlestick pattern indicators (CDL*)
            # -------------------------------------------------
            if name.startswith("CDL"):
                sig, window = candlestick_signal(
                    open_, high, low, close_, name
                )

                # Candlestick indicators always return a single window
                return name, sig.astype(np.int8), {"window": window}

            # -------------------------------------------------
            # Standard TA-Lib indicators
            # -------------------------------------------------
            func = SIGNAL_FUNCTIONS.get(name)

            # Skip unknown indicators safely
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None, None

            # Auto-extract required arguments from function signature
            args = [
                data[arg]
                for arg in func.__code__.co_varnames
                if arg in data
            ]

            # Execute indicator function
            result = func(*args)

            # -------------------------------------------------
            # Normalize return format
            # -------------------------------------------------
            if isinstance(result, tuple) and len(result) == 2:
                signal_array, window = result
            else:
                signal_array, window = result, None

            # -------------------------------------------------
            # Unpack window parameters into readable form
            # -------------------------------------------------
            unpacked_window = {}

            if window is not None:
                # Multi-parameter indicators
                if isinstance(window, (list, tuple)):

                    # MACD / MACDEXT
                    if "MACD" in name and len(window) == 3:
                        unpacked_window = {
                            "fastperiod": window[0],
                            "slowperiod": window[1],
                            "signalperiod": window[2]
                        }

                    # ADOSC, PPO, STOCHF
                    elif (
                        "ADOSC" in name
                        or "PPO" in name
                        or "STOCHF" in name
                    ) and len(window) >= 2:
                        unpacked_window = {
                            "fastperiod": window[0],
                            "slowperiod": window[1]
                        }

                    # STOCH / STOCHRSI family
                    elif any(x in name for x in ["STOCH", "STOCHRSI"]) and len(window) >= 3:
                        unpacked_window = {
                            "fastk_period": window[0],
                            "slowk_period": window[1],
                            "slowd_period": window[2]
                        }

                    # Fallback for unknown multi-parameter indicators
                    else:
                        unpacked_window = {
                            f"param{i}": w for i, w in enumerate(window)
                        }

                # Single-period indicators (RSI, EMA, SMA, etc.)
                else:
                    unpacked_window = {"period": window}

            return name, signal_array.astype(np.int8), unpacked_window

        except Exception as e:
            # Fail-safe: indicator errors should never crash pipeline
            logger.warning(f"Error calling {name}: {e}")
            return None, None, None

    # ---------------------------------------------------------
    # Parallel execution of active indicators
    # ---------------------------------------------------------
    active_indicators = [
        name for name, active in flags.items() if active
    ]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, active_indicators)

    # ---------------------------------------------------------
    # Collect indicator outputs
    # ---------------------------------------------------------
    for name, sig, window_params in results:
        if name is not None and sig is not None:
            # Store indicator signal
            signals_dict[name] = sig

            # Store only meaningful window parameters
            if window_params:
                filtered_window = {
                    k: v for k, v in window_params.items()
                    if v not in (None, 0)
                }
                if filtered_window:
                    windows_dict[name] = filtered_window

    # ---------------------------------------------------------
    # Majority voting mechanism
    # ---------------------------------------------------------
    if signals_dict:
        # Stack signals: shape → (time, indicators)
        all_signals = np.column_stack(list(signals_dict.values()))

        # Count buy and sell votes per timestep
        buy_votes = np.sum(all_signals == 1, axis=1)
        sell_votes = np.sum(all_signals == -1, axis=1)

        # Resolve final signal via majority rule
        final_signal = np.where(
            buy_votes > sell_votes, 1,
            np.where(sell_votes > buy_votes, -1, 0)
        ).astype(np.int8)

    else:
        # No active indicators → neutral signal
        final_signal = np.zeros(len(timestamps), dtype=np.int8)

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    return (
        pd.DataFrame({
            "datetime": timestamps,
            "signals": final_signal
        }),
        windows_dict
    )
