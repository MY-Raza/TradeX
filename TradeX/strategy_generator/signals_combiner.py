import pandas as pd
import numpy as np

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
    timestamps,
    model_signals=None
):
    """
    Executes all active indicators in parallel and combines their outputs
    using majority voting to produce a final trading signal.

    Signal convention:
        +1 → Buy
        -1 → Sell
         0 → Neutral / No consensus
        NaN → Majority of indicators returned NaN

    NaNs in indicator outputs are handled:
        - Rows with any NaN are dropped before voting
        - Final DataFrame has no NaNs

    Args:
        flags (dict[str, bool]): Indicator activation map.
        open_, high, low, close_, volume (np.ndarray): OHLCV market data.
        timestamps (array-like): Datetime index for alignment.

    Returns:
        tuple:
            - pd.DataFrame: Columns = ["datetime", "signals"], NaN rows dropped
            - dict: Indicator → window/parameter configuration
    """

    signals_dict = {}
    windows_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close_, "volume": volume}

    def compute_signal(name):
        try:
            if name.lower().startswith("cdl"):
                if name.islower():
                    name = name.upper()
                sig, window = candlestick_signal(open_, high, low, close_, name)
                return name, sig.astype(np.float32), {"window": window}

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
    results = [compute_signal(name) for name in active_indicators]

    for name, sig, window_params in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig
            if window_params:
                filtered_window = {k: v for k, v in window_params.items() if v not in (None, 0)}
                if filtered_window:
                    windows_dict[name] = filtered_window
    if model_signals is not None:
            try:
                model_signals["datetime"] = pd.to_datetime(model_signals["datetime"])
                model_signals = model_signals.set_index("datetime")
                aligned_model = model_signals.reindex(
                    pd.to_datetime(timestamps)
                )["signal"].to_numpy(dtype = np.float32)
                signals_dict["ML_MODEL"] = aligned_model
            except Exception as e:
                logger.warning(f"Error integrating ML model signals: {e}")   
            
    # ---------------------------------------------------------
    # Create DataFrame of all signals
    # ---------------------------------------------------------
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values())).astype(np.float32)
        all_signals_df = pd.DataFrame(
            all_signals,
            columns=list(signals_dict.keys()),
            index=pd.to_datetime(timestamps)
        ).reset_index()
        all_signals_df.rename(columns={"index": "datetime"}, inplace=True)

        # Drop rows with any NaN values
        all_signals_df = all_signals_df.dropna(axis=0, how='any').reset_index(drop=True)
        # ---------------------------------------------------------
        # Majority voting on valid rows
        # ---------------------------------------------------------
        final_signal = []
        for i, row in all_signals_df.iterrows():
            votes = row.iloc[1:].to_numpy(dtype=np.float32)
    
            buy_votes = np.sum(np.isclose(votes, 1))
            sell_votes = np.sum(np.isclose(votes, -1))
    
            # Majority voting
            if buy_votes > sell_votes:
                final_signal.append(1)
            elif sell_votes > buy_votes:
                final_signal.append(-1)
            else:
                final_signal.append(0)
        all_signals_df["signals"] = final_signal

        # Keep only datetime and final signals
        final_df = all_signals_df[["datetime", "signals"]]  
        final_df["signals"] = final_df["signals"].astype(np.int8)
    else:
        final_df = pd.DataFrame(columns=["datetime", "signals"])

    return final_df, windows_dict



