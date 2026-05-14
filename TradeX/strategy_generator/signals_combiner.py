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
    flags_array = np.random.choice([True, False], size=len(all_indicators))
    return dict(zip(all_indicators, flags_array))


def run_active_signals_with_voting(
    flags,
    open_,
    high,
    low,
    close_,
    volume,
    timestamps,
    model_signals=None,
    windows_override: dict | None = None,
):
    """
    Executes all active indicators in parallel and combines their outputs
    using majority voting to produce a final trading signal.

    Signal convention:
        +1 → Buy
        -1 → Sell
         0 → Neutral / No consensus
        NaN → Majority of indicators returned NaN

    NaN handling:
        - Rows with any NaN are dropped before voting
        - Signal columns are shifted by 1 to avoid lookahead bias
        - First and last rows are removed after the shift
        - Final DataFrame has no NaNs

    Args:
        flags (dict[str, bool]): Indicator activation map.
        open_, high, low, close_, volume (np.ndarray): OHLCV market data.
        timestamps (array-like): Datetime index for alignment.
        model_signals (pd.DataFrame | None): Optional ML model signal df with
            columns ["datetime", "signal"].
        windows_override (dict | None): Mapping of indicator_name →
            {param_key: value} loaded from the strategy's DB record.
            Supported param keys per indicator type:

                Single-period:   "period" | "timeperiod"
                MACD-style:      "fastperiod", "slowperiod", "signalperiod"
                ADOSC/PPO/STOCHF: "fastperiod", "slowperiod"
                STOCH/STOCHRSI:  "fastk_period" | "fastk",
                                 "slowk_period" | "slowk",
                                 "slowd_period"

            When a key is present and non-zero the corresponding keyword
            argument is forwarded to the indicator function.  Unknown keys
            are silently ignored so that adding new param columns to
            strategy_registry does not break existing code.

    Returns:
        tuple:
            - pd.DataFrame: Columns = ["datetime", "signals"], NaN rows dropped,
              signals shifted by 1, first and last rows removed.
            - dict: Indicator → window/parameter configuration actually used
    """

    # Normalise windows_override so callers can pass flat DB columns too
    # (e.g. {"period": 14, "fastperiod": 5}) and we map them to each indicator.
    windows_override = windows_override or {}

    # Canonical mapping from DB column names → talib keyword argument names
    _DB_TO_TALIB: dict[str, str] = {
        "period":       "timeperiod",
        "timeperiod":   "timeperiod",
        "fastperiod":   "fastperiod",
        "slowperiod":   "slowperiod",
        "signalperiod": "signalperiod",
        "fastk":        "fastk_period",
        "fastk_period": "fastk_period",
        "slowk":        "slowk_period",
        "slowk_period": "slowk_period",
        "slowd_period": "slowd_period",
    }

    def _resolve_kwargs(name: str) -> dict:
        """
        Return a kwargs dict to pass to the indicator function for *name*.
        Looks up windows_override[name] first, then falls back to a flat
        windows_override dict (backwards-compatible with the old per-run
        flat approach used in some callers).
        """
        per_indicator = windows_override.get(name)
        if not per_indicator:
            # Try treating windows_override itself as a flat param map
            per_indicator = {k: v for k, v in windows_override.items()
                             if k in _DB_TO_TALIB and v not in (None, 0)}

        if not per_indicator:
            return {}

        kwargs: dict = {}
        for db_key, val in per_indicator.items():
            if val in (None, 0):
                continue
            talib_key = _DB_TO_TALIB.get(db_key, db_key)
            kwargs[talib_key] = int(val)
        return kwargs

    signals_dict = {}
    windows_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close_, "volume": volume}

    def compute_signal(name):
        try:
            # ── Candlestick patterns ──────────────────────────────────────
            if name.lower().startswith("cdl"):
                canonical = name.upper() if name.islower() else name
                sig, window = candlestick_signal(open_, high, low, close_, canonical)
                return canonical, sig.astype(np.float32), {"window": window}

            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                logger.warning(f"No signal function found for {name}")
                return None, None, None

            # Build positional args that the function expects
            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]

            # Build keyword args from DB windows (only when the function
            # actually accepts them to avoid TypeError on strict signatures)
            extra_kwargs = _resolve_kwargs(name)
            func_params  = set(func.__code__.co_varnames)

            # Filter to params the function can actually accept
            safe_kwargs = {k: v for k, v in extra_kwargs.items() if k in func_params}

            result = func(*args, **safe_kwargs) if safe_kwargs else func(*args)

            if isinstance(result, tuple) and len(result) == 2:
                signal_array, window = result
            else:
                signal_array, window = result, None

            # ── Build windows record ──────────────────────────────────────
            unpacked_window = {}
            # Prefer the kwargs we actually sent (they came from the DB)
            if safe_kwargs:
                unpacked_window = safe_kwargs
            elif window is not None:
                if isinstance(window, (list, tuple)):
                    if "MACD" in name and len(window) == 3:
                        unpacked_window = {
                            "fastperiod":   window[0],
                            "slowperiod":   window[1],
                            "signalperiod": window[2],
                        }
                    elif any(x in name for x in ["ADOSC", "PPO", "STOCHF"]) and len(window) >= 2:
                        unpacked_window = {
                            "fastperiod": window[0],
                            "slowperiod": window[1],
                        }
                    elif any(x in name for x in ["STOCH", "STOCHRSI"]) and len(window) >= 3:
                        unpacked_window = {
                            "fastk_period": window[0],
                            "slowk_period": window[1],
                            "slowd_period": window[2],
                        }
                    else:
                        unpacked_window = {f"param{i}": w for i, w in enumerate(window)}
                else:
                    unpacked_window = {"period": window}

            return name, signal_array.astype(np.float32), unpacked_window

        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None, None

    # ── Run active indicators ──────────────────────────────────────────────
    active_indicators = [name for name, active in flags.items() if active]
    results = [compute_signal(name) for name in active_indicators]

    for name, sig, window_params in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig
            if window_params:
                filtered_window = {k: v for k, v in window_params.items() if v not in (None, 0)}
                if filtered_window:
                    windows_dict[name] = filtered_window

    # ── Optional ML model signals ──────────────────────────────────────────
    if model_signals is not None:
        try:
            model_signals["datetime"] = pd.to_datetime(model_signals["datetime"])
            model_signals = model_signals.set_index("datetime")
            aligned_model = model_signals.reindex(
                pd.to_datetime(timestamps)
            )["signal"].to_numpy(dtype=np.float32)
            signals_dict["ML_MODEL"] = aligned_model
        except Exception as e:
            logger.warning(f"Error integrating ML model signals: {e}")

    # ── Assemble DataFrame & majority-vote ─────────────────────────────────
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values())).astype(np.float32)
        all_signals_df = pd.DataFrame(
            all_signals,
            columns=list(signals_dict.keys()),
            index=pd.to_datetime(timestamps),
        ).reset_index()
        all_signals_df.rename(columns={"index": "datetime"}, inplace=True)

        # Drop rows with any NaN
        all_signals_df = all_signals_df.dropna(axis=0, how="any").reset_index(drop=True)

        # Shift signal columns by 1 to avoid lookahead bias, then drop first and last rows
        signal_cols = [col for col in all_signals_df.columns if col != "datetime"]
        all_signals_df[signal_cols] = all_signals_df[signal_cols].shift(1)
        all_signals_df = all_signals_df.iloc[1:-1].reset_index(drop=True)

        final_signal = []
        for i, row in all_signals_df.iterrows():
            votes = row.iloc[1:].to_numpy(dtype=np.float32)
            buy_votes  = np.sum(np.isclose(votes,  1))
            sell_votes = np.sum(np.isclose(votes, -1))

            if buy_votes > sell_votes:
                final_signal.append(1)
            elif sell_votes > buy_votes:
                final_signal.append(-1)
            else:
                final_signal.append(0)

        all_signals_df["signals"] = final_signal
        final_df = all_signals_df[["datetime", "signals"]].copy()
        final_df["signals"] = final_df["signals"].astype(np.int8)
    else:
        final_df = pd.DataFrame(columns=["datetime", "signals"])

    return final_df, windows_dict