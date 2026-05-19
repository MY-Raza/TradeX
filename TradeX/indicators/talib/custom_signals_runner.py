"""
custom_signals_runner.py
========================
Dynamic, registry-driven replacement for ``run_active_signals_with_voting``
in ``signals_combiner.py``.

Key differences from the original
----------------------------------
* Callers pass an **explicit list** of indicator names and candlestick
  patterns instead of a flat ``flags`` dict.
* Indicator execution is driven entirely by the ``SIGNAL_CONFIG`` registry
  in ``custom_signal_registry.py``; no per-indicator ``if``/``elif`` branches
  exist in this file.
* Adding a new indicator requires only a new entry in the registry.
* The input OHLCV data is accepted as a single ``pd.DataFrame`` with
  expected columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
  A ``datetime`` (or ``timestamp``) column is used as the time index.
* All other runtime semantics (NaN removal, 1-bar shift, majority voting)
  are **identical** to the original implementation.

Public API
----------
.. code-block:: python

    from custom_signals_runner import run_custom_signals_with_voting

    final_df, windows_used = run_custom_signals_with_voting(
        indicators=["RSI", "MACD", "ATR", "BBANDS"],
        patterns=["CDLHAMMER", "CDLENGULFING"],
        windows={
            "RSI":  {"period": 21},
            "MACD": {"fastperiod": 8, "slowperiod": 21, "signalperiod": 9},
        },
        df=ohlcv_dataframe,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

# ── Project-local imports ──────────────────────────────────────────────────
# call_indicator lives in the same indicators module your project already uses.
from TradeX.indicators.talib.indicators import call_indicator  # noqa: E402

from TradeX.indicators.talib.custom_signal_registry import (
    SIGNAL_CONFIG,
    CDL_PATTERNS,
    get_config,
    is_candlestick,
)
from TradeX.indicators.talib.custom_signal_utils import (
    apply_signal_type,
    build_indicator_kwargs,
    unpack_window_config,
    majority_vote,
    assemble_signals_df,
    apply_lookahead_shift,
)

# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------


# ===========================================================================
# Core: single-indicator dynamic executor
# ===========================================================================

def _execute_indicator(
    name: str,
    ohlcv: dict[str, np.ndarray],
    windows: dict,
) -> tuple[np.ndarray | None, dict]:
    """
    Dynamically execute one TA-Lib indicator and return a +1/-1/0 signal.

    The function:
    1. Looks up the indicator's registry entry.
    2. Assembles the positional input arrays required by TA-Lib.
    3. Merges caller-supplied ``windows`` overrides with registry defaults.
    4. Calls ``call_indicator()`` with those arrays and keyword arguments.
    5. Converts the raw output into a signal array via ``apply_signal_type()``.
    6. Returns the signal and a cleaned window-config dict.

    Parameters
    ----------
    name : str
        Indicator name (upper-case), e.g. ``"RSI"``, ``"MACD"``.
    ohlcv : dict[str, np.ndarray]
        Dictionary containing keys ``"open"``, ``"high"``, ``"low"``,
        ``"close"``, ``"volume"``.
    windows : dict
        Caller-supplied parameter overrides (passed through unchanged to
        :func:`build_indicator_kwargs`).

    Returns
    -------
    signal_array : np.ndarray or None
        Float32 array of length ``n`` with values in {+1, -1, 0, NaN}, or
        ``None`` if execution failed.
    window_config : dict
        Cleaned parameter dict actually used (e.g. ``{"timeperiod": 14}``).
        Empty dict on failure.
    """
    config = get_config(name)
    if config is None:
        logger.warning("Indicator '%s' not found in registry — skipping.", name)
        return None, {}

    n = len(ohlcv["close"])

    # ── Build positional args ─────────────────────────────────────────────
    try:
        pos_args: list[np.ndarray] = [ohlcv[inp] for inp in config["inputs"]]
    except KeyError as exc:
        logger.warning(
            "Indicator '%s' requires '%s' but it is missing from ohlcv — skipping.",
            name, exc.args[0],
        )
        return None, {}

    # ── Build keyword args ────────────────────────────────────────────────
    kwargs = build_indicator_kwargs(name, config, windows)

    # ── Call TA-Lib ───────────────────────────────────────────────────────
    try:
        raw_values, raw_window = call_indicator(name, *pos_args, **kwargs)
    except Exception as exc:
        logger.warning("call_indicator('%s') raised %s: %s — skipping.",
                       name, type(exc).__name__, exc)
        return None, {}

    # ── Convert to signal ─────────────────────────────────────────────────
    try:
        signal_array = apply_signal_type(raw_values, config, ohlcv, n)
    except Exception as exc:
        logger.warning("apply_signal_type('%s') raised %s: %s — skipping.",
                       name, type(exc).__name__, exc)
        return None, {}

    # ── Unpack window config ──────────────────────────────────────────────
    window_config = unpack_window_config(name, config, kwargs, raw_window)

    return signal_array.astype(np.float32), window_config


def _execute_pattern(
    name: str,
    ohlcv: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, dict]:
    """
    Dynamically execute one TA-Lib candlestick pattern.

    Parameters
    ----------
    name : str
        Pattern name (upper-case), e.g. ``"CDLHAMMER"``.
    ohlcv : dict[str, np.ndarray]
        Must contain ``"open"``, ``"high"``, ``"low"``, ``"close"``.

    Returns
    -------
    signal_array : np.ndarray or None
        Float32 array with values in {+1, -1, 0}, or ``None`` on failure.
    window_config : dict
        Always ``{}`` — candlestick patterns have no user-tunable windows.
    """
    canonical = name.upper()
    if canonical not in CDL_PATTERNS:
        logger.warning("Pattern '%s' not found in CDL_PATTERNS — skipping.", name)
        return None, {}

    try:
        raw_values, _ = call_indicator(
            canonical,
            ohlcv["open"],
            ohlcv["high"],
            ohlcv["low"],
            ohlcv["close"],
        )
    except Exception as exc:
        logger.warning("call_indicator('%s') raised %s: %s — skipping.",
                       canonical, type(exc).__name__, exc)
        return None, {}

    n = len(ohlcv["close"])
    val = np.ravel(raw_values).astype(np.float32)[:n]
    signal = np.where(val > 0, 1.0, np.where(val < 0, -1.0, 0.0)).astype(np.float32)
    return signal, {}


# ===========================================================================
# Main public function
# ===========================================================================

def run_custom_signals_with_voting(
    indicators: list[str],
    patterns: list[str],
    windows: dict,
    df: pd.DataFrame,
    model_signals: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a fully customisable set of indicators and candlestick patterns,
    then combine their outputs with majority voting into a single trading
    signal per bar.

    This function is a dynamic, registry-driven equivalent of
    ``run_active_signals_with_voting`` in ``signals_combiner.py``.

    Signal convention
    -----------------
    +1  → Buy
    -1  → Sell
     0  → Neutral / tie
    NaN → Dropped (row removed before voting)

    NaN handling
    ------------
    Rows containing any NaN across all active signals are dropped before
    voting.  Signal columns are then shifted by 1 bar to eliminate
    look-ahead bias; the first and last rows produced by the shift are
    subsequently removed.

    Parameters
    ----------
    indicators : list[str]
        Indicator names to activate, e.g. ``["RSI", "MACD", "ATR"]``.
        Names are matched case-insensitively against ``SIGNAL_CONFIG``.
        Unknown names are skipped with a warning.
    patterns : list[str]
        Candlestick pattern names to activate, e.g.
        ``["CDLHAMMER", "CDLENGULFING"]``.
        Names are matched case-insensitively against ``CDL_PATTERNS``.
        Unknown names are skipped with a warning.
    windows : dict
        Parameter overrides.  Two formats are accepted:

        *Per-indicator* (recommended)::

            {
                "RSI":  {"period": 21},
                "MACD": {"fastperiod": 8, "slowperiod": 21, "signalperiod": 9},
                "ATR":  {"period": 20},
            }

        *Flat* (applies to all indicators that recognise the key)::

            {"period": 21}

        For multi-window indicators the logical key names are:
        ``"fastperiod"``, ``"slowperiod"``, ``"signalperiod"``,
        ``"fastk_period"``, ``"slowk_period"``, ``"slowd_period"``.
        For single-period indicators use ``"period"`` or ``"timeperiod"``.

        Unrecognised keys are silently ignored.
    df : pd.DataFrame
        OHLCV data frame.  Must contain columns (case-insensitive):
        ``open``, ``high``, ``low``, ``close``, ``volume``.
        Must also contain one of: ``datetime``, ``timestamp``, ``date``,
        or a ``DatetimeIndex``.
    model_signals : pd.DataFrame or None, optional
        Optional ML model signal DataFrame with columns
        ``["datetime", "signal"]``.  The ``datetime`` column must be
        parseable by ``pd.to_datetime()``.  Values are aligned to *df*'s
        timestamps.

    Returns
    -------
    final_df : pd.DataFrame
        Columns: ``["datetime", "signals"]``.
        ``signals`` is int8 (+1 / -1 / 0).
        NaN rows are removed; signals are shifted by 1 bar.
        Returns an empty DataFrame if no valid signals were produced.
    windows_used : dict
        Mapping of indicator/pattern name → parameter dict actually used,
        e.g. ``{"RSI": {"timeperiod": 21}, "MACD": {"fastperiod": 8, ...}}``.
        Indicators with no tunable parameters (e.g. candlestick patterns,
        ``HT_TRENDLINE``) are omitted.

    Examples
    --------
    >>> final_df, used = run_custom_signals_with_voting(
    ...     indicators=["RSI", "EMA", "MACD"],
    ...     patterns=["CDLHAMMER"],
    ...     windows={"RSI": {"period": 21}, "EMA": {"period": 50}},
    ...     df=ohlcv_df,
    ... )
    >>> final_df.head()
    """

    # ── 1. Validate and extract OHLCV arrays ─────────────────────────────
    df = _normalise_df(df)
    timestamps = df["datetime"].values

    ohlcv: dict[str, np.ndarray] = {
        "open":   df["open"].to_numpy(dtype=np.float64),
        "high":   df["high"].to_numpy(dtype=np.float64),
        "low":    df["low"].to_numpy(dtype=np.float64),
        "close":  df["close"].to_numpy(dtype=np.float64),
        "volume": df["volume"].to_numpy(dtype=np.float64),
    }

    signals_dict: dict[str, np.ndarray] = {}
    windows_used: dict[str, dict]       = {}

    # ── 2. Execute indicators ─────────────────────────────────────────────
    for raw_name in indicators:
        name = raw_name.upper()
        logger.debug("Executing indicator: %s", name)

        signal_array, window_config = _execute_indicator(name, ohlcv, windows)

        if signal_array is not None:
            signals_dict[name] = signal_array
            if window_config:
                windows_used[name] = window_config
        else:
            logger.warning("Indicator '%s' returned no signal — skipped.", name)

    # ── 3. Execute candlestick patterns ───────────────────────────────────
    for raw_name in patterns:
        name = raw_name.upper()
        logger.debug("Executing pattern: %s", name)

        signal_array, _ = _execute_pattern(name, ohlcv)

        if signal_array is not None:
            signals_dict[name] = signal_array
        else:
            logger.warning("Pattern '%s' returned no signal — skipped.", name)

    # ── 4. Integrate optional ML model signals ────────────────────────────
    if model_signals is not None:
        signals_dict = _integrate_model_signals(
            signals_dict, model_signals, timestamps
        )

    # ── 5. Assemble, clean, vote ──────────────────────────────────────────
    if not signals_dict:
        logger.warning("No valid signals produced — returning empty DataFrame.")
        return pd.DataFrame(columns=["datetime", "signals"]), windows_used

    all_signals_df = assemble_signals_df(signals_dict, timestamps)

    # Drop any row with a NaN in any signal column
    all_signals_df = (
        all_signals_df.dropna(axis=0, how="any").reset_index(drop=True)
    )

    if len(all_signals_df) < 3:
        logger.warning(
            "Fewer than 3 rows remain after NaN removal — returning empty DataFrame."
        )
        return pd.DataFrame(columns=["datetime", "signals"]), windows_used

    # Shift by 1 bar; remove first and last row
    all_signals_df = apply_lookahead_shift(all_signals_df)

    # ── 6. Majority voting ────────────────────────────────────────────────
    signal_cols  = [c for c in all_signals_df.columns if c != "datetime"]
    signal_matrix = all_signals_df[signal_cols].to_numpy(dtype=np.float32)
    voted         = majority_vote(signal_matrix)

    final_df = all_signals_df[["datetime"]].copy()
    final_df["signals"] = voted.astype(np.int8)

    logger.info(
        "Voting complete: %d bars, %d indicators/patterns active.",
        len(final_df), len(signals_dict),
    )

    return final_df, windows_used


# ===========================================================================
# Internal helpers
# ===========================================================================

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a ``datetime`` column and lower-case OHLCV
    column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV frame (columns may be any case).

    Returns
    -------
    pd.DataFrame
        Normalised copy with columns: ``datetime``, ``open``, ``high``,
        ``low``, ``close``, ``volume``.

    Raises
    ------
    ValueError
        If required OHLCV columns or a datetime source are missing.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Resolve datetime column
    if "datetime" not in df.columns:
        for candidate in ("timestamp", "date", "time"):
            if candidate in df.columns:
                df.rename(columns={candidate: "datetime"}, inplace=True)
                break
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
            else:
                raise ValueError(
                    "DataFrame must contain a 'datetime', 'timestamp', or 'date' "
                    "column, or have a DatetimeIndex."
                )

    df["datetime"] = pd.to_datetime(df["datetime"])

    required_ohlcv = {"open", "high", "low", "close", "volume"}
    missing = required_ohlcv - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required OHLCV columns: {missing}")

    return df


def _integrate_model_signals(
    signals_dict: dict[str, np.ndarray],
    model_signals: pd.DataFrame,
    timestamps: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Align ML model signals to the OHLCV timestamp index and add them to
    *signals_dict* under the key ``"ML_MODEL"``.

    Parameters
    ----------
    signals_dict : dict
        Existing signals dict (mutated and returned).
    model_signals : pd.DataFrame
        Must have columns ``["datetime", "signal"]``.
    timestamps : np.ndarray
        Datetime64 array from the OHLCV DataFrame.

    Returns
    -------
    dict
        Updated *signals_dict*.
    """
    try:
        ms = model_signals.copy()
        ms["datetime"] = pd.to_datetime(ms["datetime"])
        ms = ms.set_index("datetime")
        aligned = ms.reindex(pd.to_datetime(timestamps))["signal"].to_numpy(
            dtype=np.float32
        )
        signals_dict["ML_MODEL"] = aligned
        logger.debug("ML model signals integrated (%d bars).", len(aligned))
    except Exception as exc:
        logger.warning("Failed to integrate ML model signals: %s", exc)
    return signals_dict


# ===========================================================================
# Example usage
# ===========================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # ------------------------------------------------------------------
    # Build a minimal synthetic OHLCV DataFrame for demonstration
    # ------------------------------------------------------------------
    np.random.seed(42)
    n_bars = 300
    dates  = pd.date_range("2023-01-01", periods=n_bars, freq="1h")

    close  = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high   = close + np.abs(np.random.randn(n_bars) * 0.3)
    low    = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_  = close + np.random.randn(n_bars) * 0.1
    volume = np.random.randint(1_000, 10_000, n_bars).astype(float)

    sample_df = pd.DataFrame({
        "datetime": dates,
        "open":     open_,
        "high":     high,
        "low":      low,
        "close":    close,
        "volume":   volume,
    })

    # ------------------------------------------------------------------
    # Example 1 – a handful of indicators with custom windows
    # ------------------------------------------------------------------
    final_df, used = run_custom_signals_with_voting(
        indicators=["RSI", "EMA", "MACD", "ATR", "BBANDS", "STOCH"],
        patterns=["CDLHAMMER", "CDLENGULFING", "CDLDOJI"],
        windows={
            "RSI":   {"period": 21},
            "EMA":   {"period": 50},
            "MACD":  {"fastperiod": 8, "slowperiod": 21, "signalperiod": 9},
            "ATR":   {"period": 20},
            "BBANDS": {"period": 20},
            "STOCH": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
        },
        df=sample_df,
    )

    print("\n── Example 1: indicator mix ──────────────────────────────────")
    print(f"  Output rows : {len(final_df)}")
    print(f"  Signal dist : {final_df['signals'].value_counts().to_dict()}")
    print(f"  Windows used: {used}")
    print(final_df.tail())

    # ------------------------------------------------------------------
    # Example 2 – patterns only (no regular indicators)
    # ------------------------------------------------------------------
    final_df2, used2 = run_custom_signals_with_voting(
        indicators=[],
        patterns=["CDLHAMMER", "CDLMORNINGSTAR", "CDLSHOOTINGSTAR"],
        windows={},
        df=sample_df,
    )

    print("\n── Example 2: candlestick patterns only ──────────────────────")
    print(f"  Output rows : {len(final_df2)}")
    print(f"  Signal dist : {final_df2['signals'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Example 3 – high-volume multi-indicator run (all defaults)
    # ------------------------------------------------------------------
    all_indicators = [
        "SMA", "EMA", "DEMA", "TEMA", "WMA", "KAMA",
        "RSI", "MACD", "ATR", "BBANDS", "CCI", "MFI",
        "WILLR", "ADOSC", "OBV", "STOCH", "STOCHRSI",
        "PPO", "APO", "ADX", "AROON", "TRIX",
    ]
    final_df3, used3 = run_custom_signals_with_voting(
        indicators=all_indicators,
        patterns=[],
        windows={},        # use all registry defaults
        df=sample_df,
    )

    print("\n── Example 3: many indicators, all defaults ──────────────────")
    print(f"  Active indicators : {len(all_indicators)}")
    print(f"  Output rows       : {len(final_df3)}")
    print(f"  Signal dist       : {final_df3['signals'].value_counts().to_dict()}")