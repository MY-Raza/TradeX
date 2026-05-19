"""
custom_signal_utils.py
======================
Stateless utility functions shared across the custom signal runner.

Responsibilities
----------------
- Convert raw TA-Lib output arrays into +1 / -1 / 0 signal arrays
  using the ``signal_type`` declared in the registry.
- Build and normalise TA-Lib keyword-argument dicts from caller-supplied
  ``windows`` entries.
- Unpack the window configuration actually used into a clean dict for
  the ``windows_used`` return value.
- Apply majority-voting across a 2-D signal matrix.
- Shift signals by 1 bar and strip NaN rows (look-ahead bias prevention).

All functions are typed and free of side-effects; they do not mutate their
arguments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cross helpers (identical semantics to the ones in signals.py)
# ---------------------------------------------------------------------------

def _crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a boolean mask where *a* crosses above *b*."""
    return (a > b) & (np.roll(a, 1) <= np.roll(b, 1))


def _crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a boolean mask where *a* crosses below *b*."""
    return (a < b) & (np.roll(a, 1) >= np.roll(b, 1))


# ---------------------------------------------------------------------------
# Reference-price helpers
# ---------------------------------------------------------------------------

def _reference_price(
    price_expr: str,
    ohlcv: dict[str, np.ndarray],
    n: int,
) -> np.ndarray:
    """
    Compute the reference price array required by ``price_cross`` signal types.

    Parameters
    ----------
    price_expr : str
        One of ``"close"``, ``"hl2"``, ``"hlc3"``, ``"ohlc4"``,
        ``"wclprice"``.
    ohlcv : dict
        Dictionary of OHLCV arrays.
    n : int
        Expected output length (used for broadcasting).
    """
    o = ohlcv.get("open",   np.full(n, np.nan))
    h = ohlcv.get("high",   np.full(n, np.nan))
    lo = ohlcv.get("low",   np.full(n, np.nan))
    c = ohlcv.get("close",  np.full(n, np.nan))

    expr = price_expr.lower()
    if expr == "hl2":
        return (h + lo) / 2
    if expr == "hlc3":
        return (h + lo + c) / 3
    if expr == "ohlc4":
        return (o + h + lo + c) / 4
    if expr == "wclprice":
        return (h + lo + 2 * c) / 4
    # Default / "close"
    return c


# ---------------------------------------------------------------------------
# Signal-type dispatch
# ---------------------------------------------------------------------------

def apply_signal_type(
    raw: object,
    config: dict,
    ohlcv: dict[str, np.ndarray],
    n: int,
) -> np.ndarray:
    """
    Convert raw TA-Lib output into a +1 / -1 / 0 signal array.

    Parameters
    ----------
    raw : object
        The first element returned by ``call_indicator()``.  May be a scalar
        array or a tuple of arrays (for multi-output indicators).
    config : dict
        Registry entry for this indicator (from ``SIGNAL_CONFIG``).
    ohlcv : dict[str, np.ndarray]
        Dictionary containing ``"open"``, ``"high"``, ``"low"``, ``"close"``,
        ``"volume"`` arrays.
    n : int
        Length of the price series.

    Returns
    -------
    np.ndarray
        Float32 array of length *n* containing +1 / -1 / 0 / NaN.
    """
    stype = config["signal_type"]

    if stype == "price_cross":
        return _signal_price_cross(raw, config, ohlcv, n)
    if stype == "zero_cross":
        return _signal_zero_cross(raw, n)
    if stype == "threshold":
        return _signal_threshold(raw, config, n)
    if stype == "mean_cross":
        return _signal_mean_cross(raw, n)
    if stype == "line_cross":
        return _signal_line_cross(raw, config, n)
    if stype == "prev_cross":
        return _signal_prev_cross(raw, n)
    if stype == "bbands":
        return _signal_bbands(raw, ohlcv, n)
    if stype == "pattern":
        return _signal_pattern(raw, n)
    if stype == "trend_mode":
        return _signal_trend_mode(raw, n)

    raise ValueError(f"Unknown signal_type: '{stype}'")


# ── Individual converters ──────────────────────────────────────────────────

def _to_array(val: object, n: int) -> np.ndarray:
    """Safely flatten and trim/pad *val* to length *n*."""
    arr = np.ravel(val).astype(np.float32)
    if len(arr) >= n:
        return arr[:n]
    # pad front with NaN if shorter (should not happen with TA-Lib)
    padded = np.full(n, np.nan, dtype=np.float32)
    padded[n - len(arr):] = arr
    return padded


def _signal_price_cross(
    raw: object,
    config: dict,
    ohlcv: dict[str, np.ndarray],
    n: int,
) -> np.ndarray:
    """Buy when price > indicator, sell when price < indicator."""
    indicator = _to_array(raw, n)
    price_expr = config.get("price_expr", "close")
    price = _reference_price(price_expr, ohlcv, n)

    signal = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(indicator) & ~np.isnan(price)
    signal[valid & (price > indicator)] = 1
    signal[valid & (price < indicator)] = -1
    signal[valid & (price == indicator)] = 0
    return signal


def _signal_zero_cross(raw: object, n: int) -> np.ndarray:
    """Buy when value > 0, sell when value < 0."""
    val = _to_array(raw, n)
    signal = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > 0)] = 1
    signal[valid & (val < 0)] = -1
    signal[valid & (val == 0)] = 0
    return signal


def _signal_threshold(raw: object, config: dict, n: int) -> np.ndarray:
    """Buy below *os*, sell above *ob*.  Neutral in between."""
    val = _to_array(raw, n)
    ob: float = config["ob"]
    os_: float = config["os"]
    signal = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    # Handle indicators where ob == os (e.g. ROCR whose pivot is 1.0)
    if ob == os_:
        signal[valid & (val > ob)] = 1
        signal[valid & (val < os_)] = -1
        signal[valid & (val == ob)] = 0
    else:
        lo, hi = min(ob, os_), max(ob, os_)
        # Whichever extreme is "oversold" → buy signal
        if os_ < ob:                         # normal case: os low, ob high
            signal[valid & (val < os_)] = 1
            signal[valid & (val > ob)] = -1
        else:                                # inverted case: WILLR style
            signal[valid & (val < os_)] = -1
            signal[valid & (val > ob)] = 1
        signal[valid & (val >= lo) & (val <= hi)] = 0
    return signal


def _signal_mean_cross(raw: object, n: int) -> np.ndarray:
    """Buy when value > its mean, sell when value ≤ its mean."""
    val = _to_array(raw, n)
    mean = float(np.nanmean(val))
    signal = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > mean)] = 1
    signal[valid & (val <= mean)] = -1
    return signal


def _signal_line_cross(raw: object, config: dict, n: int) -> np.ndarray:
    """Buy on crossover of line A above line B; sell on crossunder."""
    i0, i1 = config["line_indices"]
    # raw is a tuple of arrays
    line_a = _to_array(raw[i0], n)
    line_b = _to_array(raw[i1], n)

    valid = ~np.isnan(line_a) & ~np.isnan(line_b)
    co = _crossover(line_a, line_b)
    cu = _crossunder(line_a, line_b)

    signal = np.full(n, np.nan, dtype=np.float32)
    signal[valid & co] = 1
    signal[valid & cu] = -1
    signal[valid & ~co & ~cu] = 0
    return signal


def _signal_prev_cross(raw: object, n: int) -> np.ndarray:
    """Buy when value rises vs previous bar, sell when it falls."""
    val = _to_array(raw, n)
    prev = np.roll(val, 1)
    signal = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    valid[0] = False                          # no previous for first bar
    signal[valid & (val > prev)] = 1
    signal[valid & (val < prev)] = -1
    signal[valid & (val == prev)] = 0
    return signal


def _signal_bbands(
    raw: object,
    ohlcv: dict[str, np.ndarray],
    n: int,
) -> np.ndarray:
    """
    Bollinger Bands crossover logic:
    buy on close crossing above lower band, sell on crossing above upper band.
    """
    # raw is a tuple (upper, middle, lower) or ((upper, middle, lower), window)
    # call_indicator returns (values, window); here raw is just values
    upper  = _to_array(raw[0], n)
    lower  = _to_array(raw[2], n)
    close  = _to_array(ohlcv["close"], n)

    valid = ~np.isnan(upper) & ~np.isnan(lower) & ~np.isnan(close)
    co = _crossover(lower, close)
    cu = _crossunder(upper, close)

    signal = np.full(n, np.nan, dtype=np.float32)
    signal[valid & co] = 1
    signal[valid & cu] = -1
    signal[valid & ~co & ~cu] = 0
    return signal


def _signal_pattern(raw: object, n: int) -> np.ndarray:
    """Candlestick pattern: positive → buy, negative → sell, zero → neutral."""
    val = _to_array(raw, n)
    signal = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > 0)] = 1
    signal[valid & (val < 0)] = -1
    signal[valid & (val == 0)] = 0
    return signal


def _signal_trend_mode(raw: object, n: int) -> np.ndarray:
    """HT_TRENDMODE: 1 → trending (buy), 0 → cycling (sell)."""
    val = _to_array(raw, n)
    signal = np.full(n, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val == 1)] = 1
    signal[valid & (val != 1)] = -1
    return signal


# ---------------------------------------------------------------------------
# Parameter / window helpers
# ---------------------------------------------------------------------------

def build_indicator_kwargs(
    name: str,
    config: dict,
    windows: dict,
) -> dict[str, int | float]:
    """
    Build the keyword-argument dict to pass to ``call_indicator()``.

    The caller-supplied ``windows`` dict may contain either:

    * A nested dict keyed by indicator name::

        windows = {"RSI": {"period": 21}, "MACD": {"fastperiod": 8}}

    * Flat per-parameter overrides (legacy / simple use-case)::

        windows = {"period": 21}

    In both cases the values are mapped to their TA-Lib kwarg equivalents
    using ``config["params"]``.

    Parameters
    ----------
    name : str
        Indicator name (upper-case).
    config : dict
        Registry entry for this indicator.
    windows : dict
        Caller-supplied window/parameter overrides.

    Returns
    -------
    dict
        Ready-to-unpack keyword arguments for ``call_indicator()``.
    """
    param_map: dict[str, str] = config.get("params", {})
    defaults:  dict           = config.get("default_params", {})

    # Start from defaults
    kwargs: dict[str, int | float] = dict(defaults)

    # Determine the source of overrides: per-indicator block or flat dict
    per_indicator = windows.get(name)
    if isinstance(per_indicator, dict):
        raw_overrides = per_indicator
    else:
        # Flat dict: pick only keys that this indicator recognises
        all_logical = set(param_map.keys())
        raw_overrides = {k: v for k, v in windows.items()
                         if k in all_logical and isinstance(v, (int, float))}

    for logical_key, val in raw_overrides.items():
        if val in (None, 0):
            continue
        # Map logical → talib key; fall back to using the key as-is
        talib_key = param_map.get(logical_key, logical_key)
        if talib_key in defaults or not param_map:
            kwargs[talib_key] = int(val) if isinstance(val, float) and val == int(val) else val

    return kwargs


def unpack_window_config(
    name: str,
    config: dict,
    kwargs_used: dict,
    raw_window: object,
) -> dict:
    """
    Build the ``windows_used`` entry for one indicator.

    Prefers the ``kwargs_used`` dict (what was actually sent to TA-Lib) over
    the ``raw_window`` returned by ``call_indicator()`` to ensure consistency.

    Parameters
    ----------
    name : str
        Indicator name.
    config : dict
        Registry entry.
    kwargs_used : dict
        Keyword arguments actually passed to ``call_indicator()``.
    raw_window : object
        Second return value of ``call_indicator()``.

    Returns
    -------
    dict
        Cleaned parameter dict, e.g. ``{"timeperiod": 14}`` or
        ``{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}``.
        Empty dict if no meaningful window info is available.
    """
    # Prefer what we sent (most accurate reflection of what ran)
    if kwargs_used:
        return {k: v for k, v in kwargs_used.items() if v not in (None, 0)}

    # Fall back to raw_window from call_indicator
    if raw_window is None:
        return {}
    if isinstance(raw_window, dict):
        return {k: v for k, v in raw_window.items() if v not in (None, 0)}
    if isinstance(raw_window, (int, float)):
        return {"period": raw_window}
    if isinstance(raw_window, (list, tuple)):
        param_keys = list(config.get("params", {}).values())
        if param_keys and len(param_keys) == len(raw_window):
            return {k: v for k, v in zip(param_keys, raw_window)
                    if v not in (None, 0)}
        return {f"param{i}": w for i, w in enumerate(raw_window)
                if w not in (None, 0)}
    return {}


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------

def majority_vote(signals_matrix: np.ndarray) -> np.ndarray:
    """
    Apply majority voting across a 2-D signal matrix.

    Parameters
    ----------
    signals_matrix : np.ndarray, shape (n_bars, n_signals)
        Float32 array where each column is one indicator's signal series.
        Expected values: +1, -1, 0 (NaN rows should already be removed).

    Returns
    -------
    np.ndarray, shape (n_bars,)
        int8 array: +1 (buy majority), -1 (sell majority), 0 (tie/neutral).
    """
    n = signals_matrix.shape[0]
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        row   = signals_matrix[i]
        buys  = int(np.sum(np.isclose(row,  1.0)))
        sells = int(np.sum(np.isclose(row, -1.0)))
        if buys > sells:
            result[i] = 1
        elif sells > buys:
            result[i] = -1
        # else: 0 (tie) is already set
    return result


# ---------------------------------------------------------------------------
# DataFrame assembly helpers
# ---------------------------------------------------------------------------

def assemble_signals_df(
    signals_dict: dict[str, np.ndarray],
    timestamps: object,
) -> pd.DataFrame:
    """
    Stack per-indicator signal arrays into a single DataFrame aligned on
    *timestamps*.

    Parameters
    ----------
    signals_dict : dict[str, np.ndarray]
        Mapping of indicator name → signal array.
    timestamps : array-like
        Datetime values used as the DataFrame index.

    Returns
    -------
    pd.DataFrame
        Columns: ``datetime`` + one column per indicator, indexed 0 … n-1.
    """
    matrix = np.column_stack(list(signals_dict.values())).astype(np.float32)
    df = pd.DataFrame(
        matrix,
        columns=list(signals_dict.keys()),
        index=pd.to_datetime(timestamps),
    ).reset_index()
    df.rename(columns={"index": "datetime"}, inplace=True)
    return df


def apply_lookahead_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift all signal columns by 1 bar to prevent look-ahead bias, then drop
    the first and last rows introduced by the shift.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``datetime`` column.  All other columns are treated as
        signal columns.

    Returns
    -------
    pd.DataFrame
        Shifted DataFrame with the first and last rows removed.
    """
    signal_cols = [c for c in df.columns if c != "datetime"]
    df = df.copy()
    df[signal_cols] = df[signal_cols].shift(1)
    df = df.iloc[1:-1].reset_index(drop=True)
    return df