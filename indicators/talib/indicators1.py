from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from TradeX.utils.common.logs import get_logger

logger = get_logger("indicators")

# ============================================================
#  MASTER LIST (ALL INDICATORS)
# ============================================================

ALL_INDICATORS = (
    # -------------------------
    # Overlap Studies
    # -------------------------
    "BBANDS", "DEMA", "EMA", "HT_TRENDLINE", "KAMA",
    "MA", "MAMA", "MIDPOINT", "MIDPRICE", "SAR",
    "SAREXT", "SMA", "T3", "TEMA", "TRIMA", "WMA",
    # -------------------------
    # Momentum Indicators
    # -------------------------
    "ADX", "ADXR", "APO", "AROON", "AROONOSC",
    "BOP", "CCI", "CMO", "DX", "MACD",
    "MACDEXT", "MFI", "MINUS_DI",
    "MINUS_DM", "MOM", "PLUS_DI", "PLUS_DM",
    "PPO", "ROC", "ROCP", "ROCR", "ROCR100",
    "RSI", "STOCH", "STOCHF", "STOCHRSI",
    "TRIX", "WILLR",
    # -------------------------
    # Volume Indicators
    # -------------------------
    "AD", "ADOSC", "OBV",
    # -------------------------
    # Volatility Indicators
    # -------------------------
    "ATR", "NATR", "TRANGE",
    # -------------------------
    # Price Transform Indicators
    # -------------------------
    "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE",
    # -------------------------
    # Cycle Indicators
    # -------------------------
    "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR",
    "HT_SINE", "HT_TRENDMODE",
    # -------------------------
    # Statistic Indicators
    # -------------------------
    "LINEARREG", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "LINEARREG_SLOPE",
    "STDDEV", "TSF", "VAR",

    # ---------------------------
    # CANDLESTICK PATTERN
    # --------------------------
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE",
    "CDL3LINESTRIKE", "CDL3OUTSIDE", "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS", "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER",
    "CDLDOJI", "CDLDOJISTAR", "CDLDRAGONFLYDOJI",
    "CDLENGULFING", "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI", "CDLHAMMER",
    "CDLHANGINGMAN", "CDLHARAMI",
    "CDLHARAMICROSS", "CDLHIGHWAVE",
    "CDLHIKKAKE", "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS",
    "CDLINNECK", "CDLINVERTEDHAMMER",
    "CDLKICKING", "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM", "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE", "CDLMARUBOZU",
    "CDLMATCHINGLOW", "CDLMATHOLD",
    "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR",
    "CDLONNECK", "CDLPIERCING",
    "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR",
    "CDLSHORTLINE", "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH",
    "CDLTAKURI", "CDLTASUKIGAP",
    "CDLTHRUSTING", "CDLTRISTAR",
    "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS"
)

TA_DEFAULT_WINDOWS = {
    # -------------------------
    # Overlap Studies
    # -------------------------
    "BBANDS": 5,
    "DEMA": 30,
    "EMA": 30,
    "KAMA": 10,
    "MA": 30,
    "MAMA": 0,  # adaptive, has fast/slow limits instead
    "MIDPOINT": 14,
    "MIDPRICE": 14,
    "SAR": 0,  # acceleration factor based, no typical window
    "SAREXT": 0,
    "SMA": 30,
    "T3": 5,
    "TEMA": 30,
    "TRIMA": 30,
    "WMA": 30,

    # -------------------------
    # Momentum Indicators
    # -------------------------
    "ADX": 14,
    "ADXR": 14,
    "APO": 12,  # fast 12, slow 26 by default
    "AROON": 14,
    "AROONOSC": 14,
    "CCI": 14,
    "CMO": 14,
    "DX": 14,
    "MACD": (12, 26, 9),
    "MACDFIX": (12, 26, 9),
    "MFI": 14,
    "MINUS_DI": 14,
    "MINUS_DM": 14,
    "MOM": 10,
    "PLUS_DI": 14,
    "PLUS_DM": 14,
    "PPO": (12, 26),
    "ROC": 10,
    "ROCP": 10,
    "ROCR": 10,
    "ROCR100": 10,
    "RSI": 14,
    "STOCH": (14, 3, 3),
    "STOCHF": (14, 3),
    "STOCHRSI": (14, 14, 3, 3),
    "TRIX": 30,
    "ULTOSC": (7, 14, 28),
    "WILLR": 14,

    # -------------------------
    # Volume Indicators
    # -------------------------
    "ADOSC": (3, 10),

    # -------------------------
    # Volatility Indicators
    # -------------------------
    "ATR": 14,
    "NATR": 14,

    # -------------------------
    # Statistic Indicators
    # -------------------------
    "CORREL": 30,
    "LINEARREG": 14,
    "LINEARREG_ANGLE": 14,
    "LINEARREG_INTERCEPT": 14,
    "LINEARREG_SLOPE": 14,
    "STDDEV": 5,
    "TSF": 14,
    "VAR": 5,
}

# ============================================================
# UNIVERSAL INDICATOR CALLER
# ============================================================

def call_indicator(name: str, *args, **kwargs):
    if not hasattr(talib, name):
        raise ValueError(f"Indicator '{name}' not found in TA-Lib")

    func = getattr(talib, name)
    values = func(*args, **kwargs)

    # Determine window
    if "timeperiod" in kwargs:
        window = kwargs["timeperiod"]
    elif any(k in kwargs for k in ["fastperiod", "slowperiod", "signalperiod"]):
        window = tuple(kwargs.get(k) for k in ["fastperiod", "slowperiod", "signalperiod"])
    else:
        window = TA_DEFAULT_WINDOWS.get(name, None)

    return values, window


# ============================================================
# INDICATOR CALL SPECS
# Each entry maps an indicator name to the talib call arguments.
# Multi-output indicators expand into named sub-columns.
# ============================================================

# Indicators that require volume input
_VOLUME_INDICATORS = {"AD", "ADOSC", "OBV", "MFI"}

# Indicators that output multiple arrays — mapped to suffix names
_MULTI_OUTPUT_SUFFIXES: dict[str, list[str]] = {
    "BBANDS":    ["upper", "middle", "lower"],
    "MAMA":      ["mama", "fama"],
    "MACD":      ["macd", "signal", "hist"],
    "MACDEXT":   ["macd", "signal", "hist"],
    "AROON":     ["down", "up"],
    "STOCH":     ["slowk", "slowd"],
    "STOCHF":    ["fastk", "fastd"],
    "STOCHRSI":  ["fastk", "fastd"],
    "HT_PHASOR": ["inphase", "quadrature"],
    "HT_SINE":   ["sine", "leadsine"],
}

# Indicators that need high+low (no close) or high+low+close
_PRICE_INPUTS: dict[str, list[str]] = {
    # needs open, high, low, close (candlestick patterns)
    # handled automatically for CDL* below
}


def _build_col_name(indicator: str, suffix: str | None = None) -> str:
    """Construct a clean column name like 'ta_rsi', 'ta_bbands_upper'."""
    base = f"ta_{indicator.lower()}"
    return f"{base}_{suffix}" if suffix else base


def compute_all_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator in ALL_INDICATORS using call_indicator() and
    return a DataFrame aligned to df_ohlcv's index with one column per
    indicator output.  Multi-output indicators (BBANDS, MACD, etc.) produce
    multiple columns with descriptive suffixes.

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        Must contain columns: open, high, low, close, volume.
        The 'datetime' column is preserved as the index key.

    Returns
    -------
    pd.DataFrame
        Same row count as df_ohlcv.  Columns: 'datetime' + all ta_* columns.
        NaN rows (indicator warm-up period) are forward-filled then zero-filled
        so the result can be merged directly with ml_features without losing rows.
    """
    if df_ohlcv.empty:
        raise ValueError("df_ohlcv is empty — cannot compute indicators.")

    required = ["open", "high", "low", "close", "volume"]
    missing  = [c for c in required if c not in df_ohlcv.columns]
    if missing:
        raise ValueError(f"df_ohlcv is missing required OHLCV columns: {missing}")

    # Extract price series as float64 arrays (TA-Lib requirement)
    op  = df_ohlcv["open"].values.astype(np.float64)
    hi  = df_ohlcv["high"].values.astype(np.float64)
    lo  = df_ohlcv["low"].values.astype(np.float64)
    cl  = df_ohlcv["close"].values.astype(np.float64)
    vol = df_ohlcv["volume"].values.astype(np.float64)

    result_cols: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    for name in ALL_INDICATORS:
        if not hasattr(talib, name):
            logger.warning(f"  TA-Lib has no function '{name}' — skipping.")
            skipped.append(name)
            continue

        try:
            # ── Determine input signature ─────────────────────
            if name.startswith("CDL"):
                # All candlestick patterns: open, high, low, close
                raw, _ = call_indicator(name, op, hi, lo, cl)

            elif name in _VOLUME_INDICATORS:
                if name == "AD":
                    raw, _ = call_indicator(name, hi, lo, cl, vol)
                elif name == "OBV":
                    raw, _ = call_indicator(name, cl, vol)
                elif name == "ADOSC":
                    raw, _ = call_indicator(name, hi, lo, cl, vol,
                                            fastperiod=3, slowperiod=10)
                elif name == "MFI":
                    raw, _ = call_indicator(name, hi, lo, cl, vol,
                                            timeperiod=14)
                else:
                    raw, _ = call_indicator(name, hi, lo, cl, vol)

            elif name in ("MIDPRICE", "SAR", "SAREXT", "ATR", "NATR",
                          "TRANGE", "ADX", "ADXR", "AROON", "AROONOSC",
                          "MINUS_DI", "MINUS_DM", "PLUS_DI", "PLUS_DM",
                          "DX", "CCI", "WILLR", "STOCH", "STOCHF", "MOM"):
                # Needs high + low + close (or high+low for some)
                if name in ("SAR", "SAREXT"):
                    raw, _ = call_indicator(name, hi, lo)
                elif name in ("TRANGE",):
                    raw, _ = call_indicator(name, hi, lo, cl)
                elif name in ("ADX", "ADXR", "AROON", "AROONOSC",
                              "MINUS_DI", "MINUS_DM", "PLUS_DI", "PLUS_DM", "DX"):
                    raw, _ = call_indicator(name, hi, lo, cl, timeperiod=14)
                elif name == "CCI":
                    raw, _ = call_indicator(name, hi, lo, cl, timeperiod=14)
                elif name == "WILLR":
                    raw, _ = call_indicator(name, hi, lo, cl, timeperiod=14)
                elif name == "ATR":
                    raw, _ = call_indicator(name, hi, lo, cl, timeperiod=14)
                elif name == "NATR":
                    raw, _ = call_indicator(name, hi, lo, cl, timeperiod=14)
                elif name == "STOCH":
                    raw, _ = call_indicator(name, hi, lo, cl,
                                            fastk_period=14, slowk_period=3, slowd_period=3)
                elif name == "STOCHF":
                    raw, _ = call_indicator(name, hi, lo, cl,
                                            fastk_period=14, fastd_period=3)
                elif name == "MOM":
                    raw, _ = call_indicator(name, cl, timeperiod=10)
                elif name == "MIDPRICE":
                    raw, _ = call_indicator(name, hi, lo, timeperiod=14)
                else:
                    raw, _ = call_indicator(name, hi, lo, cl)

            elif name == "STOCHRSI":
                raw, _ = call_indicator(name, cl,
                                        timeperiod=14, fastk_period=14,
                                        fastd_period=3, fastd_matype=0)

            elif name == "BBANDS":
                raw, _ = call_indicator(name, cl, timeperiod=5)

            elif name == "MACD":
                raw, _ = call_indicator(name, cl,
                                        fastperiod=12, slowperiod=26, signalperiod=9)

            elif name == "MACDEXT":
                raw, _ = call_indicator(name, cl,
                                        fastperiod=12, slowperiod=26, signalperiod=9)

            elif name == "MAMA":
                raw, _ = call_indicator(name, cl,
                                        fastlimit=0.5, slowlimit=0.05)

            elif name == "AROON":
                raw, _ = call_indicator(name, hi, lo, timeperiod=14)

            elif name == "APO":
                raw, _ = call_indicator(name, cl,
                                        fastperiod=12, slowperiod=26)

            elif name == "PPO":
                raw, _ = call_indicator(name, cl,
                                        fastperiod=12, slowperiod=26)

            elif name in ("HT_PHASOR", "HT_SINE"):
                raw, _ = call_indicator(name, cl)

            elif name in ("BOP",):
                raw, _ = call_indicator(name, op, hi, lo, cl)

            elif name in ("AVGPRICE",):
                raw, _ = call_indicator(name, op, hi, lo, cl)

            elif name in ("MEDPRICE", "TYPPRICE", "WCLPRICE"):
                if name == "MEDPRICE":
                    raw, _ = call_indicator(name, hi, lo)
                elif name == "TYPPRICE":
                    raw, _ = call_indicator(name, hi, lo, cl)
                elif name == "WCLPRICE":
                    raw, _ = call_indicator(name, hi, lo, cl)

            else:
                # Default: close only, use TA_DEFAULT_WINDOWS if available
                window = TA_DEFAULT_WINDOWS.get(name)
                if isinstance(window, int) and window > 0:
                    raw, _ = call_indicator(name, cl, timeperiod=window)
                else:
                    raw, _ = call_indicator(name, cl)

            # ── Unpack multi-output indicators ────────────────
            suffixes = _MULTI_OUTPUT_SUFFIXES.get(name)

            if suffixes:
                # raw is a tuple of arrays
                if not isinstance(raw, (tuple, list)):
                    raw = (raw,)
                for arr, sfx in zip(raw, suffixes):
                    col = _build_col_name(name, sfx)
                    result_cols[col] = np.asarray(arr, dtype=np.float64)
            else:
                col = _build_col_name(name)
                result_cols[col] = np.asarray(raw, dtype=np.float64)

        except Exception as exc:
            logger.warning(f"  Failed to compute '{name}': {exc} — skipping.")
            skipped.append(name)
            continue

    if skipped:
        logger.warning(f"Skipped {len(skipped)} indicator(s): {skipped}")

    # ── Assemble DataFrame ────────────────────────────────────
    df_indicators = pd.DataFrame(result_cols, index=df_ohlcv.index)

    # Forward-fill warm-up NaNs, then zero-fill any remaining
    df_indicators = df_indicators.ffill().fillna(0.0)

    # Reattach datetime for merge
    df_indicators.insert(0, "datetime", df_ohlcv["datetime"].values)

    logger.info(
        f"Indicators computed → {len(df_indicators.columns) - 1} columns "
        f"from {len(ALL_INDICATORS)} indicators "
        f"({len(skipped)} skipped)."
    )

    return df_indicators