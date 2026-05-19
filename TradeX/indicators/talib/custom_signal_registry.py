"""
custom_signal_registry.py
=========================
Central registry that describes every supported indicator and candlestick
pattern to the dynamic signal runner.

Design principle
----------------
Adding a new indicator to the system requires **only** an entry here.
No other file needs to be touched.

Registry entry schema
---------------------
Each entry in ``SIGNAL_CONFIG`` is a dict with the following keys:

``inputs`` : list[str]
    Ordered list of OHLCV array names the indicator needs.
    Valid tokens: ``"open"``, ``"high"``, ``"low"``, ``"close"``, ``"volume"``.

``params`` : dict[str, str]
    Mapping of *logical* parameter names (used in the ``windows`` argument
    of :func:`run_custom_signals_with_voting`) to the TA-Lib keyword argument
    names actually passed to ``call_indicator()``.

    Example::

        {"period": "timeperiod"}           # single-period indicator
        {"fastperiod": "fastperiod",
         "slowperiod": "slowperiod",
         "signalperiod": "signalperiod"}   # MACD-style

``default_params`` : dict[str, int | float]
    Fallback values used when the caller does not supply a ``windows`` entry
    for this indicator.  Keys must match the TA-Lib kwarg names (the *values*
    of ``params``).

``signal_type`` : str
    Controls how the raw indicator output is converted to +1 / -1 / 0:

    ``"price_cross"``
        Output crosses the close price → buy above, sell below.
    ``"zero_cross"``
        Output crosses zero → buy when positive, sell when negative.
    ``"threshold"``
        Output has distinct overbought/oversold thresholds.
        Requires ``"ob"`` (overbought) and ``"os"`` (oversold) in the entry.
    ``"mean_cross"``
        Output compared against its own rolling mean.
    ``"line_cross"``
        Two output lines cross each other (e.g. MACD / signal).
        Requires ``"line_indices"`` with the two indices into the raw tuple.
    ``"prev_cross"``
        Compared to its own previous value (e.g. OBV).
    ``"pattern"``
        Candlestick pattern: raw value > 0 → buy, < 0 → sell.
    ``"trend_mode"``
        Binary mode indicator (e.g. HT_TRENDMODE).

``ob`` / ``os`` : float  (only for ``"threshold"``)
    Overbought / oversold levels.

``line_indices`` : tuple[int, int]  (only for ``"line_cross"``)
    Which two positions in the tuple returned by ``call_indicator()`` are the
    fast/slow lines to cross.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Main registry
# ---------------------------------------------------------------------------

SIGNAL_CONFIG: dict[str, dict] = {

    # =========================================================================
    # Overlap / Moving-Average studies  (close → price_cross)
    # =========================================================================
    "SMA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "price_cross",
    },
    "EMA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "price_cross",
    },
    "DEMA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "price_cross",
    },
    "TEMA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "price_cross",
    },
    "TRIMA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "price_cross",
    },
    "WMA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "price_cross",
    },
    "KAMA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 10},
        "signal_type": "price_cross",
    },
    "MA": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "price_cross",
    },
    "T3": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 5},
        "signal_type": "price_cross",
    },
    "HT_TRENDLINE": {
        "inputs": ["close"],
        "params": {},
        "default_params": {},
        "signal_type": "price_cross",
    },
    "MIDPOINT": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "price_cross",
    },
    "LINEARREG": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "price_cross",
    },
    "TSF": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "price_cross",
    },

    # =========================================================================
    # Overlap studies – OHLC inputs
    # =========================================================================
    "MIDPRICE": {
        "inputs": ["high", "low"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "price_cross",         # compared vs (high+low)/2
        "price_expr": "hl2",                  # hint for the signal executor
    },
    "SAR": {
        "inputs": ["high", "low"],
        "params": {},
        "default_params": {"acceleration": 0.02, "maximum": 0.2},
        "signal_type": "price_cross",
        "price_expr": "close",
    },
    "SAREXT": {
        "inputs": ["high", "low"],
        "params": {},
        "default_params": {},
        "signal_type": "price_cross",
        "price_expr": "close",
    },
    "AVGPRICE": {
        "inputs": ["open", "high", "low", "close"],
        "params": {},
        "default_params": {},
        "signal_type": "price_cross",
        "price_expr": "ohlc4",
    },
    "MEDPRICE": {
        "inputs": ["high", "low"],
        "params": {},
        "default_params": {},
        "signal_type": "price_cross",
        "price_expr": "hl2",
    },
    "TYPPRICE": {
        "inputs": ["high", "low", "close"],
        "params": {},
        "default_params": {},
        "signal_type": "price_cross",
        "price_expr": "hlc3",
    },
    "WCLPRICE": {
        "inputs": ["high", "low", "close"],
        "params": {},
        "default_params": {},
        "signal_type": "price_cross",
        "price_expr": "wclprice",
    },

    # =========================================================================
    # MAMA  (adaptive; dual-output line_cross)
    # =========================================================================
    "MAMA": {
        "inputs": ["close"],
        "params": {},
        "default_params": {"fastlimit": 0.5, "slowlimit": 0.05},
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (mama, fama)
    },

    # =========================================================================
    # Bollinger Bands  (special: three outputs, crossover of bands)
    # =========================================================================
    "BBANDS": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 5, "nbdevup": 2, "nbdevdn": 2},
        "signal_type": "bbands",
    },

    # =========================================================================
    # MACD family  (line_cross)
    # =========================================================================
    "MACD": {
        "inputs": ["close"],
        "params": {
            "fastperiod":   "fastperiod",
            "slowperiod":   "slowperiod",
            "signalperiod": "signalperiod",
        },
        "default_params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (macd_line, signal_line)
    },
    "MACDEXT": {
        "inputs": ["close"],
        "params": {
            "fastperiod":   "fastperiod",
            "slowperiod":   "slowperiod",
            "signalperiod": "signalperiod",
        },
        "default_params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "signal_type": "line_cross",
        "line_indices": (0, 1),
    },

    # =========================================================================
    # PPO / APO  (zero_cross on a single output)
    # =========================================================================
    "PPO": {
        "inputs": ["close"],
        "params": {
            "fastperiod": "fastperiod",
            "slowperiod": "slowperiod",
        },
        "default_params": {"fastperiod": 12, "slowperiod": 26},
        "signal_type": "zero_cross",
    },
    "APO": {
        "inputs": ["close"],
        "params": {
            "fastperiod": "fastperiod",
            "slowperiod": "slowperiod",
        },
        "default_params": {"fastperiod": 12, "slowperiod": 26},
        "signal_type": "zero_cross",
    },

    # =========================================================================
    # Stochastic family
    # =========================================================================
    "STOCH": {
        "inputs": ["high", "low", "close"],
        "params": {
            "fastk_period": "fastk_period",
            "slowk_period": "slowk_period",
            "slowd_period": "slowd_period",
        },
        "default_params": {
            "fastk_period": 14,
            "slowk_period": 3,
            "slowd_period": 3,
        },
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (slowk, slowd)
    },
    "STOCHF": {
        "inputs": ["high", "low", "close"],
        "params": {
            "fastk_period": "fastk_period",
            "fastd_period": "fastd_period",
        },
        "default_params": {"fastk_period": 14, "fastd_period": 3},
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (fastk, fastd)
    },
    "STOCHRSI": {
        "inputs": ["close"],
        "params": {
            "timeperiod":   "timeperiod",
            "fastk_period": "fastk_period",
            "fastd_period": "fastd_period",
        },
        "default_params": {
            "timeperiod":   14,
            "fastk_period": 3,
            "fastd_period": 3,
        },
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (fastk, fastd)
    },

    # =========================================================================
    # ADOSC  (zero_cross, fast/slow window)
    # =========================================================================
    "ADOSC": {
        "inputs": ["high", "low", "close", "volume"],
        "params": {
            "fastperiod": "fastperiod",
            "slowperiod": "slowperiod",
        },
        "default_params": {"fastperiod": 3, "slowperiod": 10},
        "signal_type": "zero_cross",
    },

    # =========================================================================
    # Momentum / oscillators
    # =========================================================================
    "RSI": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "threshold",
        "ob": 70.0,
        "os": 30.0,
    },
    "CCI": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "WILLR": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "threshold",
        "ob": -20.0,
        "os": -80.0,
    },
    "MFI": {
        "inputs": ["high", "low", "close", "volume"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "threshold",
        "ob": 80.0,
        "os": 20.0,
    },
    "CMO": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "MOM": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 10},
        "signal_type": "zero_cross",
    },
    "ROC": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 10},
        "signal_type": "zero_cross",
    },
    "ROCP": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 10},
        "signal_type": "zero_cross",
    },
    "ROCR": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 10},
        "signal_type": "threshold",
        "ob": 1.0,
        "os": 1.0,                           # ROCR: >1 buy, <1 sell
    },
    "ROCR100": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 10},
        "signal_type": "threshold",
        "ob": 100.0,
        "os": 100.0,
    },
    "TRIX": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 30},
        "signal_type": "zero_cross",
    },
    "BOP": {
        "inputs": ["open", "high", "low", "close"],
        "params": {},
        "default_params": {},
        "signal_type": "zero_cross",
    },

    # =========================================================================
    # Directional movement
    # =========================================================================
    "ADX": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "mean_cross",
    },
    "ADXR": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "AROON": {
        "inputs": ["high", "low"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (aroon_up, aroon_down)
    },
    "AROONOSC": {
        "inputs": ["high", "low"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "DX": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "mean_cross",
    },
    "MINUS_DI": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "MINUS_DM": {
        "inputs": ["high", "low"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "PLUS_DI": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "PLUS_DM": {
        "inputs": ["high", "low"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },

    # =========================================================================
    # Volume
    # =========================================================================
    "OBV": {
        "inputs": ["close", "volume"],
        "params": {},
        "default_params": {},
        "signal_type": "prev_cross",
    },
    "AD": {
        "inputs": ["high", "low", "close", "volume"],
        "params": {},
        "default_params": {},
        "signal_type": "prev_cross",
    },

    # =========================================================================
    # Volatility
    # =========================================================================
    "ATR": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "mean_cross",
    },
    "NATR": {
        "inputs": ["high", "low", "close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "mean_cross",
    },
    "TRANGE": {
        "inputs": ["high", "low", "close"],
        "params": {},
        "default_params": {},
        "signal_type": "mean_cross",
    },

    # =========================================================================
    # Statistics
    # =========================================================================
    "LINEARREG_ANGLE": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "LINEARREG_INTERCEPT": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "LINEARREG_SLOPE": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 14},
        "signal_type": "zero_cross",
    },
    "STDDEV": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 5},
        "signal_type": "mean_cross",
    },
    "VAR": {
        "inputs": ["close"],
        "params": {"period": "timeperiod"},
        "default_params": {"timeperiod": 5},
        "signal_type": "mean_cross",
    },

    # =========================================================================
    # Hilbert Transform / Cycle
    # =========================================================================
    "HT_DCPERIOD": {
        "inputs": ["close"],
        "params": {},
        "default_params": {},
        "signal_type": "prev_cross",
    },
    "HT_DCPHASE": {
        "inputs": ["close"],
        "params": {},
        "default_params": {},
        "signal_type": "prev_cross",
    },
    "HT_PHASOR": {
        "inputs": ["close"],
        "params": {},
        "default_params": {},
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (inphase, quadrature)
    },
    "HT_SINE": {
        "inputs": ["close"],
        "params": {},
        "default_params": {},
        "signal_type": "line_cross",
        "line_indices": (0, 1),              # (sine, leadsine)
    },
    "HT_TRENDMODE": {
        "inputs": ["close"],
        "params": {},
        "default_params": {},
        "signal_type": "trend_mode",
    },
}


# ---------------------------------------------------------------------------
# Candlestick pattern registry
# All CDL* patterns share the same OHLC inputs and "pattern" signal_type.
# ---------------------------------------------------------------------------

CDL_PATTERNS: frozenset[str] = frozenset({
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE", "CDL3LINESTRIKE",
    "CDL3OUTSIDE", "CDL3STARSINSOUTH", "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY", "CDLADVANCEBLOCK", "CDLBELTHOLD",
    "CDLBREAKAWAY", "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER", "CDLDOJI",
    "CDLDOJISTAR", "CDLDRAGONFLYDOJI", "CDLENGULFING",
    "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI", "CDLHAMMER", "CDLHANGINGMAN",
    "CDLHARAMI", "CDLHARAMICROSS", "CDLHIGHWAVE",
    "CDLHIKKAKE", "CDLHIKKAKEMOD", "CDLHOMINGPIGEON",
    "CDLIDENTICAL3CROWS", "CDLINNECK", "CDLINVERTEDHAMMER",
    "CDLKICKING", "CDLKICKINGBYLENGTH", "CDLLADDERBOTTOM",
    "CDLLONGLEGGEDDOJI", "CDLLONGLINE", "CDLMARUBOZU",
    "CDLMATCHINGLOW", "CDLMATHOLD", "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR", "CDLONNECK", "CDLPIERCING",
    "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS", "CDLSEPARATINGLINES",
    "CDLSHOOTINGSTAR", "CDLSHORTLINE", "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH", "CDLTAKURI",
    "CDLTASUKIGAP", "CDLTHRUSTING", "CDLTRISTAR",
    "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS", "CDLXSIDEGAP3METHODS",
})


def is_candlestick(name: str) -> bool:
    """Return ``True`` when *name* refers to a known CDL pattern."""
    return name.upper() in CDL_PATTERNS


def get_config(name: str) -> dict | None:
    """
    Return the registry entry for *name*, or ``None`` if not found.

    Parameters
    ----------
    name : str
        Indicator name, case-insensitive.
    """
    return SIGNAL_CONFIG.get(name.upper())


def list_indicators() -> list[str]:
    """Return a sorted list of all registered indicator names."""
    return sorted(SIGNAL_CONFIG.keys())


def list_patterns() -> list[str]:
    """Return a sorted list of all registered candlestick pattern names."""
    return sorted(CDL_PATTERNS)