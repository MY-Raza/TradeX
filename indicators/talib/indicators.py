import talib

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
    "MACDEXT", "MACDFIX", "MFI", "MINUS_DI",
    "MINUS_DM", "MOM", "PLUS_DI", "PLUS_DM",
    "PPO", "ROC", "ROCP", "ROCR", "ROCR100",
    "RSI", "STOCH", "STOCHF", "STOCHRSI",
    "TRIX", "ULTOSC", "WILLR",
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
    "BETA", "CORREL", "LINEARREG", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "LINEARREG_SLOPE",
    "STDDEV", "TSF", "VAR",
    # -------------------------
    # Math Transform Indicators
    # -------------------------
    "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH",
    "EXP", "FLOOR", "LN", "LOG10", "SIN", "SINH",
    "SQRT", "TAN", "TANH",
    # -------------------------
    # Math Operators
    # -------------------------
    "ADD", "DIV", "MAX", "MAXINDEX", "MIN",
    "MININDEX", "MULT", "SUB", "SUM",
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

# ============================================================
# UNIVERSAL INDICATOR CALLER
# ============================================================

def call_indicator(name: str, *args, window=None, **kwargs):
    """
    Dynamically call any TA-Lib indicator or candlestick pattern.

    Parameters:
    ----------
    name : str
        TA-Lib function name (e.g., 'RSI', 'MACD', 'CDLENGULFING')
    *args :
        Positional inputs (price arrays)
    window : int, optional
        Time period or window to override the default
    **kwargs :
        Indicator parameters (timeperiod, fastperiod, etc.)

    Returns:
    -------
    numpy.ndarray or tuple
    """
    if window is not None:
        # Override 'timeperiod' or 'period' if the indicator accepts them
        if 'timeperiod' in kwargs:
            kwargs['timeperiod'] = window
        elif 'period' in kwargs:
            kwargs['period'] = window

    if not hasattr(talib, name):
        raise ValueError(f"Indicator '{name}' not found in TA-Lib")
    
    func = getattr(talib, name)
    return func(*args, **kwargs)

