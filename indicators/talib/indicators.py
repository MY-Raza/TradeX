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
        # For MACD-style indicators
        window = tuple(kwargs.get(k) for k in ["fastperiod", "slowperiod", "signalperiod"])
    else:
        window = TA_DEFAULT_WINDOWS.get(name, None)

    return values, window
