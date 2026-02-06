import numpy as np
from TradeX.indicators.talib.indicators import call_indicator, TA_DEFAULT_WINDOWS

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def crossover(a, b):
    return (a > b) & (np.roll(a, 1) <= np.roll(b, 1))

def crossunder(a, b):
    return (a < b) & (np.roll(a, 1) >= np.roll(b, 1))

# =========================================================
# MOVING AVERAGE SIGNALS
# =========================================================

def sma_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("SMA", 14)
    sma, window = call_indicator("SMA", close, timeperiod=period)
    signal = np.where(close > sma, 1, np.where(close < sma, -1, 0))
    return signal, window


def ema_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("EMA", 14)
    ema, window = call_indicator("EMA", close, timeperiod=period)
    signal = np.where(close > ema, 1, np.where(close < ema, -1, 0))
    return signal, window


def dema_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("DEMA", 14)
    dema, window = call_indicator("DEMA", close, timeperiod=period)
    signal = np.where(close > dema, 1, np.where(close < dema, -1, 0))
    return signal, window


def tema_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("TEMA", 14)
    tema, window = call_indicator("TEMA", close, timeperiod=period)
    signal = np.where(close > tema, 1, np.where(close < tema, -1, 0))
    return signal, window


def trima_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("TRIMA", 14)
    trima, window = call_indicator("TRIMA", close, timeperiod=period)
    signal = np.where(close > trima, 1, np.where(close < trima, -1, 0))
    return signal, window


def wma_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("WMA", 14)
    wma, window = call_indicator("WMA", close, timeperiod=period)
    signal = np.where(close > wma, 1, np.where(close < wma, -1, 0))
    return signal, window


def kama_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("KAMA", 14)
    kama, window = call_indicator("KAMA", close, timeperiod=period)
    signal = np.where(close > kama, 1, np.where(close < kama, -1, 0))
    return signal, window


def ma_signal(close, period=None, ma_type=0):
    period = period or TA_DEFAULT_WINDOWS.get("MA", 14)
    ma, window = call_indicator("MA", close, timeperiod=period, matype=ma_type)
    signal = np.where(close > ma, 1, np.where(close < ma, -1, 0))
    return signal, window


# =========================================================
# ADVANCED MOVING AVERAGES
# =========================================================

def ht_trendline_signal(close):
    trend, window = call_indicator("HT_TRENDLINE", close)
    signal = np.where(close > trend, 1, np.where(close < trend, -1, 0))
    return signal, window


def mama_signal(close, fastlimit=0.5, slowlimit=0.05):
    mama, fama, window = call_indicator("MAMA", close, fastlimit=fastlimit, slowlimit=slowlimit)
    signals = np.zeros_like(close)
    signals[crossover(mama, fama)] = 1
    signals[crossunder(mama, fama)] = -1
    return signals, window


# =========================================================
# BANDS & MIDPOINTS
# =========================================================

def bbands_signal(close, period=None, nbdev=2):
    period = period or TA_DEFAULT_WINDOWS.get("BBANDS", 20)
    upper, mid, lower, window = call_indicator(
        "BBANDS", close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev
    )
    signals = np.zeros_like(close)
    signals[crossover(lower, close)] = 1
    signals[crossunder(upper, close)] = -1
    return signals, window


def midpoint_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("MIDPOINT", 14)
    mid, window = call_indicator("MIDPOINT", close, timeperiod=period)
    signal = np.where(close > mid, 1, np.where(close < mid, -1, 0))
    return signal, window


def midprice_signal(high, low, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("MIDPRICE", 14)
    mid, window = call_indicator("MIDPRICE", high, low, timeperiod=period)
    price = (high + low) / 2
    signal = np.where(price > mid, 1, np.where(price < mid, -1, 0))
    return signal, window


# =========================================================
# PARABOLIC SAR
# =========================================================

def sar_signal(high, low, close, acceleration=0.02, maximum=0.2):
    sar, window = call_indicator("SAR", high, low, acceleration=acceleration, maximum=maximum)
    signal = np.where(close > sar, 1, np.where(close < sar, -1, 0))
    return signal, window

# =========================================================
# MACD / OSCILLATORS
# =========================================================

def macd_signal(close):
    macd, signal, _ = call_indicator("MACD", close)
    signals = np.zeros_like(close)
    signals[crossover(macd, signal)] = 1
    signals[crossunder(macd, signal)] = -1
    return signals

def apo_signal(close):
    apo = call_indicator("APO", close)
    return np.where(apo > 0, 1, np.where(apo < 0, -1, 0))

def ppo_signal(close):
    ppo = call_indicator("PPO", close)
    return np.where(ppo > 0, 1, np.where(ppo < 0, -1, 0))

# =========================================================
# MOMENTUM INDICATORS
# =========================================================

def rsi_signal(close, period=None, overbought=70, oversold=30):
    period = period or TA_DEFAULT_WINDOWS.get("RSI", 14)
    rsi, window = call_indicator("RSI", close, timeperiod=period)
    signal = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
    return signal, window


def cci_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("CCI", 14)
    cci, window = call_indicator("CCI", high, low, close, timeperiod=period)
    signal = np.where(cci > 0, 1, np.where(cci < 0, -1, 0))
    return signal, window


def willr_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("WILLR", 14)
    will, window = call_indicator("WILLR", high, low, close, timeperiod=period)
    signal = np.where(will < -80, 1, np.where(will > -20, -1, 0))
    return signal, window

def adxr_signal(high, low, close, period=14):
    adxr, window = call_indicator("ADXR", high, low, close, timeperiod=period)
    signal = np.where(adxr > 0, 1, np.where(adxr < 0, -1, 0))
    return signal, window

def aroon_signal(high, low, period=14):
    (aroon_up, aroon_down), window = call_indicator("AROON", high, low, timeperiod=period)
    signal = np.where(aroon_up > aroon_down, 1, np.where(aroon_up < aroon_down, -1, 0))
    return signal, window

def aroonosc_signal(high, low, period=14):
    aroon_osc, window = call_indicator("AROONOSC", high, low, timeperiod=period)
    signal = np.where(aroon_osc > 0, 1, np.where(aroon_osc < 0, -1, 0))
    return signal, window

def bop_signal(open_, high, low, close_):
    bop, window = call_indicator("BOP", open_, high, low, close_)
    signal = np.where(bop > 0, 1, np.where(bop < 0, -1, 0))
    return signal, window   

def cmo_signal(close, period=14):
    cmo, window = call_indicator("CMO", close, timeperiod=period)
    signal = np.where(cmo > 0, 1, np.where(cmo < 0, -1, 0))
    return signal, window

def macdext_signal(close, fastperiod=12, slowperiod=26, signalperiod=9):
    (macd, signal_line, _), window = call_indicator(
        "MACDEXT",
        close,
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod
    )

    signals = np.zeros_like(close)
    signals[crossover(macd, signal_line)] = 1
    signals[crossunder(macd, signal_line)] = -1
    return signals, window 

def minus_di_signal(high, low, close, period=14):
    mdi, window = call_indicator("MINUS_DI", high, low, close, timeperiod=period)
    signal = np.where(mdi > 0, 1, np.where(mdi < 0, -1, 0))
    return signal, window

def minus_dm_signal(high, low, period=14):
    mdm, window = call_indicator("MINUS_DM", high, low, timeperiod=period)
    signal = np.where(mdm > 0, 1, np.where(mdm < 0, -1, 0))
    return signal, window

def mom_signal(close, period=10):
    mom, window = call_indicator("MOM", close, timeperiod=period)
    signal = np.where(mom > 0, 1, np.where(mom < 0, -1, 0))
    return signal, window

def plus_di_signal(high, low, close, period=14):
    pdi, window = call_indicator("PLUS_DI", high, low, close, timeperiod=period)
    signal = np.where(pdi > 0, 1, np.where(pdi < 0, -1, 0))
    return signal, window

# =========================================================
# VOLUME INDICATORS
# =========================================================

def mfi_signal(high, low, close, volume, period=14):
    mfi, window = call_indicator("MFI", high, low, close, volume, timeperiod=period)
    signal = np.where(mfi < 20, 1, np.where(mfi > 80, -1, 0))
    return signal, window

def obv_signal(close, volume):
    obv, window = call_indicator("OBV", close, volume)
    signal = np.where(obv > np.roll(obv, 1), 1, -1)
    return signal, window

def ad_signal(high, low, close, volume):
    ad, window = call_indicator("AD", high, low, close, volume)
    signal = np.where(ad > np.roll(ad, 1), 1, -1)
    return signal, window

def adosc_signal(high, low, close, volume, fastperiod=3, slowperiod=10):
    adosc, window = call_indicator(
        "ADOSC", high, low, close, volume,
        fastperiod=fastperiod, slowperiod=slowperiod
    )
    signal = np.where(adosc > 0, 1, np.where(adosc < 0, -1, 0))
    return signal, window

def atr_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("ATR", 14)
    atr, window = call_indicator("ATR", high, low, close, timeperiod=period)
    mean = np.nanmean(atr)
    signal = np.where(atr > mean, 1, -1)
    return signal, window


def natr_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("NATR", 14)
    natr, window = call_indicator("NATR", high, low, close, timeperiod=period)
    mean = np.nanmean(natr)
    signal = np.where(natr > mean, 1, -1)
    return signal, window


def wclprice_signal(high, low, close):
    wcl = call_indicator("WCLPRICE", high, low, close)
    price = (high + low + close) / 3
    return np.where(price > wcl, 1, np.where(price < wcl, -1, 0))

# =========================================================
# VOLATILITY
# =========================================================


def trange_signal(high, low, close):
    tr, window = call_indicator("TRANGE", high, low, close)
    mean = np.nanmean(tr)
    signal = np.where(tr > mean, 1, -1)
    return signal, window

def linearreg_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG", 14)
    val, window = call_indicator("LINEARREG", close, timeperiod=period)
    signal = np.where(close > val, 1, np.where(close < val, -1, 0))
    return signal, window


def linearreg_angle_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG_ANGLE", 14)
    val, window = call_indicator("LINEARREG_ANGLE", close, timeperiod=period)
    signal = np.where(val > 0, 1, -1)
    return signal, window


def linearreg_intercept_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG_INTERCEPT", 14)
    val, window = call_indicator("LINEARREG_INTERCEPT", close, timeperiod=period)
    signal = np.where(val > 0, 1, -1)
    return signal, window


def linearreg_slope_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG_SLOPE", 14)
    val, window = call_indicator("LINEARREG_SLOPE", close, timeperiod=period)
    signal = np.where(val > 0, 1, -1)
    return signal, window


def tsf_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("TSF", 14)
    val, window = call_indicator("TSF", close, timeperiod=period)
    signal = np.where(close > val, 1, np.where(close < val, -1, 0))
    return signal, window


def stddev_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("STDDEV", 5)
    val, window = call_indicator("STDDEV", close, timeperiod=period)
    mean = np.nanmean(val)
    signal = np.where(val > mean, 1, -1)
    return signal, window


def var_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("VAR", 5)
    val, window = call_indicator("VAR", close, timeperiod=period)
    mean = np.nanmean(val)
    signal = np.where(val > mean, 1, -1)
    return signal, window



# ========================================================
# Hilbert Transform
# ========================================================

def ht_dcperiod_signal(close):
    val, window = call_indicator("HT_DCPERIOD", close)
    signal = np.where(val > np.roll(val, 1), 1, -1)
    return signal, window

def ht_dcphase_signal(close):
    val, window = call_indicator("HT_DCPHASE", close)
    signal = np.where(val > np.roll(val, 1), 1, -1)
    return signal, window

def ht_phasor_signal(close):
    (inphase, quadrature), window = call_indicator("HT_PHASOR", close)
    signals = np.zeros_like(close)
    signals[crossover(inphase, quadrature)] = 1
    signals[crossunder(inphase, quadrature)] = -1
    return signals, window

def ht_sine_signal(close):
    (sine, leadsine), window = call_indicator("HT_SINE", close)
    signals = np.zeros_like(close)
    signals[crossover(sine, leadsine)] = 1
    signals[crossunder(sine, leadsine)] = -1
    return signals, window

def ht_trendmode_signal(close):
    val, window = call_indicator("HT_TRENDMODE", close)
    signal = np.where(val == 1, 1, -1)
    return signal, window


# =========================================================
# CANDLESTICK PATTERNS
# =========================================================

def candlestick_signal(open, high, low, close, pattern_name):
    val, window = call_indicator(pattern_name, open, high, low, close)
    signals = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signals, window

# ============================================
# Math Transform
# ===========================================

def asin_signal(close):
    val, window = call_indicator("ASIN", close)
    signal = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signal, window

def ceil_signal(close):
    val, window = call_indicator("CEIL", close)
    return val, window

def cosh_signal(close):
    val, window = call_indicator("COSH", close)
    return val, window

def log10_signal(close):
    val, window = call_indicator("LOG10", close)
    return val, window

def sinh_signal(close):
    val, window = call_indicator("SINH", close)
    return val, window

def tan_signal(close):
    val, window = call_indicator("TAN", close)
    return val, window

def tanh_signal(close):
    val, window = call_indicator("TANH", close)
    return val, window

def max_signal(close1, close2):
    val, window = call_indicator("MAX", close1, close2)
    signal = np.where(close1 > close2, 1, -1)
    return signal, window

def min_signal(close1, close2):
    val, window = call_indicator("MIN", close1, close2)
    signal = np.where(close1 < close2, 1, -1)
    return signal, window

def minindex_signal(close, period=14):
    val, window = call_indicator("MININDEX", close, timeperiod=period)
    return val, window

def acos_signal(close):
    val, window = call_indicator("ACOS", close)
    signal = np.where(val > 0, 1, -1)
    return signal, window

def atan_signal(close):
    val, window = call_indicator("ATAN", close)
    signal = np.where(val > 0, 1, -1)
    return signal, window

def cos_signal(close):
    val, window = call_indicator("COS", close)
    signal = np.where(val > 0, 1, -1)
    return signal, window

def exp_signal(close):
    val, window = call_indicator("EXP", close)
    signal = np.where(val > np.mean(val), 1, -1)
    return signal, window

def floor_signal(close):
    val, window = call_indicator("FLOOR", close)
    signal = np.where(val > np.mean(val), 1, -1)
    return signal, window

def ln_signal(close):
    val, window = call_indicator("LN", close)
    signal = np.where(val > np.mean(val), 1, -1)
    return signal, window

def sin_signal(close):
    val, window = call_indicator("SIN", close)
    signal = np.where(val > 0, 1, -1)
    return signal, window

def sqrt_signal(close):
    val, window = call_indicator("SQRT", close)
    signal = np.where(val > np.mean(val), 1, -1)
    return signal, window

# =====================================================
# Price Transform
# =====================================================
def avgprice_signal(open_, high, low, close):
    avg, window = call_indicator("AVGPRICE", open_, high, low, close)
    price = (open_ + high + low + close) / 4
    signal = np.where(price > avg, 1, np.where(price < avg, -1, 0))
    return signal, window

def medprice_signal(high, low):
    med, window = call_indicator("MEDPRICE", high, low)
    price = (high + low) / 2
    signal = np.where(price > med, 1, np.where(price < med, -1, 0))
    return signal, window

def typprice_signal(high, low, close):
    typ, window = call_indicator("TYPPRICE", high, low, close)
    price = (high + low + close) / 3
    signal = np.where(price > typ, 1, np.where(price < typ, -1, 0))
    return signal, window

def adx_signal(high, low, close, period=14):
    val, window = call_indicator("ADX", high, low, close, timeperiod=period)
    mean = np.nanmean(val)
    signal = np.where(val > mean, 1, -1)
    return signal, window

def rocp_signal(close, period=10):
    val, window = call_indicator("ROCP", close, timeperiod=period)
    signal = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signal, window

def rocr_signal(close, period=10):
    val, window = call_indicator("ROCR", close, timeperiod=period)
    signal = np.where(val > 1, 1, np.where(val < 1, -1, 0))
    return signal, window

def rocr100_signal(close, period=10):
    val, window = call_indicator("ROCR100", close, timeperiod=period)
    signal = np.where(val > 100, 1, np.where(val < 100, -1, 0))
    return signal, window

def stochrsi_signal(close, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0):
    (fastk, fastd), window = call_indicator(
        "STOCHRSI",
        close,
        timeperiod=timeperiod,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    signals = np.zeros_like(close)
    signals[crossover(fastk, fastd)] = 1
    signals[crossunder(fastk, fastd)] = -1
    return signals, window

def t3_signal(close, period=14, vfactor=0.7):
    t3, window = call_indicator("T3", close, timeperiod=period, vfactor=vfactor)
    signal = np.where(close > t3, 1, np.where(close < t3, -1, 0))
    return signal, window

def roc_signal(close, period=10):
    val, window = call_indicator("ROC", close, timeperiod=period)
    signal = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signal, window

def stochf_signal(high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0):
    (fastk, fastd), window = call_indicator(
        "STOCHF",
        high, low, close,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    signals = np.zeros_like(close)
    signals[crossover(fastk, fastd)] = 1
    signals[crossunder(fastk, fastd)] = -1
    return signals, window

def trix_signal(close, period=14):
    val, window = call_indicator("TRIX", close, timeperiod=period)
    signal = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signal, window

def sarext_signal(high, low, close):
    sar, window = call_indicator(
        "SAREXT",
        high, low,
        startValue=0,
        offsetOnReverse=0,
        accelerationInitLong=0.02,
        accelerationInitShort=0.02,
        accelerationMaxLong=0.2,
        accelerationMaxShort=0.2,
        accelerationStepLong=0.02,
        accelerationStepShort=0.02
    )
    signal = np.where(close > sar, 1, np.where(close < sar, -1, 0))
    return signal, window

def dx_signal(high, low, close, period=14):
    val, window = call_indicator("DX", high, low, close, timeperiod=period)
    mean = np.nanmean(val)
    signal = np.where(val > mean, 1, -1)
    return signal, window

def stoch_signal(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    (slowk, slowd), window = call_indicator(
        "STOCH",
        high, low, close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=slowk_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype
    )
    signals = np.zeros_like(close)
    signals[crossover(slowk, slowd)] = 1
    signals[crossunder(slowk, slowd)] = -1
    return signals, window

# =========================================================
# SIGNAL FUNCTION REGISTRY
# =========================================================

SIGNAL_FUNCTIONS = {
    # -------------------------
    # Moving Averages
    # -------------------------
    "SMA": sma_signal,
    "EMA": ema_signal,
    "DEMA": dema_signal,
    "TEMA": tema_signal,
    "TRIMA": trima_signal,
    "WMA": wma_signal,
    "KAMA": kama_signal,
    "MA": ma_signal,
    "HT_TRENDLINE": ht_trendline_signal,
    "MAMA": mama_signal,

    # -------------------------
    # Bands & Midpoints
    # -------------------------
    "BBANDS": bbands_signal,
    "MIDPOINT": midpoint_signal,
    "MIDPRICE": midprice_signal,
    "WCLPRICE": wclprice_signal,
    "AVGPRICE": avgprice_signal,
    "MEDPRICE": medprice_signal,
    "TYPPRICE": typprice_signal,

    # -------------------------
    # Trend / SAR
    # -------------------------
    "SAR": sar_signal,

    # -------------------------
    # Oscillators
    # -------------------------
    "MACD": macd_signal,
    "APO": apo_signal,
    "PPO": ppo_signal,
    "MACDEXT": macdext_signal,
    "ROCP": rocp_signal,
    "ROCR": rocr_signal,
    "ROCR100": rocr100_signal,
    "STOCHRSI": stochrsi_signal,
    "ADX": adx_signal,
    "ADXR": adxr_signal,
    "AROON": aroon_signal,
    "AROONOSC": aroonosc_signal,
    "BOP": bop_signal,
    "CMO": cmo_signal,
    "MINUS_DI": minus_di_signal,
    "MINUS_DM": minus_dm_signal,
    "MOM": mom_signal,
    "PLUS_DI": plus_di_signal,

    # -------------------------
    # Momentum
    # -------------------------
    "RSI": rsi_signal,
    "CCI": cci_signal,
    "WILLR": willr_signal,

    # -------------------------
    # Volume
    # -------------------------
    "MFI": mfi_signal,
    "OBV": obv_signal,
    "AD": ad_signal,
    "ADOSC": adosc_signal,

    # -------------------------
    # Volatility
    # -------------------------
    "ATR": atr_signal,
    "NATR": natr_signal,
    "TRANGE": trange_signal,
    # -------------------------
    # Cycle / Hilbert
    # -------------------------
    "HT_DCPERIOD": ht_dcperiod_signal,
    "HT_DCPHASE": ht_dcphase_signal,
    "HT_PHASOR": ht_phasor_signal,
    "HT_SINE": ht_sine_signal,
    "HT_TRENDMODE": ht_trendmode_signal,

    # -------------------------
    # Statistic Indicators
    # -------------------------
    # "BETA": beta_signal,
    "LINEARREG": linearreg_signal,
    "LINEARREG_ANGLE": linearreg_angle_signal,
    "LINEARREG_INTERCEPT": linearreg_intercept_signal,
    "LINEARREG_SLOPE": linearreg_slope_signal,
    "STDDEV": stddev_signal,
    "TSF": tsf_signal,
    "VAR": var_signal,

    # -------------------------
    # Math Transform
    # -------------------------
    "ACOS": acos_signal,
    "ASIN": asin_signal,
    "ATAN": atan_signal,
    "CEIL": ceil_signal,
    "COS": cos_signal,
    "COSH": cosh_signal,
    "EXP": exp_signal,
    "FLOOR": floor_signal,
    "LN": ln_signal,
    "LOG10": log10_signal,
    "SIN": sin_signal,
    "SINH": sinh_signal,
    "SQRT": sqrt_signal,
    "TAN": tan_signal,
    "TANH": tanh_signal,


    "TRIX": trix_signal,
    "STOCHF": stochf_signal,
    "ROC": roc_signal,
    "T3": t3_signal,

    "SAREXT":sarext_signal,
    "DX":dx_signal,
    "STOCH":stoch_signal
}

