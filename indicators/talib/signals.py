import numpy as np
from TradeX.indicators.talib.indicators import call_indicator

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

def sma_signal(close, period=14):
    sma = call_indicator("SMA", close, timeperiod=period)
    return np.where(close > sma, 1, np.where(close < sma, -1, 0))

def ema_signal(close, period=14):
    ema = call_indicator("EMA", close, timeperiod=period)
    return np.where(close > ema, 1, np.where(close < ema, -1, 0))

def dema_signal(close, period=14):
    dema = call_indicator("DEMA", close, timeperiod=period)
    return np.where(close > dema, 1, np.where(close < dema, -1, 0))

def tema_signal(close, period=14):
    tema = call_indicator("TEMA", close, timeperiod=period)
    return np.where(close > tema, 1, np.where(close < tema, -1, 0))

def trima_signal(close, period=14):
    trima = call_indicator("TRIMA", close, timeperiod=period)
    return np.where(close > trima, 1, np.where(close < trima, -1, 0))

def wma_signal(close, period=14):
    wma = call_indicator("WMA", close, timeperiod=period)
    return np.where(close > wma, 1, np.where(close < wma, -1, 0))

def kama_signal(close, period=14):
    kama = call_indicator("KAMA", close, timeperiod=period)
    return np.where(close > kama, 1, np.where(close < kama, -1, 0))

def ma_signal(close, period=14, ma_type=0):
    ma = call_indicator("MA", close, timeperiod=period, matype=ma_type)
    return np.where(close > ma, 1, np.where(close < ma, -1, 0))

# =========================================================
# ADVANCED MOVING AVERAGES
# =========================================================

def ht_trendline_signal(close):
    trend = call_indicator("HT_TRENDLINE", close)
    return np.where(close > trend, 1, np.where(close < trend, -1, 0))

def mama_signal(close, fastlimit=0.5, slowlimit=0.05):
    mama, fama = call_indicator("MAMA", close, fastlimit=fastlimit, slowlimit=slowlimit)
    signals = np.zeros_like(close)
    signals[crossover(mama, fama)] = 1
    signals[crossunder(mama, fama)] = -1
    return signals

# =========================================================
# BANDS & MIDPOINTS
# =========================================================

def bbands_signal(close, period=20, nbdev=2):
    upper, mid, lower = call_indicator(
        "BBANDS", close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev
    )
    signals = np.zeros_like(close)
    signals[crossover(lower, close)] = 1
    signals[crossover(close, upper)] = -1
    return signals

def midpoint_signal(close, period=14):
    mid = call_indicator("MIDPOINT", close, timeperiod=period)
    return np.where(close > mid, 1, np.where(close < mid, -1, 0))

def midprice_signal(high, low, period=14):
    mid = call_indicator("MIDPRICE", high, low, timeperiod=period)
    price = (high + low) / 2
    return np.where(price > mid, 1, np.where(price < mid, -1, 0))

# =========================================================
# PARABOLIC SAR
# =========================================================

def sar_signal(high, low, close, acceleration=0.02, maximum=0.2):
    sar = call_indicator("SAR", high, low, acceleration=acceleration, maximum=maximum)
    return np.where(close > sar, 1, np.where(close < sar, -1, 0))

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

def rsi_signal(close, period=14, overbought=70, oversold=30):
    rsi = call_indicator("RSI", close, timeperiod=period)
    return np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))

def cci_signal(high, low, close, period=14):
    cci = call_indicator("CCI", high, low, close, timeperiod=period)
    return np.where(cci > 0, 1, np.where(cci < 0, -1, 0))

def willr_signal(high, low, close, period=14):
    will = call_indicator("WILLR", high, low, close, timeperiod=period)
    return np.where(will < -80, 1, np.where(will > -20, -1, 0))

def adxr_signal(close, period=14):
    adxr = call_indicator("ADXR", close, timeperiod=period)
    return np.where(adxr > 0, 1, np.where(adxr < 0, -1, 0))

def aroon_signal(high, low, period=14):
    aroon_up, aroon_down = call_indicator("AROON", high, low, timeperiod=period)
    return np.where(aroon_up > aroon_down, 1, np.where(aroon_up < aroon_down, -1, 0))

def aroonosc_signal(high, low, period=14):
    aroon_osc = call_indicator("AROONOSC", high, low, timeperiod=period)
    return np.where(aroon_osc > 0, 1, np.where(aroon_osc < 0, -1, 0))

def bop_signal(open_, high, low, close_):
    bop = call_indicator("BOP", open_, high, low, close_)
    return np.where(bop > 0, 1, np.where(bop < 0, -1, 0))

def cmo_signal(close, period=14):
    cmo = call_indicator("CMO", close, timeperiod=period)
    return np.where(cmo > 0, 1, np.where(cmo < 0, -1, 0))

def macdext_signal(close, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, signal, _ = call_indicator("MACDEXT", close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    signals = np.zeros_like(close)
    signals[crossover(macd, signal)] = 1
    signals[crossunder(macd, signal)] = -1
    return signals

def minus_di_signal(high, low, close, period=14):
    mdi = call_indicator("MINUS_DI", high, low, close, timeperiod=period)
    return np.where(mdi > 0, 1, np.where(mdi < 0, -1, 0))

def minus_dm_signal(high, low, period=14):
    mdm = call_indicator("MINUS_DM", high, low, timeperiod=period)
    return np.where(mdm > 0, 1, np.where(mdm < 0, -1, 0))

def mom_signal(close, period=10):
    mom = call_indicator("MOM", close, timeperiod=period)
    return np.where(mom > 0, 1, np.where(mom < 0, -1, 0))

def plus_di_signal(high, low, close, period=14):
    pdi = call_indicator("PLUS_DI", high, low, close, timeperiod=period)
    return np.where(pdi > 0, 1, np.where(pdi < 0, -1, 0))

# =========================================================
# VOLUME INDICATORS
# =========================================================

def mfi_signal(high, low, close, volume, period=14):
    mfi = call_indicator("MFI", high, low, close, volume, timeperiod=period)
    return np.where(mfi < 20, 1, np.where(mfi > 80, -1, 0))

def obv_signal(close, volume):
    obv = call_indicator("OBV", close, volume)
    return np.where(obv > np.roll(obv, 1), 1, -1)

def ad_signal(high, low, close, volume):
    ad = call_indicator("AD", high, low, close, volume)
    return np.where(ad > np.roll(ad, 1), 1, -1)

def adosc_signal(high, low, close, volume, fastperiod=3, slowperiod=10):
    adosc = call_indicator("ADOSC", high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod)
    return np.where(adosc > 0, 1, np.where(adosc < 0, -1, 0))

def natr_signal(high, low, close, period=14):
    natr = call_indicator("NATR", high, low, close, timeperiod=period)
    mean = np.nanmean(natr)
    return np.where(natr > mean, 1, -1)

def wclprice_signal(high, low, close):
    wcl = call_indicator("WCLPRICE", high, low, close)
    price = (high + low + close) / 3
    return np.where(price > wcl, 1, np.where(price < wcl, -1, 0))

# =========================================================
# VOLATILITY
# =========================================================

def atr_signal(high, low, close, period=14):
    atr = call_indicator("ATR", high, low, close, timeperiod=period)
    mean = np.nanmean(atr)
    return np.where(atr > mean, 1, -1)

def trange_signal(high, low, close):
    tr = call_indicator("TRANGE", high, low, close)
    mean = np.nanmean(tr)
    return np.where(tr > mean, 1, -1)

# def beta_signal(close, benchmark, period=14):
#     val = call_indicator("BETA", close, benchmark, timeperiod=period)
#     return np.where(val > 1, 1, np.where(val < 1, -1, 0))

def linearreg_angle_signal(close, period=14):
    val = call_indicator("LINEARREG_ANGLE", close, timeperiod=period)
    return np.where(val > 0, 1, -1)

def stddev_signal(close, period=14):
    val = call_indicator("STDDEV", close, timeperiod=period)
    mean = np.nanmean(val)
    return np.where(val > mean, 1, -1)

def var_signal(close, period=14):
    val = call_indicator("VAR", close, timeperiod=period)
    mean = np.nanmean(val)
    return np.where(val > mean, 1, -1)


# ========================================================
# Hilbert Transform
# ========================================================

def ht_dcperiod_signal(close):
    val = call_indicator("HT_DCPERIOD", close)
    return np.where(val > np.roll(val, 1), 1, -1)

def ht_dcphase_signal(close):
    val = call_indicator("HT_DCPHASE", close)
    return np.where(val > np.roll(val, 1), 1, -1)

def ht_phasor_signal(close):
    inphase, quadrature = call_indicator("HT_PHASOR", close)
    signals = np.zeros_like(close)
    signals[crossover(inphase, quadrature)] = 1
    signals[crossunder(inphase, quadrature)] = -1
    return signals

def linearreg_signal(close, period=14):
    val = call_indicator("LINEARREG", close, timeperiod=period)
    return np.where(close > val, 1, np.where(close < val, -1, 0))

def linearreg_intercept_signal(close, period=14):
    val = call_indicator("LINEARREG_INTERCEPT", close, timeperiod=period)
    return np.where(val > 0, 1, -1)

def linearreg_slope_signal(close, period=14):
    val = call_indicator("LINEARREG_SLOPE", close, timeperiod=period)
    return np.where(val > 0, 1, -1)

def tsf_signal(close, period=14):
    val = call_indicator("TSF", close, timeperiod=period)
    return np.where(close > val, 1, np.where(close < val, -1, 0))

def ht_sine_signal(close):
    sine, leadsine = call_indicator("HT_SINE", close)
    signals = np.zeros_like(close)
    signals[crossover(sine, leadsine)] = 1
    signals[crossunder(sine, leadsine)] = -1
    return signals

def ht_trendmode_signal(close):
    val = call_indicator("HT_TRENDMODE", close)
    return np.where(val == 1, 1, -1)


# =========================================================
# CANDLESTICK PATTERNS
# =========================================================

def candlestick_signal(open, high, low, close, pattern_name):
    val = call_indicator(pattern_name, open, high, low, close)
    signals = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signals,pattern_name

# ============================================
# Math Transform
# ===========================================

def asin_signal(close):
    val = call_indicator("ASIN", close)
    return np.where(val > 0, 1, np.where(val < 0, -1, 0))

def ceil_signal(close):
    val = call_indicator("CEIL", close)
    return val  # CEIL returns numeric, you can threshold if needed

def cosh_signal(close):
    val = call_indicator("COSH", close)
    return val

def log10_signal(close):
    val = call_indicator("LOG10", close)
    return val

def sinh_signal(close):
    val = call_indicator("SINH", close)
    return val

def tan_signal(close):
    val = call_indicator("TAN", close)
    return val

def tanh_signal(close):
    val = call_indicator("TANH", close)
    return val

def max_signal(close1, close2):
    val = call_indicator("MAX", close1, close2)
    return np.where(close1 > close2, 1, -1)

def min_signal(close1, close2):
    val = call_indicator("MIN", close1, close2)
    return np.where(close1 < close2, 1, -1)

def minindex_signal(close, period=14):
    val = call_indicator("MININDEX", close, timeperiod=period)
    return val

def acos_signal(close):
    val = call_indicator("ACOS", close)
    return np.where(val > 0, 1, -1)

def atan_signal(close):
    val = call_indicator("ATAN", close)
    return np.where(val > 0, 1, -1)

def cos_signal(close):
    val = call_indicator("COS", close)
    return np.where(val > 0, 1, -1)

def exp_signal(close):
    val = call_indicator("EXP", close)
    return np.where(val > np.mean(val), 1, -1)

def floor_signal(close):
    val = call_indicator("FLOOR", close)
    return np.where(val > np.mean(val), 1, -1)

def ln_signal(close):
    val = call_indicator("LN", close)
    return np.where(val > np.mean(val), 1, -1)

def sin_signal(close):
    val = call_indicator("SIN", close)
    return np.where(val > 0, 1, -1)

def sqrt_signal(close):
    val = call_indicator("SQRT", close)
    return np.where(val > np.mean(val), 1, -1)

# =====================================================
# Price Transform
# =====================================================
def avgprice_signal(open_, high, low, close):
    avg = call_indicator("AVGPRICE", open_, high, low, close)
    price = (open_ + high + low + close) / 4
    return np.where(price > avg, 1, np.where(price < avg, -1, 0))

def medprice_signal(high, low):
    med = call_indicator("MEDPRICE", high, low)
    price = (high + low) / 2
    return np.where(price > med, 1, np.where(price < med, -1, 0))

def typprice_signal(high, low, close):
    typ = call_indicator("TYPPRICE", high, low, close)
    price = (high + low + close) / 3
    return np.where(price > typ, 1, np.where(price < typ, -1, 0))



def adx_signal(high, low, close, period=14):
    val = call_indicator("ADX", high, low, close, timeperiod=period)
    mean = np.nanmean(val)
    return np.where(val > mean, 1, -1)

def rocp_signal(close, period=10):
    val = call_indicator("ROCP", close, timeperiod=period)
    return np.where(val > 0, 1, np.where(val < 0, -1, 0))

def rocr_signal(close, period=10):
    val = call_indicator("ROCR", close, timeperiod=period)
    return np.where(val > 1, 1, np.where(val < 1, -1, 0))

def rocr100_signal(close, period=10):
    val = call_indicator("ROCR100", close, timeperiod=period)
    return np.where(val > 100, 1, np.where(val < 100, -1, 0))

def stochrsi_signal(close, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0):
    fastk, fastd = call_indicator(
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
    return signals

def t3_signal(close, period=14, vfactor=0.7):
    t3 = call_indicator("T3", close, timeperiod=period, vfactor=vfactor)
    return np.where(close > t3, 1, np.where(close < t3, -1, 0))

def roc_signal(close, period=10):
    val = call_indicator("ROC", close, timeperiod=period)
    return np.where(val > 0, 1, np.where(val < 0, -1, 0))

def stochf_signal(high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0):
    fastk, fastd = call_indicator(
        "STOCHF",
        high,
        low,
        close,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    signals = np.zeros_like(close)
    signals[crossover(fastk, fastd)] = 1
    signals[crossunder(fastk, fastd)] = -1
    return signals

def trix_signal(close, period=14):
    val = call_indicator("TRIX", close, timeperiod=period)
    return np.where(val > 0, 1, np.where(val < 0, -1, 0))

def sarext_signal(high, low, close, startvalue=0, offsetonreverse=0, accelerationinit=0.02, accelerationmax=0.2, accelerationstep=0.02):
    sar = call_indicator(
        "SAREXT",
        high,
        low,
        startvalue=startvalue,
        offsetonreverse=offsetonreverse,
        accelerationinit=accelerationinit,
        accelerationmax=accelerationmax,
        accelerationstep=accelerationstep
    )
    return np.where(close > sar, 1, np.where(close < sar, -1, 0))

def dx_signal(high, low, close, period=14):
    val = call_indicator("DX", high, low, close, timeperiod=period)
    mean = np.nanmean(val)
    return np.where(val > mean, 1, -1)

def stoch_signal(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    slowk, slowd = call_indicator(
        "STOCH",
        high,
        low,
        close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=slowk_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype
    )
    signals = np.zeros_like(close)
    signals[crossover(slowk, slowd)] = 1
    signals[crossunder(slowk, slowd)] = -1
    return signals


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
    "AVGPRICE": avgprice_signal if 'avgprice_signal' in globals() else None,
    "MEDPRICE": medprice_signal if 'medprice_signal' in globals() else None,
    "TYPPRICE": typprice_signal if 'typprice_signal' in globals() else None,

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

