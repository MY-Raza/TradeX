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

# =========================================================
# VOLUME INDICATORS
# =========================================================

def mfi_signal(high, low, close, volume, period=14):
    mfi = call_indicator("MFI", high, low, close, volume, timeperiod=period)
    return np.where(mfi < 20, 1, np.where(mfi > 80, -1, 0))

def obv_signal(close, volume):
    obv = call_indicator("OBV", close, volume)
    return np.where(obv > np.roll(obv, 1), 1, -1)

# =========================================================
# VOLATILITY
# =========================================================

def atr_signal(high, low, close, period=14):
    atr = call_indicator("ATR", high, low, close, timeperiod=period)
    mean = np.nanmean(atr)
    return np.where(atr > mean, 1, -1)

# =========================================================
# CANDLESTICK PATTERNS
# =========================================================

def candlestick_signal(open, high, low, close, pattern_name):
    val = call_indicator(pattern_name, open, high, low, close)
    signals = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signals,pattern_name
