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

    signal = np.full_like(sma, np.nan, dtype=np.float32)
    valid = ~np.isnan(sma) & ~np.isnan(close)

    signal[valid & (close > sma)] = 1
    signal[valid & (close < sma)] = -1
    signal[valid & (close == sma)] = 0

    return signal, window


def ema_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("EMA", 14)
    ema, window = call_indicator("EMA", close, timeperiod=period)

    signal = np.full_like(ema, np.nan, dtype=np.float32)
    valid = ~np.isnan(ema) & ~np.isnan(close)

    signal[valid & (close > ema)] = 1
    signal[valid & (close < ema)] = -1
    signal[valid & (close == ema)] = 0

    return signal, window


def dema_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("DEMA", 14)
    dema, window = call_indicator("DEMA", close, timeperiod=period)

    signal = np.full_like(dema, np.nan, dtype=np.float32)
    valid = ~np.isnan(dema) & ~np.isnan(close)

    signal[valid & (close > dema)] = 1
    signal[valid & (close < dema)] = -1
    signal[valid & (close == dema)] = 0

    return signal, window

def tema_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("TEMA", 14)
    tema, window = call_indicator("TEMA", close, timeperiod=period)

    signal = np.full_like(tema, np.nan, dtype=np.float32)
    valid = ~np.isnan(tema) & ~np.isnan(close)

    signal[valid & (close > tema)] = 1
    signal[valid & (close < tema)] = -1
    signal[valid & (close == tema)] = 0

    return signal, window


def trima_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("TRIMA", 14)
    trima, window = call_indicator("TRIMA", close, timeperiod=period)

    signal = np.full_like(trima, np.nan, dtype=np.float32)
    valid = ~np.isnan(trima) & ~np.isnan(close)

    signal[valid & (close > trima)] = 1
    signal[valid & (close < trima)] = -1
    signal[valid & (close == trima)] = 0

    return signal, window


def wma_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("WMA", 14)
    wma, window = call_indicator("WMA", close, timeperiod=period)

    signal = np.full_like(wma, np.nan, dtype=np.float32)
    valid = ~np.isnan(wma) & ~np.isnan(close)

    signal[valid & (close > wma)] = 1
    signal[valid & (close < wma)] = -1
    signal[valid & (close == wma)] = 0

    return signal, window



def kama_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("KAMA", 14)
    kama, window = call_indicator("KAMA", close, timeperiod=period)

    signal = np.full_like(kama, np.nan, dtype=np.float32)
    valid = ~np.isnan(kama) & ~np.isnan(close)

    signal[valid & (close > kama)] = 1
    signal[valid & (close < kama)] = -1
    signal[valid & (close == kama)] = 0

    return signal, window


def ma_signal(close, period=None, ma_type=0):
    period = period or TA_DEFAULT_WINDOWS.get("MA", 14)
    ma, window = call_indicator("MA", close, timeperiod=period, matype=ma_type)

    signal = np.full_like(ma, np.nan, dtype=np.float32)
    valid = ~np.isnan(ma) & ~np.isnan(close)

    signal[valid & (close > ma)] = 1
    signal[valid & (close < ma)] = -1
    signal[valid & (close == ma)] = 0

    return signal, window



# =========================================================
# ADVANCED MOVING AVERAGES
# =========================================================

def ht_trendline_signal(close):
    trend, window = call_indicator("HT_TRENDLINE", close)

    signal = np.full_like(trend, np.nan, dtype=np.float32)
    valid = ~np.isnan(trend) & ~np.isnan(close)

    signal[valid & (close > trend)] = 1
    signal[valid & (close < trend)] = -1
    signal[valid & (close == trend)] = 0

    return signal, window



def mama_signal(close, fastlimit=0.5, slowlimit=0.05):
    mama, fama = call_indicator("MAMA", close, fastlimit=fastlimit, slowlimit=slowlimit)
    mama = np.ravel(mama)[:len(close)]
    fama = np.ravel(fama)[:len(close)]

    signal = np.full(len(close), np.nan, dtype=np.float32)

    # Only evaluate where both lines exist
    valid = ~np.isnan(mama) & ~np.isnan(fama)

    cross_up = crossover(mama, fama)[:len(close)]
    cross_down = crossunder(mama, fama)[:len(close)]

    signal[valid & cross_up] = 1
    signal[valid & cross_down] = -1

    # Optional: when valid but no crossover, mark neutral
    neutral_mask = valid & ~(cross_up | cross_down)
    signal[neutral_mask] = 0

    window = None
    return signal, window






# =========================================================
# BANDS & MIDPOINTS
# =========================================================

def bbands_signal(close, period=None, nbdev=2):
    period = period or TA_DEFAULT_WINDOWS.get("BBANDS", 20)

    result = call_indicator("BBANDS", close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)

    if isinstance(result, tuple):
        if len(result) == 4:
            upper, mid, lower, window = result
        elif len(result) == 2:
            mid, window = result
            std = np.std(close)
            upper = mid + nbdev * std
            lower = mid - nbdev * std
        else:
            raise ValueError("Unexpected BBANDS output length")
    else:
        raise ValueError("BBANDS did not return a tuple")

    close = np.ravel(close)
    upper = np.ravel(upper)[:len(close)]
    mid   = np.ravel(mid)[:len(close)]
    lower = np.ravel(lower)[:len(close)]

    signals = np.full(len(close), np.nan, dtype=np.float32)

    valid = ~np.isnan(close) & ~np.isnan(upper) & ~np.isnan(lower)

    co = crossover(lower, close)[:len(close)]
    cu = crossunder(upper, close)[:len(close)]

    signals[valid & co] = 1
    signals[valid & cu] = -1

    neutral_mask = valid & ~(co | cu)
    signals[neutral_mask] = 0

    return signals, window



def midpoint_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("MIDPOINT", 14)
    mid, window = call_indicator("MIDPOINT", close, timeperiod=period)

    close = np.ravel(close)
    mid = np.ravel(mid)[:len(close)]

    signal = np.full_like(mid, np.nan, dtype=np.float32)
    valid = ~np.isnan(mid) & ~np.isnan(close)

    signal[valid & (close > mid)] = 1
    signal[valid & (close < mid)] = -1
    signal[valid & (close == mid)] = 0

    return signal, window


def midprice_signal(high, low, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("MIDPRICE", 14)
    mid, window = call_indicator("MIDPRICE", high, low, timeperiod=period)

    high = np.ravel(high)
    low = np.ravel(low)
    price = (high + low) / 2
    mid = np.ravel(mid)[:len(price)]

    signal = np.full_like(mid, np.nan, dtype=np.float32)
    valid = ~np.isnan(mid) & ~np.isnan(price)

    signal[valid & (price > mid)] = 1
    signal[valid & (price < mid)] = -1
    signal[valid & (price == mid)] = 0

    return signal, window


# =========================================================
# PARABOLIC SAR
# =========================================================

def sar_signal(high, low, close, acceleration=0.02, maximum=0.2):
    sar, window = call_indicator("SAR", high, low, acceleration=acceleration, maximum=maximum)

    close = np.ravel(close)
    sar = np.ravel(sar)[:len(close)]

    signal = np.full_like(sar, np.nan, dtype=np.float32)
    valid = ~np.isnan(close) & ~np.isnan(sar)

    signal[valid & (close > sar)] = 1
    signal[valid & (close < sar)] = -1
    signal[valid & (close == sar)] = 0

    return signal, window

# =========================================================
# MACD / OSCILLATORS
# =========================================================

def macd_signal(close):
    values, window = call_indicator("MACD", close)
    macd, signal_line, hist = values

    close = np.ravel(close)
    macd = np.ravel(macd)[:len(close)]
    signal_line = np.ravel(signal_line)[:len(close)]

    signals = np.full(len(close), np.nan, dtype=np.float32)

    valid = ~np.isnan(macd) & ~np.isnan(signal_line)

    co = crossover(macd, signal_line)[:len(close)]
    cu = crossunder(macd, signal_line)[:len(close)]

    signals[valid & co] = 1
    signals[valid & cu] = -1

    neutral_mask = valid & ~(co | cu)
    signals[neutral_mask] = 0

    return signals, window



def apo_signal(close):
    apo, window = call_indicator("APO", close)

    close = np.ravel(close)
    apo = np.ravel(apo)[:len(close)]

    signal = np.full_like(apo, np.nan, dtype=np.float32)
    valid = ~np.isnan(apo)

    signal[valid & (apo > 0)] = 1
    signal[valid & (apo < 0)] = -1
    signal[valid & (apo == 0)] = 0

    return signal, window


def ppo_signal(close):
    ppo, window = call_indicator("PPO", close)

    signal = np.full_like(ppo, np.nan, dtype=np.float32)  # start with NaN everywhere

    signal[ppo > 0] = 1
    signal[ppo < 0] = -1
    signal[ppo == 0] = 0  # optional, keeps exact zeros neutral

    return signal, window

# =========================================================
# MOMENTUM INDICATORS
# =========================================================

def rsi_signal(close, period=None, overbought=70, oversold=30):
    period = period or TA_DEFAULT_WINDOWS.get("RSI", 14)
    rsi, window = call_indicator("RSI", close, timeperiod=period)

    rsi = np.ravel(rsi)

    signal = np.full_like(rsi, np.nan, dtype=np.float32)
    valid = ~np.isnan(rsi)

    signal[valid & (rsi < oversold)] = 1
    signal[valid & (rsi > overbought)] = -1
    signal[valid & (rsi >= oversold) & (rsi <= overbought)] = 0

    return signal, window


def cci_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("CCI", 14)
    cci, window = call_indicator("CCI", high, low, close, timeperiod=period)

    cci = np.ravel(cci)

    signal = np.full_like(cci, np.nan, dtype=np.float32)
    valid = ~np.isnan(cci)

    signal[valid & (cci > 0)] = 1
    signal[valid & (cci < 0)] = -1
    signal[valid & (cci == 0)] = 0

    return signal, window

def willr_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("WILLR", 14)
    will, window = call_indicator("WILLR", high, low, close, timeperiod=period)

    will = np.ravel(will)

    signal = np.full_like(will, np.nan, dtype=np.float32)
    valid = ~np.isnan(will)

    signal[valid & (will < -80)] = 1
    signal[valid & (will > -20)] = -1
    signal[valid & (will >= -80) & (will <= -20)] = 0

    return signal, window

def adxr_signal(high, low, close, period=14):
    adxr, window = call_indicator("ADXR", high, low, close, timeperiod=period)

    adxr = np.ravel(adxr)

    signal = np.full_like(adxr, np.nan, dtype=np.float32)
    valid = ~np.isnan(adxr)

    signal[valid & (adxr > 0)] = 1
    signal[valid & (adxr < 0)] = -1
    signal[valid & (adxr == 0)] = 0

    return signal, window

def aroon_signal(high, low, period=14):
    (aroon_up, aroon_down), window = call_indicator("AROON", high, low, timeperiod=period)

    aroon_up = np.ravel(aroon_up)
    aroon_down = np.ravel(aroon_down)

    signal = np.full_like(aroon_up, np.nan, dtype=np.float32)
    valid = ~np.isnan(aroon_up) & ~np.isnan(aroon_down)

    signal[valid & (aroon_up > aroon_down)] = 1
    signal[valid & (aroon_up < aroon_down)] = -1
    signal[valid & (aroon_up == aroon_down)] = 0

    return signal, window

def aroonosc_signal(high, low, period=14):
    aroon_osc, window = call_indicator("AROONOSC", high, low, timeperiod=period)

    aroon_osc = np.ravel(aroon_osc)

    signal = np.full_like(aroon_osc, np.nan, dtype=np.float32)
    valid = ~np.isnan(aroon_osc)

    signal[valid & (aroon_osc > 0)] = 1
    signal[valid & (aroon_osc < 0)] = -1
    signal[valid & (aroon_osc == 0)] = 0

    return signal, window

def bop_signal(open, high, low, close):
    bop, window = call_indicator("BOP", open, high, low, close)

    bop = np.ravel(bop)
    signal = np.full_like(bop, np.nan, dtype=np.float32)
    valid = ~np.isnan(bop)

    signal[valid & (bop > 0)] = 1
    signal[valid & (bop < 0)] = -1
    signal[valid & (bop == 0)] = 0

    return signal, window

def cmo_signal(close, period=14):
    cmo, window = call_indicator("CMO", close, timeperiod=period)

    cmo = np.ravel(cmo)
    signal = np.full_like(cmo, np.nan, dtype=np.float32)
    valid = ~np.isnan(cmo)

    signal[valid & (cmo > 0)] = 1
    signal[valid & (cmo < 0)] = -1
    signal[valid & (cmo == 0)] = 0

    return signal, window


def macdext_signal(close, fastperiod=12, slowperiod=26, signalperiod=9):
    (macd, signal_line, _), window = call_indicator(
        "MACDEXT", close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )

    close = np.ravel(close)
    macd = np.ravel(macd)[:len(close)]
    signal_line = np.ravel(signal_line)[:len(close)]

    signal = np.full(len(close), np.nan, dtype=np.float32)
    valid = ~np.isnan(macd) & ~np.isnan(signal_line)

    co = crossover(macd, signal_line)[:len(close)]
    cu = crossunder(macd, signal_line)[:len(close)]

    signal[valid & co] = 1
    signal[valid & cu] = -1
    neutral_mask = valid & ~(co | cu)
    signal[neutral_mask] = 0

    return signal, window

def minus_di_signal(high, low, close, period=14):
    mdi, window = call_indicator("MINUS_DI", high, low, close, timeperiod=period)

    mdi = np.ravel(mdi)
    signal = np.full_like(mdi, np.nan, dtype=np.float32)
    valid = ~np.isnan(mdi)

    signal[valid & (mdi > 0)] = 1
    signal[valid & (mdi < 0)] = -1
    signal[valid & (mdi == 0)] = 0

    return signal, window

def minus_dm_signal(high, low, period=14):
    mdm, window = call_indicator("MINUS_DM", high, low, timeperiod=period)

    mdm = np.ravel(mdm)
    signal = np.full_like(mdm, np.nan, dtype=np.float32)
    valid = ~np.isnan(mdm)

    signal[valid & (mdm > 0)] = 1
    signal[valid & (mdm < 0)] = -1
    signal[valid & (mdm == 0)] = 0

    return signal, window

def mom_signal(close, period=10):
    mom, window = call_indicator("MOM", close, timeperiod=period)

    mom = np.ravel(mom)
    signal = np.full_like(mom, np.nan, dtype=np.float32)
    valid = ~np.isnan(mom)

    signal[valid & (mom > 0)] = 1
    signal[valid & (mom < 0)] = -1
    signal[valid & (mom == 0)] = 0

    return signal, window

def plus_di_signal(high, low, close, period=14):
    pdi, window = call_indicator("PLUS_DI", high, low, close, timeperiod=period)

    pdi = np.ravel(pdi)
    signal = np.full_like(pdi, np.nan, dtype=np.float32)
    valid = ~np.isnan(pdi)

    signal[valid & (pdi > 0)] = 1
    signal[valid & (pdi < 0)] = -1
    signal[valid & (pdi == 0)] = 0

    return signal, window


def plus_dm_signal(high, low, period=14):
    pdm, window = call_indicator("PLUS_DM", high, low, timeperiod=period)

    pdm = np.ravel(pdm)
    signal = np.full_like(pdm, np.nan, dtype=np.float32)
    valid = ~np.isnan(pdm)

    signal[valid & (pdm > 0)] = 1
    signal[valid & (pdm < 0)] = -1
    signal[valid & (pdm == 0)] = 0

    return signal, window



# =========================================================
# VOLUME INDICATORS
# =========================================================

def mfi_signal(high, low, close, volume, period=14):
    mfi, window = call_indicator("MFI", high, low, close, volume, timeperiod=period)

    mfi = np.ravel(mfi)
    signal = np.full_like(mfi, np.nan, dtype=np.float32)
    valid = ~np.isnan(mfi)

    signal[valid & (mfi < 20)] = 1
    signal[valid & (mfi > 80)] = -1
    signal[valid & (mfi >= 20) & (mfi <= 80)] = 0

    return signal, window


def obv_signal(close, volume):
    obv, window = call_indicator("OBV", close, volume)

    obv = np.ravel(obv)
    signal = np.full_like(obv, np.nan, dtype=np.float32)
    valid = ~np.isnan(obv)

    prev = np.roll(obv, 1)
    # avoid first element comparison (NaN)
    valid[0] = False

    signal[valid & (obv > prev)] = 1
    signal[valid & (obv < prev)] = -1
    signal[valid & (obv == prev)] = 0

    return signal, window


def ad_signal(high, low, close, volume):
    ad, window = call_indicator("AD", high, low, close, volume)

    ad = np.ravel(ad)
    signal = np.full_like(ad, np.nan, dtype=np.float32)
    valid = ~np.isnan(ad)

    prev = np.roll(ad, 1)
    valid[0] = False

    signal[valid & (ad > prev)] = 1
    signal[valid & (ad < prev)] = -1
    signal[valid & (ad == prev)] = 0

    return signal, window


def adosc_signal(high, low, close, volume, fastperiod=3, slowperiod=10):
    adosc, window = call_indicator(
        "ADOSC", high, low, close, volume,
        fastperiod=fastperiod, slowperiod=slowperiod
    )

    adosc = np.ravel(adosc)
    signal = np.full_like(adosc, np.nan, dtype=np.float32)
    valid = ~np.isnan(adosc)

    signal[valid & (adosc > 0)] = 1
    signal[valid & (adosc < 0)] = -1
    signal[valid & (adosc == 0)] = 0

    # Clean window to only meaningful values
    if isinstance(window, (list, tuple)):
        window = [w for w in window if isinstance(w, (int, float)) and w > 0][:2]
    else:
        window = [fastperiod, slowperiod]

    return signal, window



def atr_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("ATR", 14)
    atr, window = call_indicator("ATR", high, low, close, timeperiod=period)

    atr = np.ravel(atr)
    mean = np.nanmean(atr)
    signal = np.full_like(atr, np.nan, dtype=np.float32)
    valid = ~np.isnan(atr)

    signal[valid & (atr > mean)] = 1
    signal[valid & (atr <= mean)] = -1

    return signal, window


def natr_signal(high, low, close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("NATR", 14)
    natr, window = call_indicator("NATR", high, low, close, timeperiod=period)

    natr = np.ravel(natr)
    mean = np.nanmean(natr)
    signal = np.full_like(natr, np.nan, dtype=np.float32)
    valid = ~np.isnan(natr)

    signal[valid & (natr > mean)] = 1
    signal[valid & (natr <= mean)] = -1

    return signal, window


def wclprice_signal(high, low, close):
    wcl, window = call_indicator("WCLPRICE", high, low, close)

    price = (high + low + close) / 3
    price = np.ravel(price)
    wcl = np.ravel(wcl)[:len(price)]

    signal = np.full_like(price, np.nan, dtype=np.float32)
    valid = ~np.isnan(price) & ~np.isnan(wcl)

    signal[valid & (price > wcl)] = 1
    signal[valid & (price < wcl)] = -1
    signal[valid & (price == wcl)] = 0

    return signal, window



# =========================================================
# VOLATILITY
# =========================================================

def trange_signal(high, low, close):
    tr, window = call_indicator("TRANGE", high, low, close)
    tr = np.ravel(tr)
    mean = np.nanmean(tr)

    signal = np.full_like(tr, np.nan, dtype=np.float32)
    valid = ~np.isnan(tr)

    signal[valid & (tr > mean)] = 1
    signal[valid & (tr <= mean)] = -1

    return signal, window


def linearreg_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG", 14)
    val, window = call_indicator("LINEARREG", close, timeperiod=period)
    
    close = np.ravel(close)
    val = np.ravel(val)[:len(close)]

    signal = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(val) & ~np.isnan(close)

    signal[valid & (close > val)] = 1
    signal[valid & (close < val)] = -1
    signal[valid & (close == val)] = 0

    return signal, window

def linearreg_angle_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG_ANGLE", 14)
    val, window = call_indicator("LINEARREG_ANGLE", close, timeperiod=period)
    
    val = np.ravel(val)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)

    signal[valid & (val > 0)] = 1
    signal[valid & (val <= 0)] = -1

    return signal, window


def linearreg_intercept_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG_INTERCEPT", 14)
    val, window = call_indicator("LINEARREG_INTERCEPT", close, timeperiod=period)
    
    val = np.ravel(val)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)

    signal[valid & (val > 0)] = 1
    signal[valid & (val <= 0)] = -1

    return signal, window

def linearreg_slope_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("LINEARREG_SLOPE", 14)
    val, window = call_indicator("LINEARREG_SLOPE", close, timeperiod=period)
    
    val = np.ravel(val)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)

    signal[valid & (val > 0)] = 1
    signal[valid & (val <= 0)] = -1

    return signal, window

def tsf_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("TSF", 14)
    val, window = call_indicator("TSF", close, timeperiod=period)

    close = np.ravel(close)
    val = np.ravel(val)[:len(close)]

    signal = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(val) & ~np.isnan(close)

    signal[valid & (close > val)] = 1
    signal[valid & (close < val)] = -1
    signal[valid & (close == val)] = 0

    return signal, window

def stddev_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("STDDEV", 5)
    val, window = call_indicator("STDDEV", close, timeperiod=period)

    val = np.ravel(val)
    mean = np.nanmean(val)

    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)

    signal[valid & (val > mean)] = 1
    signal[valid & (val <= mean)] = -1

    return signal, window

def var_signal(close, period=None):
    period = period or TA_DEFAULT_WINDOWS.get("VAR", 5)
    val, window = call_indicator("VAR", close, timeperiod=period)

    val = np.ravel(val)
    mean = np.nanmean(val)

    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)

    signal[valid & (val > mean)] = 1
    signal[valid & (val <= mean)] = -1

    return signal, window

# ========================================================
# Hilbert Transform
# ========================================================
def ht_dcperiod_signal(close):
    val, window = call_indicator("HT_DCPERIOD", close)
    val = np.ravel(val)
    
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    
    prev = np.roll(val, 1)
    valid[0] = False  # first element has no previous to compare

    signal[valid & (val > prev)] = 1
    signal[valid & (val <= prev)] = -1

    return signal, window

def ht_dcphase_signal(close):
    val, window = call_indicator("HT_DCPHASE", close)
    val = np.ravel(val)
    
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    
    prev = np.roll(val, 1)
    valid[0] = False

    signal[valid & (val > prev)] = 1
    signal[valid & (val <= prev)] = -1

    return signal, window

def ht_phasor_signal(close):
    (inphase, quadrature), window = call_indicator("HT_PHASOR", close)
    
    close = np.ravel(close)
    inphase = np.ravel(inphase)[:len(close)]
    quadrature = np.ravel(quadrature)[:len(close)]

    signal = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(inphase) & ~np.isnan(quadrature)

    temp_signal = np.zeros_like(close)
    temp_signal[crossover(inphase, quadrature)] = 1
    temp_signal[crossunder(inphase, quadrature)] = -1

    signal[valid] = temp_signal[valid]

    return signal, window

def ht_sine_signal(close):
    (sine, leadsine), window = call_indicator("HT_SINE", close)
    
    close = np.ravel(close)
    sine = np.ravel(sine)[:len(close)]
    leadsine = np.ravel(leadsine)[:len(close)]

    signal = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(sine) & ~np.isnan(leadsine)

    temp_signal = np.zeros_like(close)
    temp_signal[crossover(sine, leadsine)] = 1
    temp_signal[crossunder(sine, leadsine)] = -1

    signal[valid] = temp_signal[valid]

    return signal, window

def ht_trendmode_signal(close):
    val, window = call_indicator("HT_TRENDMODE", close)
    val = np.ravel(val)

    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)

    signal[valid & (val == 1)] = 1
    signal[valid & (val != 1)] = -1

    return signal, window

# =========================================================
# CANDLESTICK PATTERNS
# =========================================================

def candlestick_signal(open, high, low, close, pattern_name):
    val, window = call_indicator(pattern_name, open, high, low, close)
    signals = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signals, window

# =====================================================
# Price Transform
# =====================================================
def avgprice_signal(open, high, low, close):
    avg, window = call_indicator("AVGPRICE", open, high, low, close)
    price = (open + high + low + close) / 4
    signal = np.full_like(price, np.nan, dtype=np.float32)
    valid = ~np.isnan(avg) & ~np.isnan(price)
    signal[valid & (price > avg)] = 1
    signal[valid & (price < avg)] = -1
    signal[valid & (price == avg)] = 0
    return signal, window


def medprice_signal(high, low):
    med, window = call_indicator("MEDPRICE", high, low)
    price = (high + low) / 2
    signal = np.full_like(price, np.nan, dtype=np.float32)
    valid = ~np.isnan(med) & ~np.isnan(price)
    signal[valid & (price > med)] = 1
    signal[valid & (price < med)] = -1
    signal[valid & (price == med)] = 0
    return signal, window


def typprice_signal(high, low, close):
    typ, window = call_indicator("TYPPRICE", high, low, close)
    price = (high + low + close) / 3
    signal = np.full_like(price, np.nan, dtype=np.float32)
    valid = ~np.isnan(typ) & ~np.isnan(price)
    signal[valid & (price > typ)] = 1
    signal[valid & (price < typ)] = -1
    signal[valid & (price == typ)] = 0
    return signal, window


def adx_signal(high, low, close, period=14):
    val, window = call_indicator("ADX", high, low, close, timeperiod=period)
    mean = np.nanmean(val)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > mean)] = 1
    signal[valid & (val < mean)] = -1
    signal[valid & (val == mean)] = 0
    return signal, window


def rocp_signal(close, period=10):
    val, window = call_indicator("ROCP", close, timeperiod=period)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > 0)] = 1
    signal[valid & (val < 0)] = -1
    signal[valid & (val == 0)] = 0
    return signal, window


def rocr_signal(close, period=10):
    val, window = call_indicator("ROCR", close, timeperiod=period)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > 1)] = 1
    signal[valid & (val < 1)] = -1
    signal[valid & (val == 1)] = 0
    return signal, window


def rocr100_signal(close, period=10):
    val, window = call_indicator("ROCR100", close, timeperiod=period)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > 100)] = 1
    signal[valid & (val < 100)] = -1
    signal[valid & (val == 100)] = 0
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
    signals = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(fastk) & ~np.isnan(fastd)
    signals[valid & crossover(fastk, fastd)] = 1
    signals[valid & crossunder(fastk, fastd)] = -1
    signals[valid & ~(crossover(fastk, fastd) | crossunder(fastk, fastd))] = 0
    return signals, window


def t3_signal(close, period=14, vfactor=0.7):
    t3, window = call_indicator("T3", close, timeperiod=period, vfactor=vfactor)
    signal = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(t3) & ~np.isnan(close)
    signal[valid & (close > t3)] = 1
    signal[valid & (close < t3)] = -1
    signal[valid & (close == t3)] = 0
    return signal, window


def roc_signal(close, period=10):
    val, window = call_indicator("ROC", close, timeperiod=period)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > 0)] = 1
    signal[valid & (val < 0)] = -1
    signal[valid & (val == 0)] = 0
    return signal, window


def stochf_signal(high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0):
    (fastk, fastd), window = call_indicator(
        "STOCHF",
        high, low, close,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    signals = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(fastk) & ~np.isnan(fastd)
    signals[valid & crossover(fastk, fastd)] = 1
    signals[valid & crossunder(fastk, fastd)] = -1
    signals[valid & ~(crossover(fastk, fastd) | crossunder(fastk, fastd))] = 0
    return signals, window


def trix_signal(close, period=14):
    val, window = call_indicator("TRIX", close, timeperiod=period)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > 0)] = 1
    signal[valid & (val < 0)] = -1
    signal[valid & (val == 0)] = 0
    return signal, window


def sarext_signal(high, low, close):
    sar, window = call_indicator("SAREXT", high, low)
    signal = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(sar) & ~np.isnan(close)
    signal[valid & (close > sar)] = 1
    signal[valid & (close < sar)] = -1
    signal[valid & (close == sar)] = 0
    return signal, window


def dx_signal(high, low, close, period=14):
    val, window = call_indicator("DX", high, low, close, timeperiod=period)
    mean = np.nanmean(val)
    signal = np.full_like(val, np.nan, dtype=np.float32)
    valid = ~np.isnan(val)
    signal[valid & (val > mean)] = 1
    signal[valid & (val < mean)] = -1
    signal[valid & (val == mean)] = 0
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
    signals = np.full_like(close, np.nan, dtype=np.float32)
    valid = ~np.isnan(slowk) & ~np.isnan(slowd)
    signals[valid & crossover(slowk, slowd)] = 1
    signals[valid & crossunder(slowk, slowd)] = -1
    signals[valid & ~(crossover(slowk, slowd) | crossunder(slowk, slowd))] = 0
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
    "PLUS_DM": plus_dm_signal,

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

    "TRIX": trix_signal,
    "STOCHF": stochf_signal,
    "ROC": roc_signal,
    "T3": t3_signal,

    "SAREXT":sarext_signal,
    "DX":dx_signal,
    "STOCH":stoch_signal
}

