import numpy as np
from indicators import *
# =========================================================
# SIGNALS FOR MOVING AVERAGES
# =========================================================

def sma_signal(close, period=14):
    sma_values = sma(close, period)
    signals = np.where(close > sma_values, 1, np.where(close < sma_values, -1, 0))
    return signals

def ema_signal(close, period=14):
    ema_values = ema(close, period)
    signals = np.where(close > ema_values, 1, np.where(close < ema_values, -1, 0))
    return signals

def dema_signal(close, period=14):
    dema_values = dema(close, period)
    signals = np.where(close > dema_values, 1, np.where(close < dema_values, -1, 0))
    return signals

def tema_signal(close, period=14):
    tema_values = tema(close, period)
    signals = np.where(close > tema_values, 1, np.where(close < tema_values, -1, 0))
    return signals

def trima_signal(close, period=14):
    trima_values = trima(close, period)
    signals = np.where(close > trima_values, 1, np.where(close < trima_values, -1, 0))
    return signals

def wma_signal(close, period=14):
    wma_values = wma(close, period)
    signals = np.where(close > wma_values, 1, np.where(close < wma_values, -1, 0))
    return signals

def t3_signal(close, period=14, vfactor=0.7):
    t3_values = t3(close, period, vfactor)
    signals = np.where(close > t3_values, 1, np.where(close < t3_values, -1, 0))
    return signals

def kama_signal(close, period=14):
    kama_values = kama(close, period)
    signals = np.where(close > kama_values, 1, np.where(close < kama_values, -1, 0))
    return signals

def ma_signal(close, period=14, ma_type=0):
    ma_values = ma(close, period, ma_type)
    signals = np.where(close > ma_values, 1, np.where(close < ma_values, -1, 0))
    return signals

# =========================================================
# SIGNALS FOR ADVANCED / ADAPTIVE MOVING AVERAGES
# =========================================================

def ht_trendline_signal(close):
    trendline = ht_trendline(close)
    signals = np.where(close > trendline, 1, np.where(close < trendline, -1, 0))
    return signals

def mama_signal(close, fastlimit=0.5, slowlimit=0.05):
    mama_val, fama_val = mama(close, fastlimit, slowlimit)
    # Signal based on MAMA crossing FAMA
    signals = np.zeros_like(close)
    signals[(mama_val > fama_val) & (np.roll(mama_val, 1) <= np.roll(fama_val, 1))] = 1  # Buy crossover
    signals[(mama_val < fama_val) & (np.roll(mama_val, 1) >= np.roll(fama_val, 1))] = -1 # Sell crossover
    return signals

def mavp_signal(close, periods, minperiod=2, maxperiod=30, ma_type=0):
    mavp_values = mavp(close, periods, minperiod, maxperiod, ma_type)
    signals = np.where(close > mavp_values, 1, np.where(close < mavp_values, -1, 0))
    return signals

# =========================================================
# SIGNALS FOR BANDS & MIDPOINTS
# =========================================================

def bbands_signal(close, period=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = bbands(close, period, nbdevup, nbdevdn)
    signals = np.zeros_like(close)
    # Buy when price crosses below lower band
    signals[(close < lower) & (np.roll(close, 1) >= np.roll(lower, 1))] = 1
    # Sell when price crosses above upper band
    signals[(close > upper) & (np.roll(close, 1) <= np.roll(upper, 1))] = -1
    return signals

def midpoint_signal(close, period=14):
    mp = midpoint(close, period)
    signals = np.where(close > mp, 1, np.where(close < mp, -1, 0))
    return signals

def midprice_signal(high, low, period=14):
    mp = midprice(high, low, period)
    signals = np.where((high+low)/2 > mp, 1, np.where((high+low)/2 < mp, -1, 0))
    return signals

# =========================================================
# SIGNALS FOR PARABOLIC SAR
# =========================================================

def sar_signal(close, high, low, acceleration=0.02, maximum=0.2):
    sar_values = sar(high, low, acceleration, maximum)
    signals = np.where(close > sar_values, 1, np.where(close < sar_values, -1, 0))
    return signals

def sarext_signal(close, high, low,
                  startvalue=-183,
                  offsetonreverse=0,
                  accelerationinitlong=0.02,
                  accelerationlong=0.02,
                  accelerationmaxlong=0.2,
                  accelerationinitshort=0.02,
                  accelerationshort=0.02,
                  accelerationmaxshort=0.2):
    sar_values = sarext(
        high, low,
        startvalue,
        offsetonreverse,
        accelerationinitlong,
        accelerationlong,
        accelerationmaxlong,
        accelerationinitshort,
        accelerationshort,
        accelerationmaxshort
    )
    signals = np.where(close > sar_values, 1, np.where(close < sar_values, -1, 0))
    return signals
# =========================================================
# DIRECTIONAL MOVEMENT & ADX SIGNALS
# =========================================================

def adx_signal(high, low, close, period=14, threshold=25):
    adx_val = adx(high, low, close, period)
    plus = plus_di(high, low, close, period)
    minus = minus_di(high, low, close, period)
    signals = np.zeros_like(close)
    # Trend only if ADX > threshold
    signals[(adx_val > threshold) & (plus > minus)] = 1   # Buy
    signals[(adx_val > threshold) & (plus < minus)] = -1  # Sell
    return signals

def adxr_signal(high, low, close, period=14, threshold=25):
    adxr_val = adxr(high, low, close, period)
    plus = plus_di(high, low, close, period)
    minus = minus_di(high, low, close, period)
    signals = np.zeros_like(close)
    signals[(adxr_val > threshold) & (plus > minus)] = 1
    signals[(adxr_val > threshold) & (plus < minus)] = -1
    return signals

# =========================================================
# PRICE OSCILLATOR SIGNALS
# =========================================================

def apo_signal(close, fastperiod=12, slowperiod=26, matype=0):
    apo_val = apo(close, fastperiod, slowperiod, matype)
    signals = np.where(apo_val > 0, 1, np.where(apo_val < 0, -1, 0))
    return signals

def ppo_signal(close, fastperiod=12, slowperiod=26, matype=0):
    ppo_val = ppo(close, fastperiod, slowperiod, matype)
    signals = np.where(ppo_val > 0, 1, np.where(ppo_val < 0, -1, 0))
    return signals

def macd_signal(close, fastperiod=12, slowperiod=26, signalperiod=9):
    macd_val, signal, hist = macd(close, fastperiod, slowperiod, signalperiod)
    signals = np.zeros_like(close)
    # Buy when MACD crosses above Signal
    signals[(macd_val > signal) & (np.roll(macd_val, 1) <= np.roll(signal, 1))] = 1
    # Sell when MACD crosses below Signal
    signals[(macd_val < signal) & (np.roll(macd_val, 1) >= np.roll(signal, 1))] = -1
    return signals

def macdext_signal(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
    macd_val, signal, hist = macdext(close, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype)
    signals = np.zeros_like(close)
    signals[(macd_val > signal) & (np.roll(macd_val, 1) <= np.roll(signal, 1))] = 1
    signals[(macd_val < signal) & (np.roll(macd_val, 1) >= np.roll(signal, 1))] = -1
    return signals

def macdfix_signal(close, signalperiod=9):
    macd_val, signal, hist = macdfix(close, signalperiod)
    signals = np.zeros_like(close)
    signals[(macd_val > signal) & (np.roll(macd_val, 1) <= np.roll(signal, 1))] = 1
    signals[(macd_val < signal) & (np.roll(macd_val, 1) >= np.roll(signal, 1))] = -1
    return signals

# =========================================================
# MOMENTUM INDICATOR SIGNALS
# =========================================================

def cci_signal(high, low, close, period=14):
    cci_val = cci(high, low, close, period)
    signals = np.where(cci_val > 0, 1, np.where(cci_val < 0, -1, 0))
    return signals

def mom_signal(close, period=10):
    mom_val = mom(close, period)
    signals = np.where(mom_val > 0, 1, np.where(mom_val < 0, -1, 0))
    return signals

def roc_signal(close, period=10):
    roc_val = roc(close, period)
    signals = np.where(roc_val > 0, 1, np.where(roc_val < 0, -1, 0))
    return signals

def rocp_signal(close, period=10):
    rocp_val = rocp(close, period)
    signals = np.where(rocp_val > 0, 1, np.where(rocp_val < 0, -1, 0))
    return signals

def rocr_signal(close, period=10):
    rocr_val = rocr(close, period)
    signals = np.where(rocr_val > 1, 1, np.where(rocr_val < 1, -1, 0))
    return signals

def rocr100_signal(close, period=10):
    rocr100_val = rocr100(close, period)
    signals = np.where(rocr100_val > 100, 1, np.where(rocr100_val < 100, -1, 0))
    return signals

def trix_signal(close, period=30):
    trix_val = trix(close, period)
    signals = np.where(trix_val > 0, 1, np.where(trix_val < 0, -1, 0))
    return signals

def cmo_signal(close, period=14):
    cmo_val = cmo(close, period)
    signals = np.where(cmo_val > 0, 1, np.where(cmo_val < 0, -1, 0))
    return signals

# =========================================================
# VOLUME / MONEY FLOW SIGNALS
# =========================================================

def mfi_signal(high, low, close, volume, period=14):
    mfi_val = mfi(high, low, close, volume, period)
    signals = np.zeros_like(mfi_val)
    signals[mfi_val < 20] = 1   # Buy oversold
    signals[mfi_val > 80] = -1  # Sell overbought
    return signals

def bop_signal(open_, high, low, close):
    bop_val = bop(open_, high, low, close)
    signals = np.where(bop_val > 0, 1, np.where(bop_val < 0, -1, 0))
    return signals

# =========================================================
# AROON INDICATOR SIGNALS
# =========================================================

def aroon_signal(high, low, period=14):
    aroon_up, aroon_down = aroon(high, low, period)
    signals = np.zeros_like(aroon_up)
    signals[aroon_up > aroon_down] = 1   # Uptrend
    signals[aroon_down > aroon_up] = -1  # Downtrend
    return signals

def aroonosc_signal(high, low, period=14):
    osc = aroonosc(high, low, period)
    signals = np.where(osc > 0, 1, np.where(osc < 0, -1, 0))
    return signals

# =========================================================
# RELATIVE STRENGTH & STOCHASTIC SIGNALS
# =========================================================

def rsi_signal(close, period=14, overbought=70, oversold=30):
    rsi_val = rsi(close, period)
    signals = np.where(rsi_val > overbought, -1, np.where(rsi_val < oversold, 1, 0))
    return signals

def stoch_signal(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    slowk, slowd = stoch(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
    signals = np.where(slowk > slowd, 1, np.where(slowk < slowd, -1, 0))
    return signals

def stochf_signal(high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0):
    fastk, fastd = stochf(high, low, close, fastk_period, fastd_period, fastd_matype)
    signals = np.where(fastk > fastd, 1, np.where(fastk < fastd, -1, 0))
    return signals

def stochrsi_signal(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    fastk, fastd = stochrsi(close, timeperiod, fastk_period, fastd_period, fastd_matype)
    signals = np.where(fastk > fastd, 1, np.where(fastk < fastd, -1, 0))
    return signals

def ultosc_signal(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    ult = ultosc(high, low, close, timeperiod1, timeperiod2, timeperiod3)
    signals = np.where(ult > 70, -1, np.where(ult < 30, 1, 0))
    return signals

def willr_signal(high, low, close, period=14):
    will = willr(high, low, close, period)
    signals = np.where(will > -20, -1, np.where(will < -80, 1, 0))
    return signals

def ad_signal(high, low, close, volume):
    ad_val = ad(high, low, close, volume)
    signals = np.where(ad_val > 0, 1, np.where(ad_val < 0, -1, 0))
    return signals

def adosc_signal(high, low, close, volume, fastperiod=3, slowperiod=10):
    osc = adosc(high, low, close, volume, fastperiod, slowperiod)
    signals = np.where(osc > 0, 1, np.where(osc < 0, -1, 0))
    return signals

def obv_signal(close, volume):
    obv_val = obv(close, volume)
    signals = np.zeros_like(obv_val)
    signals[obv_val > np.roll(obv_val, 1)] = 1
    signals[obv_val < np.roll(obv_val, 1)] = -1
    return signals

# =========================================================
# HILBERT TRANSFORM SIGNALS
# =========================================================

def ht_trendmode_signal(close):
    mode = ht_trendmode(close)
    signals = np.where(mode == 1, 1, -1)  # 1=trend, 0=cycle
    return signals

def ht_phasor_signal(close):
    inphase, quadrature = ht_phasor(close)
    signals = np.where(inphase > quadrature, 1, -1)
    return signals

def ht_sine_signal(close):
    sine, leadsine = ht_sine(close)
    signals = np.where(sine > leadsine, 1, -1)
    return signals

# =========================================================
# PRICE TRANSFORM SIGNALS
# =========================================================

def avgprice_signal(open_, high, low, close):
    avg = avgprice(open_, high, low, close)
    signals = np.where(close > avg, 1, -1)
    return signals

def medprice_signal(high, low, close):
    med = medprice(high, low)
    signals = np.where(close > med, 1, -1)
    return signals

def typprice_signal(high, low, close):
    typ = typprice(high, low, close)
    signals = np.where(close > typ, 1, -1)
    return signals

def wclprice_signal(high, low, close):
    wcl = wclprice(high, low, close)
    signals = np.where(close > wcl, 1, -1)
    return signals

# =========================================================
# VOLATILITY SIGNALS
# =========================================================

def atr_signal(high, low, close, period=14):
    atr_val = atr(high, low, close, period)
    signals = np.where(atr_val > np.mean(atr_val), 1, -1)
    return signals

def natr_signal(high, low, close, period=14):
    natr_val = natr(high, low, close, period)
    signals = np.where(natr_val > np.mean(natr_val), 1, -1)
    return signals

def trange_signal(high, low, close):
    tr_val = trange(high, low, close)
    signals = np.where(tr_val > np.mean(tr_val), 1, -1)
    return signals

# =========================================================
# CANDLESTICK PATTERN SIGNALS
# =========================================================

def candlestick_signal(open_, high, low, close, pattern_name):
    val = candlestick_pattern(open_, high, low, close, pattern_name)
    signals = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signals, pattern_name

# =========================================================
# STATISTICAL / REGRESSION SIGNALS
# =========================================================

def beta_signal(close, ref, period=5):
    val = beta(close, ref, period)
    signals = np.where(val > 1, 1, np.where(val < 1, -1, 0))
    return signals

def correl_signal(close, ref, period=30):
    val = correl(close, ref, period)
    signals = np.where(val > 0, 1, np.where(val < 0, -1, 0))
    return signals

def linearreg_angle_signal(close, period=14):
    angle = linearreg_angle(close, period)
    signals = np.where(angle > 0, 1, np.where(angle < 0, -1, 0))
    return signals

def linearreg_slope_signal(close, period=14):
    slope = linearreg_slope(close, period)
    signals = np.where(slope > 0, 1, np.where(slope < 0, -1, 0))
    return signals

def stddev_signal(close, period=14):
    val = stddev(close, period)
    signals = np.where(close > np.mean(close), 1, -1)
    return signals

def tsf_signal(close, period=14):
    val = tsf(close, period)
    signals = np.where(close > val, 1, -1)
    return signals

def var_signal(close, period=14):
    val = var(close, period)
    signals = np.where(val > np.mean(val), 1, -1)
    return signals
