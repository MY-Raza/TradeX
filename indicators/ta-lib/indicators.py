# indicators.py
import numpy as np
import talib


# ---------------------------
# Moving Averages
# ---------------------------
def sma(close, period=14):
    return talib.SMA(close, timeperiod=period)

def ema(close, period=14):
    return talib.EMA(close, timeperiod=period)

def dema(close, period=14):
    return talib.DEMA(close, timeperiod=period)

def tema(close, period=14):
    return talib.TEMA(close, timeperiod=period)

def trima(close, period=14):
    return talib.TRIMA(close, timeperiod=period)

def wma(close, period=14):
    return talib.WMA(close, timeperiod=period)

def t3(close, period=14, vfactor=0.7):
    return talib.T3(close, timeperiod=period, vfactor=vfactor)

def kama(close, period=14):
    return talib.KAMA(close, timeperiod=period)

def ma(close, period=14, ma_type=0):
    """
    ma_type:
    0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA
    """
    return talib.MA(close, timeperiod=period, matype=ma_type)


# ---------------------------
# Advanced / Adaptive MAs
# ---------------------------
def ht_trendline(close):
    return talib.HT_TRENDLINE(close)

def mama(close, fastlimit=0.5, slowlimit=0.05):
    mama, fama = talib.MAMA(close, fastlimit=fastlimit, slowlimit=slowlimit)
    return mama, fama

def mavp(close, periods, minperiod=2, maxperiod=30, ma_type=0):
    close = np.asarray(close, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)
    return talib.MAVP(
        close,
        periods,
        minperiod=minperiod,
        maxperiod=maxperiod,
        matype=ma_type
    )


# ---------------------------
# Bands & Midpoints
# ---------------------------
def bbands(close, period=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn
    )
    return upper, middle, lower

def midpoint(close, period=14):
    return talib.MIDPOINT(close, timeperiod=period)

def midprice(high, low, period=14):
    return talib.MIDPRICE(high, low, timeperiod=period)


# ---------------------------
# Parabolic SAR
# ---------------------------
def sar(high, low, acceleration=0.02, maximum=0.2):
    return talib.SAR(high, low, acceleration=acceleration, maximum=maximum)

def sarext(
    high, low,
    startvalue=-183,
    offsetonreverse=0,
    accelerationinitlong=0.02,
    accelerationlong=0.02,
    accelerationmaxlong=0.2,
    accelerationinitshort=0.02,
    accelerationshort=0.02,
    accelerationmaxshort=0.2
):
    return talib.SAREXT(
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

# indicators.py (additional functions)
import numpy as np
import talib

# ---------------------------
# Directional Movement & ADX
# ---------------------------
def adx(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ADX(high, low, close, timeperiod=period)

def adxr(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ADXR(high, low, close, timeperiod=period)

def plus_di(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.PLUS_DI(high, low, close, timeperiod=period)

def minus_di(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.MINUS_DI(high, low, close, timeperiod=period)

def plus_dm(high, low, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.PLUS_DM(high, low, timeperiod=period)

def minus_dm(high, low, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.MINUS_DM(high, low, timeperiod=period)

def dx(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.DX(high, low, close, timeperiod=period)


# ---------------------------
# Price Oscillators
# ---------------------------
def apo(close, fastperiod=12, slowperiod=26, matype=0):
    close = np.asarray(close, dtype=np.float64)
    return talib.APO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

def ppo(close, fastperiod=12, slowperiod=26, matype=0):
    close = np.asarray(close, dtype=np.float64)
    return talib.PPO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

def macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
    close = np.asarray(close, dtype=np.float64)
    macd_val, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return macd_val, signal, hist

def macdext(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
    close = np.asarray(close, dtype=np.float64)
    macd_val, signal, hist = talib.MACDEXT(
        close,
        fastperiod=fastperiod,
        fastmatype=fastmatype,
        slowperiod=slowperiod,
        slowmatype=slowmatype,
        signalperiod=signalperiod,
        signalmatype=signalmatype
    )
    return macd_val, signal, hist

def macdfix(close, signalperiod=9):
    close = np.asarray(close, dtype=np.float64)
    macd_val, signal, hist = talib.MACDFIX(close, signalperiod=signalperiod)
    return macd_val, signal, hist


# ---------------------------
# Momentum Indicators
# ---------------------------
def cci(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.CCI(high, low, close, timeperiod=period)

def mom(close, period=10):
    close = np.asarray(close, dtype=np.float64)
    return talib.MOM(close, timeperiod=period)

def roc(close, period=10):
    close = np.asarray(close, dtype=np.float64)
    return talib.ROC(close, timeperiod=period)

def rocp(close, period=10):
    close = np.asarray(close, dtype=np.float64)
    return talib.ROCP(close, timeperiod=period)

def rocr(close, period=10):
    close = np.asarray(close, dtype=np.float64)
    return talib.ROCR(close, timeperiod=period)

def rocr100(close, period=10):
    close = np.asarray(close, dtype=np.float64)
    return talib.ROCR100(close, timeperiod=period)

def trix(close, period=30):
    close = np.asarray(close, dtype=np.float64)
    return talib.TRIX(close, timeperiod=period)

def cmo(close, period=14):
    close = np.asarray(close, dtype=np.float64)
    return talib.CMO(close, timeperiod=period)


# ---------------------------
# Volume / Money Flow
# ---------------------------
def mfi(high, low, close, volume, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.MFI(high, low, close, volume, timeperiod=period)

def bop(open_, high, low, close):
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.BOP(open_, high, low, close)


# ---------------------------
# Aroon Indicators
# ---------------------------
def aroon(high, low, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    aroon_up, aroon_down = talib.AROON(high, low, timeperiod=period)
    return aroon_up, aroon_down

def aroonosc(high, low, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.AROONOSC(high, low, timeperiod=period)


# ---------------------------
# Relative Strength & Stochastic
# ---------------------------
def rsi(close, period=14):
    close = np.asarray(close, dtype=np.float64)
    return talib.RSI(close, timeperiod=period)

def stoch(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    slowk, slowd = talib.STOCH(high, low, close,
                               fastk_period=fastk_period,
                               slowk_period=slowk_period,
                               slowk_matype=slowk_matype,
                               slowd_period=slowd_period,
                               slowd_matype=slowd_matype)
    return slowk, slowd

def stochf(high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    fastk, fastd = talib.STOCHF(high, low, close,
                                fastk_period=fastk_period,
                                fastd_period=fastd_period,
                                fastd_matype=fastd_matype)
    return fastk, fastd

def stochrsi(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    close = np.asarray(close, dtype=np.float64)
    fastk, fastd = talib.STOCHRSI(close,
                                  timeperiod=timeperiod,
                                  fastk_period=fastk_period,
                                  fastd_period=fastd_period,
                                  fastd_matype=fastd_matype)
    return fastk, fastd


# ---------------------------
# Ultimate Oscillator & Williams %R
# ---------------------------
def ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ULTOSC(high, low, close,
                        timeperiod1=timeperiod1,
                        timeperiod2=timeperiod2,
                        timeperiod3=timeperiod3)

def willr(high, low, close, period=14):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.WILLR(high, low, close, timeperiod=period)

def ad(high, low, close, volume):
    """
    Chaikin Accumulation/Distribution Line
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.AD(high, low, close, volume)

def adosc(high, low, close, volume, fastperiod=3, slowperiod=10):
    """
    Chaikin A/D Oscillator
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.ADOSC(high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod)

def obv(close, volume):
    """
    On Balance Volume
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.OBV(close, volume)

# ---------------------------
# Hilbert Transform Indicators
# ---------------------------
def ht_dcperiod(close):
    """
    Hilbert Transform - Dominant Cycle Period
    Returns the estimated dominant cycle period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.HT_DCPERIOD(close)

def ht_dcphase(close):
    """
    Hilbert Transform - Dominant Cycle Phase
    Returns the estimated dominant cycle phase.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.HT_DCPHASE(close)

def ht_phasor(close):
    """
    Hilbert Transform - Phasor Components
    Returns InPhase and Quadrature components.
    """
    close = np.asarray(close, dtype=np.float64)
    inphase, quadrature = talib.HT_PHASOR(close)
    return inphase, quadrature

def ht_sine(close):
    """
    Hilbert Transform - SineWave
    Returns sine and leadsine components.
    """
    close = np.asarray(close, dtype=np.float64)
    sine, leadsine = talib.HT_SINE(close)
    return sine, leadsine

def ht_trendmode(close):
    """
    Hilbert Transform - Trend vs Cycle Mode
    Returns 0 for cycle, 1 for trend.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.HT_TRENDMODE(close)
