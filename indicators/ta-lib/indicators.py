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
    startvalue=0,
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
