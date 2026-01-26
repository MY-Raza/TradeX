# main.py
import numpy as np
from indicators import *

# ---------------------------
# Generate random OHLCV data
# ---------------------------
np.random.seed(42)
data_length = 100

# Open between 50 and 200
open_ = np.random.uniform(50, 200, data_length).astype(np.float64)
# High = open + random 0-5
high = open_ + np.random.uniform(0, 5, data_length).astype(np.float64)
# Low = open - random 0-5, but not below 50
low = np.maximum(open_ - np.random.uniform(0, 5, data_length), 50).astype(np.float64)
# Close between low and high
close = np.random.uniform(low, high).astype(np.float64)
# Volume between 1000 and 5000
volume = np.random.uniform(1000, 5000, data_length).astype(np.float64)
# Reference series for statistical indicators
ref = np.random.uniform(50, 200, data_length).astype(np.float64)
# Variable periods for MAVP
periods = np.random.uniform(5, 30, data_length).astype(np.float64)

# ---------------------------
# Moving Averages
# ---------------------------
print("SMA:", sma(close)[-5:])
print("EMA:", ema(close)[-5:])
print("DEMA:", dema(close)[-5:])
print("TEMA:", tema(close)[-5:])
print("WMA:", wma(close)[-5:])
print("KAMA:", kama(close)[-5:])
print("HT Trendline:", ht_trendline(close)[-5:])

# ---------------------------
# Bollinger Bands & Midpoints
# ---------------------------
upper, middle, lower = bbands(close)
print("BBANDS Upper:", upper[-5:])
print("BBANDS Middle:", middle[-5:])
print("BBANDS Lower:", lower[-5:])
print("MIDPOINT:", midpoint(close)[-5:])
print("MIDPRICE:", midprice(high, low)[-5:])

# ---------------------------
# Adaptive / Advanced MAs
# ---------------------------
mama_val, fama_val = mama(close)
print("MAMA:", mama_val[-5:])
print("FAMA:", fama_val[-5:])
print("MAVP:", mavp(close, periods)[-5:])

# ---------------------------
# Parabolic SAR
# ---------------------------
print("SAR:", sar(high, low)[-5:])
print("SAREXT:", sarext(high, low)[-5:])

# ---------------------------
# Directional Movement & ADX
# ---------------------------
print("ADX:", adx(high, low, close)[-5:])
print("ADXR:", adxr(high, low, close)[-5:])
print("PLUS_DI:", plus_di(high, low, close)[-5:])
print("MINUS_DI:", minus_di(high, low, close)[-5:])

# ---------------------------
# MACD & Price Oscillators
# ---------------------------
macd_val, signal, hist = macd(close)
print("MACD:", macd_val[-5:])
print("MACD Signal:", signal[-5:])
print("MACD Hist:", hist[-5:])

# ---------------------------
# Relative Strength
# ---------------------------
print("RSI:", rsi(close)[-5:])

# ---------------------------
# Momentum Indicators
# ---------------------------
print("MOM:", mom(close)[-5:])
print("ROC:", roc(close)[-5:])
print("ROCP:", rocp(close)[-5:])
print("ROCR:", rocr(close)[-5:])
print("ROCR100:", rocr100(close)[-5:])
print("TRIX:", trix(close)[-5:])
print("CMO:", cmo(close)[-5:])

# ---------------------------
# Aroon Indicators
# ---------------------------
aroon_up, aroon_down = aroon(high, low)
print("AROON Up:", aroon_up[-5:])
print("AROON Down:", aroon_down[-5:])
print("AROONOSC:", aroonosc(high, low)[-5:])

# ---------------------------
# Ultimate Oscillator & Williams %R
# ---------------------------
print("ULTOSC:", ultosc(high, low, close)[-5:])
print("WILLR:", willr(high, low, close)[-5:])

# ---------------------------
# Money Flow Indicators
# ---------------------------
print("MFI:", mfi(high, low, close, volume)[-5:])
print("BOP:", bop(open_, high, low, close)[-5:])

# ---------------------------
# Stochastic Indicators
# ---------------------------
slowk, slowd = stoch(high, low, close)
print("STOCH SlowK:", slowk[-5:])
print("STOCH SlowD:", slowd[-5:])

fastk, fastd = stochf(high, low, close)
print("STOCHF FastK:", fastk[-5:])
print("STOCHF FastD:", fastd[-5:])

stochrsi_k, stochrsi_d = stochrsi(close)
print("STOCHRSI FastK:", stochrsi_k[-5:])
print("STOCHRSI FastD:", stochrsi_d[-5:])

# ---------------------------
# Accumulation / Distribution
# ---------------------------
ad_line = ad(high, low, close, volume)
adosc_line = adosc(high, low, close, volume)
obv_line = obv(close, volume)
print("AD Line:", ad_line[-5:])
print("ADOSC:", adosc_line[-5:])
print("OBV:", obv_line[-5:])

# ---------------------------
# Hilbert Transform
# ---------------------------
inphase, quadrature = ht_phasor(close)
sine, leadsine = ht_sine(close)
print("HT_DC Period:", ht_dcperiod(close)[-5:])
print("HT_DC Phase:", ht_dcphase(close)[-5:])
print("HT Phasor InPhase:", inphase[-5:])
print("HT Phasor Quadrature:", quadrature[-5:])
print("HT Sine:", sine[-5:])
print("HT LeadSine:", leadsine[-5:])
print("HT TrendMode:", ht_trendmode(close)[-5:])

# ---------------------------
# Price Transform
# ---------------------------
print("AVGPRICE:", avgprice(open_, high, low, close)[-5:])
print("MEDPRICE:", medprice(high, low)[-5:])
print("TYPPRICE:", typprice(high, low, close)[-5:])
print("WCLPRICE:", wclprice(high, low, close)[-5:])

# ---------------------------
# Volatility Indicators
# ---------------------------
print("ATR:", atr(high, low, close)[-5:])
print("NATR:", natr(high, low, close)[-5:])
print("TRANGE:", trange(high, low, close)[-5:])

# ---------------------------
# Candlestick Pattern Recognition
# ---------------------------
print("CDLDOJI:", candlestick_pattern(open_, high, low, close, "CDLDOJI")[-5:])
print("CDLENGULFING:", candlestick_pattern(open_, high, low, close, "CDLENGULFING")[-5:])
print("CDLMORNINGSTAR:", candlestick_pattern(open_, high, low, close, "CDLMORNINGSTAR")[-5:])

# ---------------------------
# Statistical / Regression Indicators
# ---------------------------
print("BETA:", beta(close, ref)[-5:])
print("CORREL:", correl(close, ref)[-5:])
print("LinearReg:", linearreg(close)[-5:])
print("LinearReg Angle:", linearreg_angle(close)[-5:])
print("LinearReg Intercept:", linearreg_intercept(close)[-5:])
print("LinearReg Slope:", linearreg_slope(close)[-5:])
print("STDDEV:", stddev(close)[-5:])
print("TSF:", tsf(close)[-5:])
print("VAR:", var(close)[-5:])
