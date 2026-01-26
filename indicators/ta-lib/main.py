# main.py
import numpy as np
from indicators import *
from TradeX.utils.common.logs import get_logger

logger = get_logger("indicators_main")

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
logger.info(f"SMA:, {sma(close)[-5:]}")
logger.info(f"EMA: {ema(close)[-5:]}")
logger.info(f"DEMA: {dema(close)[-5:]}")
logger.info(f"TEMA: {tema(close)[-5:]}")
logger.info(f"WMA: {wma(close)[-5:]}")
logger.info(f"KAMA: {kama(close)[-5:]}")
logger.info(f"HT Trendline: {ht_trendline(close)[-5:]}")

# ---------------------------
# Bollinger Bands & Midpoints
# ---------------------------
upper, middle, lower = bbands(close)
logger.info(f"BBANDS Upper: {upper[-5:]}")
logger.info(f"BBANDS Middle: {middle[-5:]}")
logger.info(f"BBANDS Lower: {lower[-5:]}")
logger.info(f"MIDPOINT: {midpoint(close)[-5:]}")
logger.info(f"MIDPRICE: {midprice(high, low)[-5:]}")

# ---------------------------
# Adaptive / Advanced MAs
# ---------------------------
mama_val, fama_val = mama(close)
logger.info(f"MAMA: {mama_val[-5:]}")
logger.info(f"FAMA: {fama_val[-5:]}")
logger.info(f"MAVP: {mavp(close, periods)[-5:]}")

# ---------------------------
# Parabolic SAR
# ---------------------------
logger.info(f"SAR: {sar(high, low)[-5:]}")
logger.info(f"SAREXT: {sarext(high, low)[-5:]}")

# ---------------------------
# Directional Movement & ADX
# ---------------------------
logger.info(f"ADX: {adx(high, low, close)[-5:]}")
logger.info(f"ADXR: {adxr(high, low, close)[-5:]}")
logger.info(f"PLUS_DI: {plus_di(high, low, close)[-5:]}")
logger.info(f"MINUS_DI: {minus_di(high, low, close)[-5:]}")

# ---------------------------
# MACD & Price Oscillators
# ---------------------------
macd_val, signal, hist = macd(close)
logger.info(f"MACD: {macd_val[-5:]}")
logger.info(f"MACD Signal: {signal[-5:]}")
logger.info(f"MACD Hist: {hist[-5:]}")

# ---------------------------
# Relative Strength
# ---------------------------
logger.info(f"RSI: {rsi(close)[-5:]}")

# ---------------------------
# Momentum Indicators
# ---------------------------
logger.info(f"MOM: {mom(close)[-5:]}")
logger.info(f"ROC: {roc(close)[-5:]}")
logger.info(f"ROCP: {rocp(close)[-5:]}")
logger.info(f"ROCR: {rocr(close)[-5:]}")
logger.info(f"ROCR100: {rocr100(close)[-5:]}")
logger.info(f"TRIX: {trix(close)[-5:]}")
logger.info(f"CMO: {cmo(close)[-5:]}")

# ---------------------------
# Aroon Indicators
# ---------------------------
aroon_up, aroon_down = aroon(high, low)
logger.info(f"AROON Up: {aroon_up[-5:]}")
logger.info(f"AROON Down: {aroon_down[-5:]}")
logger.info(f"AROONOSC: {aroonosc(high, low)[-5:]}")

# ---------------------------
# Ultimate Oscillator & Williams %R
# ---------------------------
logger.info(f"ULTOSC: {ultosc(high, low, close)[-5:]}")
logger.info(f"WILLR: {willr(high, low, close)[-5:]}")

# ---------------------------
# Money Flow Indicators
# ---------------------------
logger.info(f"MFI: {mfi(high, low, close, volume)[-5:]}")
logger.info(f"BOP: {bop(open_, high, low, close)[-5:]}")

# ---------------------------
# Stochastic Indicators
# ---------------------------
slowk, slowd = stoch(high, low, close)
logger.info(f"STOCH SlowK: {slowk[-5:]}")
logger.info(f"STOCH SlowD: {slowd[-5:]}")

fastk, fastd = stochf(high, low, close)
logger.info(f"STOCHF FastK: {fastk[-5:]}")
logger.info(f"STOCHF FastD: {fastd[-5:]}")

stochrsi_k, stochrsi_d = stochrsi(close)
logger.info(f"STOCHRSI FastK: {stochrsi_k[-5:]}")
logger.info(f"STOCHRSI FastD: {stochrsi_d[-5:]}")

# ---------------------------
# Accumulation / Distribution
# ---------------------------
ad_line = ad(high, low, close, volume)
adosc_line = adosc(high, low, close, volume)
obv_line = obv(close, volume)
logger.info(f"AD Line: {ad_line[-5:]}")
logger.info(f"ADOSC: {adosc_line[-5:]}")
logger.info(f"OBV: {obv_line[-5:]}")

# ---------------------------
# Hilbert Transform
# ---------------------------
inphase, quadrature = ht_phasor(close)
sine, leadsine = ht_sine(close)
logger.info(f"HT_DC Period: {ht_dcperiod(close)[-5:]}")
logger.info(f"HT_DC Phase: {ht_dcphase(close)[-5:]}")
logger.info(f"HT Phasor InPhase: {inphase[-5:]}")
logger.info(f"HT Phasor Quadrature: {quadrature[-5:]}")
logger.info(f"HT Sine: {sine[-5:]}")
logger.info(f"HT LeadSine: {leadsine[-5:]}")
logger.info(f"HT TrendMode: {ht_trendmode(close)[-5:]}")

# ---------------------------
# Price Transform
# ---------------------------
logger.info(f"AVGPRICE: {avgprice(open_, high, low, close)[-5:]}")
logger.info(f"MEDPRICE: {medprice(high, low)[-5:]}")
logger.info(f"TYPPRICE: {typprice(high, low, close)[-5:]}")
logger.info(f"WCLPRICE: {wclprice(high, low, close)[-5:]}")

# ---------------------------
# Volatility Indicators
# ---------------------------
logger.info(f"ATR: {atr(high, low, close)[-5:]}")
logger.info(f"NATR: {natr(high, low, close)[-5:]}")
logger.info(f"TRANGE: {trange(high, low, close)[-5:]}")

# ---------------------------
# Candlestick Pattern Recognition
# ---------------------------
logger.info(f"CDLDOJI: {candlestick_pattern(open_, high, low, close, 'CDLDOJI')[-5:]}")
logger.info(f"CDLENGULFING: {candlestick_pattern(open_, high, low, close, 'CDLENGULFING')[-5:]}")
logger.info(f"CDLMORNINGSTAR: {candlestick_pattern(open_, high, low, close, 'CDLMORNINGSTAR')[-5:]}")

# ---------------------------
# Statistical / Regression Indicators
# ---------------------------
logger.info(f"BETA: {beta(close, ref)[-5:]}")
logger.info(f"CORREL: {correl(close, ref)[-5:]}")
logger.info(f"LinearReg: {linearreg(close)[-5:]}")
logger.info(f"LinearReg Angle: {linearreg_angle(close)[-5:]}")
logger.info(f"LinearReg Intercept: {linearreg_intercept(close)[-5:]}")
logger.info(f"LinearReg Slope: {linearreg_slope(close)[-5:]}")
logger.info(f"STDDEV: {stddev(close)[-5:]}")
logger.info(f"TSF: {tsf(close)[-5:]}")
logger.info(f"VAR: {var(close)[-5:]}")