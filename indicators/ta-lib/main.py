# main.py
import numpy as np
from indicators import *

# ---------------------------
# Generate random price data
# ---------------------------
np.random.seed(42)

data_length = 100

close = np.random.uniform(100, 200, data_length).astype(float)
high = close + np.random.uniform(0, 5, data_length)
low = close - np.random.uniform(0, 5, data_length)

# Variable periods for MAVP
periods = np.random.randint(5, 30, data_length)

# ---------------------------
# Call indicators
# ---------------------------
print("SMA:", sma(close)[-5:])
print("EMA:", ema(close)[-5:])
print("DEMA:", dema(close)[-5:])
print("TEMA:", tema(close)[-5:])
print("WMA:", wma(close)[-5:])
print("KAMA:", kama(close)[-5:])
print("HT Trendline:", ht_trendline(close)[-5:])

upper, middle, lower = bbands(close)
print("BBANDS Upper:", upper[-5:])
print("BBANDS Middle:", middle[-5:])
print("BBANDS Lower:", lower[-5:])

mama_val, fama_val = mama(close)
print("MAMA:", mama_val[-5:])
print("FAMA:", fama_val[-5:])

print("MIDPOINT:", midpoint(close)[-5:])
print("MIDPRICE:", midprice(high, low)[-5:])

print("SAR:", sar(high, low)[-5:])
print("SAREXT:", sarext(high, low)[-5:])

print("MAVP:", mavp(close, periods)[-5:])
