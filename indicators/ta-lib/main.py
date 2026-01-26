"""
main.py

Example usage of TA-Lib indicator wrappers.
"""

import numpy as np
from indicators import call_indicator, indicator_help, get_all_indicators


# ----------------------------
# Generate random market data
# ----------------------------
np.random.seed(42)
length = 100

open_ = np.random.random(length) * 100
high = open_ + np.random.random(length) * 5
low = open_ - np.random.random(length) * 5
close = low + np.random.random(length) * (high - low)
volume = np.random.randint(100, 1000, length)


# ----------------------------
# Example: RSI
# ----------------------------
rsi = call_indicator(
    name="RSI",
    close=close,
    timeperiod=14
)

print("RSI values:")
print(rsi)


# ----------------------------
# Example: MACD
# ----------------------------
macd, macd_signal, macd_hist = call_indicator(
    name="MACD",
    close=close,
    fastperiod=12,
    slowperiod=26,
    signalperiod=9
)

print("\nMACD:")
print(macd)


# ----------------------------
# Example: Bollinger Bands
# ----------------------------
upper, middle, lower = call_indicator(
    name="BBANDS",
    close=close,
    timeperiod=20
)

print("\nBollinger Bands:")
print("Upper:", upper)


# ----------------------------
# List all available indicators
# ----------------------------
print("\nTotal Indicators Available:", len(get_all_indicators()))


# ----------------------------
# Show indicator documentation
# ----------------------------
indicator_help("ADX")
