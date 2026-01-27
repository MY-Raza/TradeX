# signals.py
import numpy as np

def generate_signals(indicators):
    """
    Generate trading signals based on multiple indicators.
    Returns an array: 1 for buy, -1 for sell, 0 for hold
    """
    # Extract relevant indicators
    close = indicators['close']
    short_ma = indicators['sma']  # e.g., 10-period SMA
    long_ma = indicators['ema']   # e.g., 30-period EMA
    macd = indicators['macd']
    macd_signal = indicators['macd_signal']
    rsi = indicators['rsi']
    upper_bb = indicators['bb_upper']
    lower_bb = indicators['bb_lower']
    atr = indicators['atr']

    signals = np.zeros(len(close))

    for i in range(1, len(close)):
        # Trend: Moving Averages + MACD
        if short_ma[i] > long_ma[i] and macd[i] > macd_signal[i]:
            trend_signal = 1  # Buy
        elif short_ma[i] < long_ma[i] and macd[i] < macd_signal[i]:
            trend_signal = -1  # Sell
        else:
            trend_signal = 0

        # Momentum: RSI
        if rsi[i] > 70:
            momentum_signal = -1  # Overbought → Sell
        elif rsi[i] < 30:
            momentum_signal = 1   # Oversold → Buy
        else:
            momentum_signal = 0

        # Volatility: Bollinger Bands
        if close[i] > upper_bb[i]:
            volatility_signal = -1  # Price too high → Sell
        elif close[i] < lower_bb[i]:
            volatility_signal = 1   # Price too low → Buy
        else:
            volatility_signal = 0

        # Combine signals (majority vote)
        combined = trend_signal + momentum_signal + volatility_signal
        if combined > 0:
            signals[i] = 1
        elif combined < 0:
            signals[i] = -1
        else:
            signals[i] = 0

    return signals
