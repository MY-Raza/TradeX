import numpy as np

def generate_signals(data):
    """
    Generate trading signals from indicator outputs.
    data: dict containing all indicator arrays with keys:
        'close', 'sma', 'ema', 'adx', 'plus_di', 'minus_di', 'macd', 'macd_signal',
        'rsi', 'mfi', 'stoch_k', 'stoch_d', 'atr'
    Returns: np.array of signals: 1=Buy, -1=Sell, 0=Hold
    """
    
    close = data['close']
    sma = data['sma']
    ema = data['ema']
    adx = data['adx']
    plus_di = data['plus_di']
    minus_di = data['minus_di']
    macd = data['macd']
    macd_signal = data['macd_signal']
    rsi = data['rsi']
    mfi = data['mfi']
    stoch_k = data['stoch_k']
    stoch_d = data['stoch_d']
    atr = data['atr']
    
    signals = np.zeros(len(close))
    
    for i in range(len(close)):
        buy_signal = 0
        sell_signal = 0

        # ---------------------------
        # Trend direction (ADX + DI)
        # ---------------------------
        if adx[i] > 25:
            if plus_di[i] > minus_di[i]:
                buy_signal += 1
            elif minus_di[i] > plus_di[i]:
                sell_signal += 1

        # ---------------------------
        # MACD crossover
        # ---------------------------
        if i > 0:
            if macd[i-1] < macd_signal[i-1] and macd[i] > macd_signal[i]:
                buy_signal += 1
            elif macd[i-1] > macd_signal[i-1] and macd[i] < macd_signal[i]:
                sell_signal += 1

        # ---------------------------
        # RSI oversold/overbought
        # ---------------------------
        if rsi[i] < 30:
            buy_signal += 1
        elif rsi[i] > 70:
            sell_signal += 1

        # ---------------------------
        # MFI oversold/overbought
        # ---------------------------
        if mfi[i] < 20:
            buy_signal += 1
        elif mfi[i] > 80:
            sell_signal += 1

        # ---------------------------
        # Stochastic %K/%D
        # ---------------------------
        if stoch_k[i] < 20 and stoch_d[i] < 20:
            buy_signal += 1
        elif stoch_k[i] > 80 and stoch_d[i] > 80:
            sell_signal += 1

        # ---------------------------
        # ATR filter (optional: avoid signals in low volatility)
        # ---------------------------
        if atr[i] < np.mean(atr) * 0.5:
            buy_signal = 0
            sell_signal = 0

        # ---------------------------
        # Final decision
        # ---------------------------
        if buy_signal > sell_signal:
            signals[i] = 1
        elif sell_signal > buy_signal:
            signals[i] = -1
        else:
            signals[i] = 0

    return signals
