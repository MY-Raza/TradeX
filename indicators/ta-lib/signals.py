import numpy as np

def generate_signals(data):
    """
    Generate trading signals based on multiple technical indicators.

    This function takes a dictionary of price and indicator arrays and produces a
    signal array indicating potential buy, sell, or hold actions.

    Arguments:
    ----------
    data : dict
        A dictionary containing the following keys with numpy arrays or lists of equal length:
            - 'close'      : Closing prices
            - 'sma'        : Simple Moving Average
            - 'ema'        : Exponential Moving Average
            - 'adx'        : Average Directional Index
            - 'plus_di'    : Plus Directional Indicator (+DI)
            - 'minus_di'   : Minus Directional Indicator (-DI)
            - 'macd'       : MACD line
            - 'macd_signal': MACD signal line
            - 'rsi'        : Relative Strength Index
            - 'mfi'        : Money Flow Index
            - 'stoch_k'    : Stochastic %K
            - 'stoch_d'    : Stochastic %D
            - 'atr'        : Average True Range

    Returns:
    --------
    np.ndarray
        Array of trading signals with same length as input arrays:
            -  1 : Buy signal
            - -1 : Sell signal
            -  0 : Hold / No action
    """
    
    # Extract indicator arrays from input dictionary
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
    
    # Initialize signal array
    signals = np.zeros(len(close))
    
    # Iterate over each time step
    for i in range(len(close)):
        buy_signal = 0
        sell_signal = 0

        # ---------------------------
        # Trend strength using ADX and Directional Indicators
        # ADX > 25 indicates a strong trend
        # +DI > -DI => bullish trend, potential buy
        # -DI > +DI => bearish trend, potential sell
        # ---------------------------
        if adx[i] > 25:
            if plus_di[i] > minus_di[i]:
                buy_signal += 1
            elif minus_di[i] > plus_di[i]:
                sell_signal += 1

        # ---------------------------
        # MACD crossover strategy
        # Buy when MACD crosses above signal line
        # Sell when MACD crosses below signal line
        # ---------------------------
        if i > 0:
            if macd[i-1] < macd_signal[i-1] and macd[i] > macd_signal[i]:
                buy_signal += 1
            elif macd[i-1] > macd_signal[i-1] and macd[i] < macd_signal[i]:
                sell_signal += 1

        # ---------------------------
        # RSI overbought/oversold levels
        # RSI < 30 => oversold => buy
        # RSI > 70 => overbought => sell
        # ---------------------------
        if rsi[i] < 30:
            buy_signal += 1
        elif rsi[i] > 70:
            sell_signal += 1

        # ---------------------------
        # MFI overbought/oversold levels
        # MFI < 20 => oversold => buy
        # MFI > 80 => overbought => sell
        # ---------------------------
        if mfi[i] < 20:
            buy_signal += 1
        elif mfi[i] > 80:
            sell_signal += 1

        # ---------------------------
        # Stochastic oscillator
        # Both %K and %D below 20 => oversold => buy
        # Both %K and %D above 80 => overbought => sell
        # ---------------------------
        if stoch_k[i] < 20 and stoch_d[i] < 20:
            buy_signal += 1
        elif stoch_k[i] > 80 and stoch_d[i] > 80:
            sell_signal += 1

        # ---------------------------
        # ATR volatility filter
        # Ignore signals if volatility is very low (ATR < 50% of mean ATR)
        # ---------------------------
        if atr[i] < np.mean(atr) * 0.5:
            buy_signal = 0
            sell_signal = 0

        # ---------------------------
        # Final decision
        # Assign 1 for Buy, -1 for Sell, 0 for Hold/No action
        # ---------------------------
        if buy_signal > sell_signal:
            signals[i] = 1
        elif sell_signal > buy_signal:
            signals[i] = -1
        else:
            signals[i] = 0

    return signals
