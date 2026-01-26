# indicators.py
import numpy as np
import talib

# =========================================================
# MOVING AVERAGES
# =========================================================

def sma(close, period=14):
    """
    Simple Moving Average (SMA)
    
    Calculates the average of closing prices over a specified period.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods to average (default=14).
    
    Returns:
        np.ndarray: SMA values for each period. The first `period-1` values are NaN.
    """
    return talib.SMA(close, timeperiod=period)


def ema(close, period=14):
    """
    Exponential Moving Average (EMA)
    
    Calculates a weighted moving average giving more weight to recent prices.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate EMA (default=14).
    
    Returns:
        np.ndarray: EMA values for each period.
    """
    return talib.EMA(close, timeperiod=period)


def dema(close, period=14):
    """
    Double Exponential Moving Average (DEMA)
    
    Reduces lag compared to a traditional EMA by applying EMA twice.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods (default=14).
    
    Returns:
        np.ndarray: DEMA values.
    """
    return talib.DEMA(close, timeperiod=period)


def tema(close, period=14):
    """
    Triple Exponential Moving Average (TEMA)
    
    Further reduces lag by applying EMA three times.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods (default=14).
    
    Returns:
        np.ndarray: TEMA values.
    """
    return talib.TEMA(close, timeperiod=period)


def trima(close, period=14):
    """
    Triangular Moving Average (TRIMA)
    
    A weighted moving average emphasizing the middle portion of the window.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods (default=14).
    
    Returns:
        np.ndarray: TRIMA values.
    """
    return talib.TRIMA(close, timeperiod=period)


def wma(close, period=14):
    """
    Weighted Moving Average (WMA)
    
    A moving average giving more weight to recent prices linearly.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods (default=14).
    
    Returns:
        np.ndarray: WMA values.
    """
    return talib.WMA(close, timeperiod=period)


def t3(close, period=14, vfactor=0.7):
    """
    T3 Moving Average
    
    Smooths prices using a triple EMA with a volume factor.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods (default=14).
        vfactor (float, optional): Volume factor for smoothing (default=0.7).
    
    Returns:
        np.ndarray: T3 values.
    """
    return talib.T3(close, timeperiod=period, vfactor=vfactor)


def kama(close, period=14):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Adapts to market volatility, reducing noise in sideways markets.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods (default=14).
    
    Returns:
        np.ndarray: KAMA values.
    """
    return talib.KAMA(close, timeperiod=period)


def ma(close, period=14, ma_type=0):
    """
    Generic Moving Average Selector
    
    Selects between SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA based on `ma_type`.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods (default=14).
        ma_type (int, optional): Type of MA: 
            0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA
    
    Returns:
        np.ndarray: Moving average values.
    """
    return talib.MA(close, timeperiod=period, matype=ma_type)

# =========================================================
# ADVANCED / ADAPTIVE MOVING AVERAGES
# =========================================================

def ht_trendline(close):
    """
    Hilbert Transform - Trendline (HT_TRENDLINE)
    
    Estimates the trendline of the input data using Hilbert Transform.
    Useful for identifying the underlying trend of a price series.
    
    Parameters:
        close (array-like): Closing prices (or any time series data).
    
    Returns:
        np.ndarray: Trendline values for each input period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.HT_TRENDLINE(close)


def mama(close, fastlimit=0.5, slowlimit=0.05):
    """
    MESA Adaptive Moving Average (MAMA)
    
    Adaptive moving average that adjusts to price volatility using the MESA algorithm.
    Produces both MAMA and FAMA (Following Adaptive Moving Average).
    
    Parameters:
        close (array-like): Closing prices.
        fastlimit (float, optional): Fast limit for responsiveness (default=0.5).
        slowlimit (float, optional): Slow limit for smoothing (default=0.05).
    
    Returns:
        tuple of np.ndarray: 
            - mama: The adaptive moving average (faster-moving).
            - fama: Following adaptive moving average (slower-moving).
    """
    close = np.asarray(close, dtype=np.float64)
    mama_val, fama_val = talib.MAMA(close, fastlimit=fastlimit, slowlimit=slowlimit)
    return mama_val, fama_val


def mavp(close, periods, minperiod=2, maxperiod=30, ma_type=0):
    """
    Moving Average with Variable Period (MAVP)
    
    Calculates a moving average using variable periods specified by the `periods` array.
    Allows the MA to adapt to changing market conditions.
    
    Parameters:
        close (array-like): Closing prices.
        periods (array-like): Array of period values for each data point.
        minperiod (int, optional): Minimum period allowed (default=2).
        maxperiod (int, optional): Maximum period allowed (default=30).
        ma_type (int, optional): Type of moving average (0=SMA, 1=EMA, 2=WMA, etc.) (default=0).
    
    Returns:
        np.ndarray: MAVP values for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)
    return talib.MAVP(
        close,
        periods,
        minperiod=minperiod,
        maxperiod=maxperiod,
        matype=ma_type
    )

# =========================================================
# BANDS & MIDPOINTS
# =========================================================

def bbands(close, period=20, nbdevup=2, nbdevdn=2):
    """
    Bollinger Bands (BBANDS)
    
    Calculates Bollinger Bands for a series of closing prices. 
    Bollinger Bands consist of an upper band, middle band (SMA), 
    and lower band, providing a measure of price volatility.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for the moving average (default=20).
        nbdevup (float, optional): Number of standard deviations for the upper band (default=2).
        nbdevdn (float, optional): Number of standard deviations for the lower band (default=2).
    
    Returns:
        tuple of np.ndarray: 
            - upper (np.ndarray): Upper Bollinger Band.
            - middle (np.ndarray): Middle Bollinger Band (SMA).
            - lower (np.ndarray): Lower Bollinger Band.
    """
    close = np.asarray(close, dtype=np.float64)
    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn
    )
    return upper, middle, lower


def midpoint(close, period=14):
    """
    Midpoint over Period (MIDPOINT)
    
    Calculates the midpoint value of a series over the specified period.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate the midpoint (default=14).
    
    Returns:
        np.ndarray: Midpoint values for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.MIDPOINT(close, timeperiod=period)


def midprice(high, low, period=14):
    """
    Midpoint Price over Period (MIDPRICE)
    
    Calculates the mid-price of high and low over the specified period.
    Useful for identifying the average price range.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        period (int, optional): Number of periods for calculation (default=14).
    
    Returns:
        np.ndarray: Midprice values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.MIDPRICE(high, low, timeperiod=period)

# =========================================================
# PARABOLIC SAR
# =========================================================

def sar(high, low, acceleration=0.02, maximum=0.2):
    """
    Parabolic Stop and Reverse (SAR)
    
    Calculates the Parabolic SAR indicator, which identifies potential trend reversals 
    and trailing stop levels. Commonly used to determine stop-loss points in trending markets.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        acceleration (float, optional): Acceleration factor (default=0.02). Controls the sensitivity.
        maximum (float, optional): Maximum acceleration factor (default=0.2). Limits SAR speed.
    
    Returns:
        np.ndarray: SAR values for each period. Values below price indicate an uptrend, above price indicate a downtrend.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
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
    """
    Extended Parabolic SAR (SAREXT)
    
    Calculates an extended Parabolic SAR with separate acceleration factors for long and short trends.
    Provides more flexibility in tuning sensitivity and trend detection.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        startvalue (float, optional): Starting value for SAR (default=-183). Usually set to -1 for auto-detection.
        offsetonreverse (float, optional): Offset applied on reversal (default=0).
        accelerationinitlong (float, optional): Initial acceleration for long positions (default=0.02).
        accelerationlong (float, optional): Acceleration for long positions (default=0.02).
        accelerationmaxlong (float, optional): Maximum acceleration for long positions (default=0.2).
        accelerationinitshort (float, optional): Initial acceleration for short positions (default=0.02).
        accelerationshort (float, optional): Acceleration for short positions (default=0.02).
        accelerationmaxshort (float, optional): Maximum acceleration for short positions (default=0.2).
    
    Returns:
        np.ndarray: Extended SAR values for each period. Values below price indicate uptrend, above price indicate downtrend.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
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

# =========================================================
# DIRECTIONAL MOVEMENT & ADX
# =========================================================

def adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    
    Measures the strength of a trend without considering its direction.
    Higher values indicate stronger trends; lower values indicate weak or sideways markets.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate ADX (default=14).
    
    Returns:
        np.ndarray: ADX values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ADX(high, low, close, timeperiod=period)


def adxr(high, low, close, period=14):
    """
    Average Directional Index Rating (ADXR)
    
    Smoothed version of ADX that averages current ADX with its value `period` ago.
    Helps reduce noise and provides a lagged trend strength signal.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate ADXR (default=14).
    
    Returns:
        np.ndarray: ADXR values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ADXR(high, low, close, timeperiod=period)


def plus_di(high, low, close, period=14):
    """
    Plus Directional Indicator (+DI)
    
    Measures the strength of upward price movement relative to the previous period.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate +DI (default=14).
    
    Returns:
        np.ndarray: +DI values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.PLUS_DI(high, low, close, timeperiod=period)


def minus_di(high, low, close, period=14):
    """
    Minus Directional Indicator (-DI)
    
    Measures the strength of downward price movement relative to the previous period.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate -DI (default=14).
    
    Returns:
        np.ndarray: -DI values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.MINUS_DI(high, low, close, timeperiod=period)


def plus_dm(high, low, period=14):
    """
    Plus Directional Movement (+DM)
    
    Calculates upward price movement relative to the previous period.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        period (int, optional): Number of periods to calculate +DM (default=14).
    
    Returns:
        np.ndarray: +DM values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.PLUS_DM(high, low, timeperiod=period)


def minus_dm(high, low, period=14):
    """
    Minus Directional Movement (-DM)
    
    Calculates downward price movement relative to the previous period.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        period (int, optional): Number of periods to calculate -DM (default=14).
    
    Returns:
        np.ndarray: -DM values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.MINUS_DM(high, low, timeperiod=period)


def dx(high, low, close, period=14):
    """
    Directional Movement Index (DX)
    
    Measures the absolute difference between +DI and -DI relative to the sum of +DI and -DI.
    Provides a normalized measure of trend strength.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate DX (default=14).
    
    Returns:
        np.ndarray: DX values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.DX(high, low, close, timeperiod=period)

# =========================================================
# PRICE OSCILLATORS
# =========================================================

def apo(close, fastperiod=12, slowperiod=26, matype=0):
    """
    Absolute Price Oscillator (APO)
    
    Measures the difference between two moving averages (fast and slow) of closing prices.
    Used to identify trend direction and momentum.
    
    Parameters:
        close (array-like): Closing prices.
        fastperiod (int, optional): Period for the fast moving average (default=12).
        slowperiod (int, optional): Period for the slow moving average (default=26).
        matype (int, optional): Type of moving average (0=SMA, 1=EMA, 2=WMA, etc.) (default=0).
    
    Returns:
        np.ndarray: APO values for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.APO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)


def ppo(close, fastperiod=12, slowperiod=26, matype=0):
    """
    Percentage Price Oscillator (PPO)
    
    Measures the percentage difference between two moving averages (fast and slow) of closing prices.
    Similar to APO but normalized as a percentage, useful for comparing instruments of different scales.
    
    Parameters:
        close (array-like): Closing prices.
        fastperiod (int, optional): Period for the fast moving average (default=12).
        slowperiod (int, optional): Period for the slow moving average (default=26).
        matype (int, optional): Type of moving average (0=SMA, 1=EMA, 2=WMA, etc.) (default=0).
    
    Returns:
        np.ndarray: PPO values for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.PPO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)


def macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Moving Average Convergence Divergence (MACD)
    
    Measures the difference between two exponential moving averages (EMAs) of closing prices.
    Generates a MACD line, a signal line, and a histogram (difference between MACD and signal).
    Used for trend-following and momentum analysis.
    
    Parameters:
        close (array-like): Closing prices.
        fastperiod (int, optional): Period for the fast EMA (default=12).
        slowperiod (int, optional): Period for the slow EMA (default=26).
        signalperiod (int, optional): Period for the signal EMA (default=9).
    
    Returns:
        tuple of np.ndarray:
            - macd_val (np.ndarray): MACD line values.
            - signal (np.ndarray): Signal line values.
            - hist (np.ndarray): Histogram values (MACD - Signal).
    """
    close = np.asarray(close, dtype=np.float64)
    macd_val, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return macd_val, signal, hist


def macdext(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
    """
    Extended Moving Average Convergence Divergence (MACDEXT)
    
    Advanced MACD calculation allowing different moving average types for fast, slow, and signal lines.
    
    Parameters:
        close (array-like): Closing prices.
        fastperiod (int, optional): Period for the fast MA (default=12).
        fastmatype (int, optional): MA type for fast line (default=0=SMA).
        slowperiod (int, optional): Period for the slow MA (default=26).
        slowmatype (int, optional): MA type for slow line (default=0=SMA).
        signalperiod (int, optional): Period for the signal line (default=9).
        signalmatype (int, optional): MA type for signal line (default=0=SMA).
    
    Returns:
        tuple of np.ndarray:
            - macd_val (np.ndarray): MACD line values.
            - signal (np.ndarray): Signal line values.
            - hist (np.ndarray): Histogram values (MACD - Signal).
    """
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
    """
    Fixed-period MACD (MACDFIX)
    
    Variation of MACD where the slow EMA period is fixed at 26.
    Generates MACD line, signal line, and histogram.
    
    Parameters:
        close (array-like): Closing prices.
        signalperiod (int, optional): Period for the signal EMA (default=9).
    
    Returns:
        tuple of np.ndarray:
            - macd_val (np.ndarray): MACD line values.
            - signal (np.ndarray): Signal line values.
            - hist (np.ndarray): Histogram values (MACD - Signal).
    """
    close = np.asarray(close, dtype=np.float64)
    macd_val, signal, hist = talib.MACDFIX(close, signalperiod=signalperiod)
    return macd_val, signal, hist

# =========================================================
# MOMENTUM INDICATORS
# =========================================================

def cci(high, low, close, period=14):
    """
    Commodity Channel Index (CCI)
    
    Measures the variation of price from its statistical mean.
    Positive values indicate price is above average, negative below.
    Useful for identifying overbought/oversold conditions and trend strength.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=14).
    
    Returns:
        np.ndarray: CCI values for each period.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.CCI(high, low, close, timeperiod=period)


def mom(close, period=10):
    """
    Momentum (MOM)
    
    Measures the rate of change of closing prices over a specified period.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods to calculate momentum (default=10).
    
    Returns:
        np.ndarray: Momentum values for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.MOM(close, timeperiod=period)


def roc(close, period=10):
    """
    Rate of Change (ROC)
    
    Measures the percentage change in price over a specified period.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=10).
    
    Returns:
        np.ndarray: ROC values in percentage for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.ROC(close, timeperiod=period)


def rocp(close, period=10):
    """
    Rate of Change Percentage (ROCP)
    
    Measures the relative change in price over a specified period.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=10).
    
    Returns:
        np.ndarray: ROCP values as relative change for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.ROCP(close, timeperiod=period)


def rocr(close, period=10):
    """
    Rate of Change Ratio (ROCR)
    
    Calculates the ratio of current price to the price `period` ago.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=10).
    
    Returns:
        np.ndarray: ROCR values (ratio) for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.ROCR(close, timeperiod=period)


def rocr100(close, period=10):
    """
    Rate of Change Ratio 100 (ROCR100)
    
    Similar to ROCR but scaled by 100 for percentage representation.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=10).
    
    Returns:
        np.ndarray: ROCR100 values in percentage for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.ROCR100(close, timeperiod=period)


def trix(close, period=30):
    """
    Triple Exponential Average (TRIX)
    
    Measures the percentage rate of change of a triple-smoothed EMA.
    Used to filter out insignificant price movements and identify trend reversals.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=30).
    
    Returns:
        np.ndarray: TRIX values for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.TRIX(close, timeperiod=period)


def cmo(close, period=14):
    """
    Chande Momentum Oscillator (CMO)
    
    Measures momentum by calculating the difference between the sum of gains and losses.
    Oscillates between -100 and +100, helping identify overbought/oversold conditions.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=14).
    
    Returns:
        np.ndarray: CMO values for each period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.CMO(close, timeperiod=period)

# =========================================================
# VOLUME / MONEY FLOW INDICATORS
# =========================================================

def mfi(high, low, close, volume, period=14):
    """
    Money Flow Index (MFI)
    
    Measures the strength of money flowing in and out of a security over a period.
    Combines price and volume data to identify overbought or oversold conditions.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        volume (array-like): Trading volume corresponding to prices.
        period (int, optional): Number of periods for calculation (default=14).
    
    Returns:
        np.ndarray: MFI values ranging between 0 and 100.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.MFI(high, low, close, volume, timeperiod=period)


def bop(open_, high, low, close):
    """
    Balance of Power (BOP)
    
    Measures the strength of buyers versus sellers by comparing price changes.
    Positive values indicate buying pressure, negative values indicate selling pressure.
    
    Parameters:
        open_ (array-like): Opening prices.
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
    
    Returns:
        np.ndarray: BOP values, where positive indicates bullish pressure and negative indicates bearish pressure.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.BOP(open_, high, low, close)


# =========================================================
# AROON INDICATORS
# =========================================================

def aroon(high, low, period=14):
    """
    Aroon Indicator
    
    Measures the strength and direction of a trend.  
    Consists of two lines: Aroon Up and Aroon Down.
    - Aroon Up: Indicates how long it has been since the highest high within the period.
    - Aroon Down: Indicates how long it has been since the lowest low within the period.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        period (int, optional): Number of periods to calculate the Aroon indicator (default=14).
    
    Returns:
        tuple of np.ndarray: 
            aroon_up (np.ndarray): Aroon Up values ranging from 0 to 100.
            aroon_down (np.ndarray): Aroon Down values ranging from 0 to 100.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    aroon_up, aroon_down = talib.AROON(high, low, timeperiod=period)
    return aroon_up, aroon_down


def aroonosc(high, low, period=14):
    """
    Aroon Oscillator
    
    Measures the difference between Aroon Up and Aroon Down.
    Values above 0 indicate an uptrend, below 0 indicate a downtrend.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        period (int, optional): Number of periods for calculation (default=14).
    
    Returns:
        np.ndarray: Aroon Oscillator values ranging from -100 to 100.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.AROONOSC(high, low, timeperiod=period)


# =========================================================
# RELATIVE STRENGTH & STOCHASTIC INDICATORS
# =========================================================

def rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    
    Measures the speed and change of price movements to identify overbought or oversold conditions.
    
    Parameters:
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=14).
    
    Returns:
        np.ndarray: RSI values ranging from 0 to 100.
                    Values above 70 generally indicate overbought, below 30 indicate oversold.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.RSI(close, timeperiod=period)


def stoch(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    """
    Stochastic Oscillator (Slow)
    
    Measures the location of the closing price relative to the high-low range over a period.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        fastk_period (int, optional): Period for the fast %K calculation (default=14).
        slowk_period (int, optional): Smoothing period for %K (default=3).
        slowk_matype (int, optional): Moving average type for %K smoothing (default=0, simple).
        slowd_period (int, optional): Smoothing period for %D (default=3).
        slowd_matype (int, optional): Moving average type for %D smoothing (default=0, simple).
    
    Returns:
        tuple of np.ndarray: 
            slowk (np.ndarray): Smoothed %K values.
            slowd (np.ndarray): Smoothed %D values.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=slowk_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype
    )
    return slowk, slowd


def stochf(high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0):
    """
    Stochastic Oscillator (Fast)
    
    Measures the position of the closing price relative to the high-low range using fast %K and %D.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        fastk_period (int, optional): Period for the fast %K calculation (default=14).
        fastd_period (int, optional): Smoothing period for %D (default=3).
        fastd_matype (int, optional): Moving average type for %D smoothing (default=0, simple).
    
    Returns:
        tuple of np.ndarray: 
            fastk (np.ndarray): Fast %K values.
            fastd (np.ndarray): Fast %D values.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    fastk, fastd = talib.STOCHF(
        high, low, close,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    return fastk, fastd


def stochrsi(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    """
    Stochastic RSI (StochRSI)
    
    Measures the level of RSI relative to its range over a specified period, giving a more sensitive oscillator.
    
    Parameters:
        close (array-like): Closing prices.
        timeperiod (int, optional): Number of periods for RSI calculation (default=14).
        fastk_period (int, optional): Period for the fast %K calculation of StochRSI (default=5).
        fastd_period (int, optional): Smoothing period for %D (default=3).
        fastd_matype (int, optional): Moving average type for %D smoothing (default=0, simple).
    
    Returns:
        tuple of np.ndarray: 
            fastk (np.ndarray): Fast %K values of StochRSI.
            fastd (np.ndarray): Smoothed %D values of StochRSI.
    """
    close = np.asarray(close, dtype=np.float64)
    fastk, fastd = talib.STOCHRSI(
        close,
        timeperiod=timeperiod,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype
    )
    return fastk, fastd

# =========================================================
# ULTIMATE OSCILLATOR, WILLIAMS %R & VOLUME / MONEY FLOW
# =========================================================

def ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    """
    Ultimate Oscillator (ULTOSC)
    
    Combines short, medium, and long-term price action into a single oscillator to reduce false signals.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        timeperiod1 (int, optional): Short-term period (default=7).
        timeperiod2 (int, optional): Medium-term period (default=14).
        timeperiod3 (int, optional): Long-term period (default=28).
    
    Returns:
        np.ndarray: Ultimate Oscillator values ranging typically between 0 and 100.
                    Values above 70 indicate overbought, below 30 indicate oversold.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ULTOSC(
        high, low, close,
        timeperiod1=timeperiod1,
        timeperiod2=timeperiod2,
        timeperiod3=timeperiod3
    )


def willr(high, low, close, period=14):
    """
    Williams %R (WILLR)
    
    Measures overbought and oversold levels similar to the stochastic oscillator.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods for calculation (default=14).
    
    Returns:
        np.ndarray: Williams %R values ranging from -100 to 0.
                    Values above -20 indicate overbought, below -80 indicate oversold.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.WILLR(high, low, close, timeperiod=period)


def ad(high, low, close, volume):
    """
    Chaikin Accumulation/Distribution Line (AD)
    
    Combines price and volume to measure cumulative money flow into or out of a security.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        volume (array-like): Trading volume.
    
    Returns:
        np.ndarray: Accumulation/Distribution values indicating buying or selling pressure.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.AD(high, low, close, volume)


def adosc(high, low, close, volume, fastperiod=3, slowperiod=10):
    """
    Chaikin A/D Oscillator (ADOSC)
    
    Measures the momentum of the Accumulation/Distribution Line using fast and slow EMA.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        volume (array-like): Trading volume.
        fastperiod (int, optional): Period for the fast EMA (default=3).
        slowperiod (int, optional): Period for the slow EMA (default=10).
    
    Returns:
        np.ndarray: Chaikin A/D Oscillator values.
                    Positive values indicate buying pressure, negative values indicate selling pressure.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.ADOSC(high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod)


def obv(close, volume):
    """
    On Balance Volume (OBV)
    
    Measures cumulative buying and selling pressure by adding/subtracting volume on up/down days.
    
    Parameters:
        close (array-like): Closing prices.
        volume (array-like): Trading volume.
    
    Returns:
        np.ndarray: OBV values indicating cumulative buying/selling pressure.
                    Rising OBV suggests buying pressure, falling OBV suggests selling pressure.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return talib.OBV(close, volume)

# ---------------------------
# Hilbert Transform Indicators
# ---------------------------

def ht_dcperiod(close):
    """
    Hilbert Transform - Dominant Cycle Period (HT_DCPERIOD)
    
    Estimates the dominant cycle period of a time series, useful in identifying market cycles.
    
    Parameters:
        close (array-like): Closing prices of the asset.
    
    Returns:
        np.ndarray: Array of estimated dominant cycle periods.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.HT_DCPERIOD(close)


def ht_dcphase(close):
    """
    Hilbert Transform - Dominant Cycle Phase (HT_DCPHASE)
    
    Estimates the phase of the dominant cycle, indicating position within a market cycle.
    
    Parameters:
        close (array-like): Closing prices of the asset.
    
    Returns:
        np.ndarray: Array of dominant cycle phase values in degrees (0-360).
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.HT_DCPHASE(close)


def ht_phasor(close):
    """
    Hilbert Transform - Phasor Components (HT_PHASOR)
    
    Decomposes the time series into InPhase and Quadrature components.
    
    Parameters:
        close (array-like): Closing prices of the asset.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - inphase: In-phase component of the Hilbert Transform.
            - quadrature: Quadrature component of the Hilbert Transform.
    """
    close = np.asarray(close, dtype=np.float64)
    inphase, quadrature = talib.HT_PHASOR(close)
    return inphase, quadrature


def ht_sine(close):
    """
    Hilbert Transform - SineWave (HT_SINE)
    
    Computes sine and leadsine components of the Hilbert Transform, useful for cycle detection.
    
    Parameters:
        close (array-like): Closing prices of the asset.
    
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - sine: Sinewave component.
            - leadsine: Leadsine component.
    """
    close = np.asarray(close, dtype=np.float64)
    sine, leadsine = talib.HT_SINE(close)
    return sine, leadsine


def ht_trendmode(close):
    """
    Hilbert Transform - Trend vs Cycle Mode (HT_TRENDMODE)
    
    Determines if the series is in a trending or cyclic phase.
    
    Parameters:
        close (array-like): Closing prices of the asset.
    
    Returns:
        np.ndarray: 
            1 indicates trending mode, 0 indicates cyclic mode.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.HT_TRENDMODE(close)

# ---------------------------
# Price Transform Indicators
# ---------------------------

def avgprice(open_, high, low, close):
    """
    Average Price (AVGPRICE)
    
    Computes the average price of the period using the formula:
        (Open + High + Low + Close) / 4
    
    Parameters:
        open_ (array-like): Opening prices.
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
    
    Returns:
        np.ndarray: Array of average prices.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.AVGPRICE(open_, high, low, close)


def medprice(high, low):
    """
    Median Price (MEDPRICE)
    
    Computes the median price of the period using the formula:
        (High + Low) / 2
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
    
    Returns:
        np.ndarray: Array of median prices.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return talib.MEDPRICE(high, low)


def typprice(high, low, close):
    """
    Typical Price (TYPPRICE)
    
    Computes the typical price of the period using the formula:
        (High + Low + Close) / 3
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
    
    Returns:
        np.ndarray: Array of typical prices.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.TYPPRICE(high, low, close)


def wclprice(high, low, close):
    """
    Weighted Close Price (WCLPRICE)
    
    Computes the weighted close price of the period using the formula:
        (High + Low + 2*Close) / 4
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
    
    Returns:
        np.ndarray: Array of weighted close prices.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.WCLPRICE(high, low, close)

# ---------------------------
# Volatility Indicators
# ---------------------------

def atr(high, low, close, period=14):
    """
    Average True Range (ATR)
    
    Measures market volatility by calculating the average of true ranges 
    over the specified period.
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods to use for calculation. Default is 14.
    
    Returns:
        np.ndarray: Array of ATR values.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ATR(high, low, close, timeperiod=period)


def natr(high, low, close, period=14):
    """
    Normalized Average True Range (NATR)
    
    Measures market volatility normalized by price:
        NATR = (ATR / Close) * 100
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
        period (int, optional): Number of periods to use for calculation. Default is 14.
    
    Returns:
        np.ndarray: Array of NATR values (percentage).
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.NATR(high, low, close, timeperiod=period)


def trange(high, low, close):
    """
    True Range (TRANGE)
    
    Measures the range of price movement for each period:
        True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
    
    Parameters:
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Closing prices.
    
    Returns:
        np.ndarray: Array of True Range values.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.TRANGE(high, low, close)

# ---------------------------
# Candlestick Pattern Recognition
# ---------------------------

CDL_PATTERNS = [
    "CDL2CROWS","CDL3BLACKCROWS","CDL3INSIDE","CDL3LINESTRIKE","CDL3OUTSIDE",
    "CDL3STARSINSOUTH","CDL3WHITESOLDIERS","CDLABANDONEDBABY","CDLADVANCEBLOCK","CDLBELTHOLD",
    "CDLBREAKAWAY","CDLCLOSINGMARUBOZU","CDLCONCEALBABYSWALL","CDLCOUNTERATTACK","CDLDARKCLOUDCOVER",
    "CDLDOJI","CDLDOJISTAR","CDLDRAGONFLYDOJI","CDLENGULFING","CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR","CDLGAPSIDESIDEWHITE","CDLGRAVESTONEDOJI","CDLHAMMER","CDLHANGINGMAN",
    "CDLHARAMI","CDLHARAMICROSS","CDLHIGHWAVE","CDLHIKKAKE","CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON","CDLIDENTICAL3CROWS","CDLINNECK","CDLINVERTEDHAMMER","CDLKICKING",
    "CDLKICKINGBYLENGTH","CDLLADDERBOTTOM","CDLLONGLEGGEDDOJI","CDLLONGLINE","CDLMARUBOZU",
    "CDLMATCHINGLOW","CDLMATHOLD","CDLMORNINGDOJISTAR","CDLMORNINGSTAR","CDLONNECK",
    "CDLPIERCING","CDLRICKSHAWMAN","CDLRISEFALL3METHODS","CDLSEPARATINGLINES","CDLSHOOTINGSTAR",
    "CDLSHORTLINE","CDLSPINNINGTOP","CDLSTALLEDPATTERN","CDLSTICKSANDWICH","CDLTAKURI",
    "CDLTASUKIGAP","CDLTHRUSTING","CDLTRISTAR","CDLUNIQUE3RIVER","CDLUPSIDEGAP2CROWS","CDLXSIDEGAP3METHODS"
]

def candlestick_pattern(open_, high, low, close, pattern_name):
    """
    Calculate any TA-Lib candlestick pattern by its name.

    This function dynamically calls the corresponding TA-Lib candlestick 
    pattern function using the `pattern_name` provided. Each pattern 
    identifies specific price formations that could indicate bullish or 
    bearish trends.

    Parameters:
        open_ (array-like): Open prices.
        high (array-like): High prices.
        low (array-like): Low prices.
        close (array-like): Close prices.
        pattern_name (str): Name of the candlestick pattern function to call.
                            Must be one of the following TA-Lib patterns:
                            CDL_PATTERNS list.

    Returns:
        np.ndarray: Array of integers indicating pattern detection:
                    0  : No pattern detected
                    100: Bullish pattern detected
                    -100: Bearish pattern detected

    Raises:
        ValueError: If `pattern_name` is not in CDL_PATTERNS.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if pattern_name not in CDL_PATTERNS:
        raise ValueError(f"{pattern_name} is not a valid TA-Lib candlestick pattern.")

    return getattr(talib, pattern_name)(open_, high, low, close)

# ---------------------------
# Statistical / Regression Indicators
# ---------------------------

def beta(close, ref, period=5):
    """
    Beta: Measures correlation and volatility of a series relative to another series.

    Parameters:
        close (array-like): Target price series.
        ref (array-like): Reference price series for comparison.
        period (int): Number of periods to calculate Beta over (default=5).

    Returns:
        np.ndarray: Beta values, representing how the target series moves relative
                    to the reference series.
    """
    close = np.asarray(close, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return talib.BETA(close, ref, timeperiod=period)


def correl(close, ref, period=30):
    """
    Pearson's Correlation Coefficient (r) between two series.

    Parameters:
        close (array-like): Target price series.
        ref (array-like): Reference price series.
        period (int): Number of periods for correlation calculation (default=30).

    Returns:
        np.ndarray: Correlation coefficient values ranging from -1 to 1.
    """
    close = np.asarray(close, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return talib.CORREL(close, ref, timeperiod=period)


def linearreg(close, period=14):
    """
    Linear Regression: Fits a straight line to the price series.

    Parameters:
        close (array-like): Price series.
        period (int): Number of periods to use for regression (default=14).

    Returns:
        np.ndarray: Linear regression values (predicted price on the regression line).
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.LINEARREG(close, timeperiod=period)


def linearreg_angle(close, period=14):
    """
    Linear Regression Angle: Angle of the regression line in degrees.

    Parameters:
        close (array-like): Price series.
        period (int): Number of periods to calculate the regression angle (default=14).

    Returns:
        np.ndarray: Angle of the regression line.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.LINEARREG_ANGLE(close, timeperiod=period)


def linearreg_intercept(close, period=14):
    """
    Linear Regression Intercept: Y-intercept of the regression line.

    Parameters:
        close (array-like): Price series.
        period (int): Number of periods to calculate intercept (default=14).

    Returns:
        np.ndarray: Regression line intercept values.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.LINEARREG_INTERCEPT(close, timeperiod=period)


def linearreg_slope(close, period=14):
    """
    Linear Regression Slope: Slope of the regression line.

    Parameters:
        close (array-like): Price series.
        period (int): Number of periods to calculate slope (default=14).

    Returns:
        np.ndarray: Slope values representing rate of change per period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.LINEARREG_SLOPE(close, timeperiod=period)


def stddev(close, period=14):
    """
    Standard Deviation: Measures volatility of the price series.

    Parameters:
        close (array-like): Price series.
        period (int): Number of periods for calculation (default=14).

    Returns:
        np.ndarray: Standard deviation values over the period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.STDDEV(close, timeperiod=period, nbdev=1)


def tsf(close, period=14):
    """
    Time Series Forecast: Predicts the next value using linear regression.

    Parameters:
        close (array-like): Price series.
        period (int): Number of periods used for forecast calculation (default=14).

    Returns:
        np.ndarray: Forecasted values based on linear regression.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.TSF(close, timeperiod=period)


def var(close, period=14):
    """
    Variance: Measures the statistical variance of the price series.

    Parameters:
        close (array-like): Price series.
        period (int): Number of periods to calculate variance (default=14).

    Returns:
        np.ndarray: Variance values over the period.
    """
    close = np.asarray(close, dtype=np.float64)
    return talib.VAR(close, timeperiod=period, nbdev=1)