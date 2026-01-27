# signals.py
import numpy as np


def generate_signals(indicators: dict) -> np.ndarray:
    """
    Generate Buy / Sell / Hold signals using layered confirmation logic

    Signal values:
        1  -> Buy
        0  -> Hold
       -1  -> Sell
    """

    close = indicators["close"]

    # ---------------------------
    # Trend indicators
    # ---------------------------
    ema_fast = indicators["ema_fast"]
    ema_slow = indicators["ema_slow"]
    adx = indicators["adx"]
    plus_di = indicators["plus_di"]
    minus_di = indicators["minus_di"]

    # ---------------------------
    # Momentum indicators
    # ---------------------------
    rsi = indicators["rsi"]
    macd_hist = indicators["macd_hist"]
    stochrsi_k = indicators.get("stochrsi_k")
    stochrsi_d = indicators.get("stochrsi_d")

    # ---------------------------
    # Volume / Flow
    # ---------------------------
    obv = indicators["obv"]
    mfi = indicators["mfi"]

    # ---------------------------
    # Volatility
    # ---------------------------
    atr = indicators["atr"]
    atr_mean = np.nanmean(atr)

    # ---------------------------
    # Candlestick (optional)
    # ---------------------------
    cdl_engulfing = indicators.get("cdl_engulfing")

    signals = np.zeros(len(close), dtype=np.int8)

    for i in range(2, len(close)):

        # ===========================
        # 1️⃣ TREND BIAS
        # ===========================
        trend_bullish = (
            ema_fast[i] > ema_slow[i] and
            adx[i] > 20 and
            plus_di[i] > minus_di[i]
        )

        trend_bearish = (
            ema_fast[i] < ema_slow[i] and
            adx[i] > 20 and
            minus_di[i] > plus_di[i]
        )

        # ===========================
        # 2️⃣ MOMENTUM TRIGGERS
        # ===========================
        momentum_buy = (
            rsi[i] < 35 and
            macd_hist[i] > 0
        )

        momentum_sell = (
            rsi[i] > 65 and
            macd_hist[i] < 0
        )

        # Optional refinement
        if stochrsi_k is not None and stochrsi_d is not None:
            momentum_buy &= stochrsi_k[i] > stochrsi_d[i]
            momentum_sell &= stochrsi_k[i] < stochrsi_d[i]

        # ===========================
        # 3️⃣ VOLATILITY FILTER
        # ===========================
        valid_volatility = atr[i] > atr_mean

        # ===========================
        # 4️⃣ VOLUME CONFIRMATION
        # ===========================
        volume_confirm = (
            obv[i] > obv[i - 1] or
            mfi[i] > 50
        )

        # ===========================
        # 5️⃣ CANDLESTICK CONFIRMATION (OPTIONAL)
        # ===========================
        bullish_candle = True
        bearish_candle = True

        if cdl_engulfing is not None:
            bullish_candle = cdl_engulfing[i] > 0
            bearish_candle = cdl_engulfing[i] < 0

        # ===========================
        # 6️⃣ FINAL DECISION
        # ===========================
        if (
            trend_bullish and
            momentum_buy and
            volume_confirm and
            valid_volatility and
            bullish_candle
        ):
            signals[i] = 1

        elif (
            trend_bearish and
            momentum_sell and
            volume_confirm and
            valid_volatility and
            bearish_candle
        ):
            signals[i] = -1

        else:
            signals[i] = 0

    return signals
