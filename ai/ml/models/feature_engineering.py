import pandas as pd
from TradeX.indicators.talib.indicators import ALL_INDICATORS,call_indicator


def generate_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    Generate TA-Lib indicators and append them to the dataframe.
    """

    df = df.copy()

    for ind in indicators:
        try:
            if ind in ["RSI", "EMA", "SMA", "ATR", "ADX", "CCI", "MOM"]:
                values, window = call_indicator(
                    ind,
                    df["close"].values,
                    timeperiod=14
                )
                df[f"{ind}_{window}"] = values

            elif ind == "MACD":
                macd, signal, hist = call_indicator(
                    "MACD",
                    df["close"].values,
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9
                )[0]

                df["MACD"] = macd
                df["MACD_SIGNAL"] = signal
                df["MACD_HIST"] = hist

            elif ind == "BBANDS":
                upper, middle, lower = call_indicator(
                    "BBANDS",
                    df["close"].values,
                    timeperiod=20
                )[0]

                df["BB_UPPER"] = upper
                df["BB_MIDDLE"] = middle
                df["BB_LOWER"] = lower

        except Exception as e:
            print(f"Indicator {ind} failed: {e}")

    return df
