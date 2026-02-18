import pandas as pd

from TradeX.utils.db.utils import fetch_ohlcv_df
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.ai.ml.models.models import train_model
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("model_main")
# ============================================================
# FEATURE ENGINEERING
# ============================================================

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
            logger.info(f"Indicator {ind} failed: {e}")

    return df


# ============================================================
# TARGET CREATION
# ============================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target:
    1 if next candle close > current close
    0 otherwise
    """
    df = df.copy()

    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df.dropna(inplace=True)

    return df


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_ml_data(df: pd.DataFrame):

    df = df.copy()

    drop_cols = ["datetime", "future_close"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.dropna()

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    logger.info("Fetching data from database...")

    df = fetch_ohlcv_df(
        table_name="btc_1m",
        schema="data_binance",
        limit=5000
    )

    if df.empty:
        logger.info("No data found.")
        return

    logger.info("Generating indicators...")
    indicators = ["RSI", "EMA", "MACD", "BBANDS", "ATR"]
    df = generate_features(df, indicators)

    logger.info("Creating target...")
    df = create_target(df)

    logger.info("Preparing dataset...")
    X, y = prepare_ml_data(df)

    logger.info("Training model...")
    model = train_model(X, y)

    logger.info("Model training complete.")


# ============================================================

if __name__ == "__main__":
    main()
