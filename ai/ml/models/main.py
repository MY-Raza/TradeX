import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
from TradeX.utils.db.utils import fetch_ohlcv_df
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.ai.ml.models.models import train_classifier, train_regressor, get_classifier, get_regressor
from TradeX.utils.common.config_loader import get_logger, read_config
from TradeX.indicators.talib.indicators import ALL_INDICATORS
from TradeX.utils.data.data_cleaner import resample_ohlcv

logger = get_logger("model_main")

TIMEHORIZON_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440
}


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def generate_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    df = df.copy()
    for ind in indicators:
        try:
            if ind in ["RSI", "EMA", "SMA", "ATR", "ADX", "CCI", "MOM"]:
                values, window = call_indicator(ind, df["close"].values, timeperiod=14)
                df[f"{ind}_{window}"] = values

            elif ind == "MACD":
                macd, signal, hist = call_indicator("MACD", df["close"].values, fastperiod=12, slowperiod=26, signalperiod=9)[0]
                df["MACD"] = macd
                df["MACD_SIGNAL"] = signal
                df["MACD_HIST"] = hist

            elif ind == "BBANDS":
                upper, middle, lower = call_indicator("BBANDS", df["close"].values, timeperiod=20)[0]
                df["BB_UPPER"] = upper
                df["BB_MIDDLE"] = middle
                df["BB_LOWER"] = lower

        except Exception as e:
            logger.info(f"Indicator {ind} failed: {e}")

    return df


# ----------------------------
# TARGET CREATION
# ----------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df


# ----------------------------
# DATASET PREPARATION
# ----------------------------
def prepare_ml_data(df: pd.DataFrame):
    df = df.copy()
    drop_cols = ["datetime", "future_close"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna()
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def calculate_limit(timehorizon: str, days: int = 365) -> int:
    """
    Calculate the number of candles to fetch based on timehorizon and days.

    Args:
        timehorizon (str): '1m', '5m', '15m', '1h', '4h', '1d'
        days (int): Number of days you want data for (default 365)

    Returns:
        int: Number of candles
    """
    minutes_per_candle = TIMEHORIZON_TO_MINUTES.get(timehorizon.lower())
    if minutes_per_candle is None:
        raise ValueError(f"Unsupported timehorizon: {timehorizon}")

    candles_per_day = 1440 / minutes_per_candle
    total_candles = int(candles_per_day * days)
    return total_candles

def save_model(model, feature_columns, symbol, model_name, folder="saved_models"):
    """
    Save the trained model and its input features to a .pkl file
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{symbol}_{model_name}.pkl")
    
    with open(file_path, "wb") as f:
        pickle.dump({"model": model, "features": feature_columns}, f)
    
    logger.info(f"Saved model {model_name} for {symbol} at {file_path}")



# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():

    # ----------------------------
    # Load config
    # ----------------------------
    config = read_config()

    symbols = config.get("symbols", ["btc"])
    timehorizon = config.get("timehorizon", "1h")
    limit = config.get("limit", 5000)
    indicators_config = config.get("indicators", {})
    classifiers_config = config.get("classifiers", {})
    regressors_config = config.get("regressors", {})
    f_limit = calculate_limit(timehorizon="1m",days=365)

    # Only use active indicators
    active_indicators = [ind for ind, active in indicators_config.items() if active]

    logger.info(f"Config loaded | symbols={symbols} | timehorizon={timehorizon} | limit={limit}")

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol} from database...")

        df = fetch_ohlcv_df(
            table_name=f"{symbol}_1m",
            schema=f"data_binance",
            limit=f_limit
        )

        if df.empty:
            logger.info(f"No data found for {symbol}. Skipping.")
            continue

        df = resample_ohlcv(df, timehorizon)

        logger.info(f"Generating indicators for {symbol}...")
        df = generate_features(df, active_indicators)

        logger.info(f"Creating target for {symbol}...")
        df = create_target(df)

        logger.info(f"Preparing dataset for {symbol}...")
        X, y = prepare_ml_data(df)

        # ----------------------------
        # Train all active classifiers
        # ----------------------------
        for clf_name, is_active in classifiers_config.items():
            if is_active:
        # Skip MultinomialNB if features have negative values
                if clf_name.lower() == "multinomial_nb":
                    logger.info(f"Skipping {clf_name} because features contain negative values")
                    continue

                model = get_classifier(clf_name)
                if model is not None:
                    logger.info(f"Training classifier: {clf_name} for {symbol}")
                    trained_model = train_classifier(model, X, y)
                    save_model(trained_model, X.columns.tolist(), f"{symbol}_classifier", clf_name)

        # ----------------------------
        # Train all active regressors
        # ----------------------------
        for reg_name, is_active in regressors_config.items():
            if is_active:
                model = get_regressor(reg_name)
                if model:
                    logger.info(f"Training regressor: {reg_name} for {symbol}")
                    trained_model = train_regressor(model, X, y)
                    save_model(trained_model,X.columns.tolist(),f"{symbol}_regressor", reg_name)

        logger.info(f"Model training complete for {symbol}.")


# ============================================================
if __name__ == "__main__":
    main()
