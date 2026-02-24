import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from TradeX.utils.db.utils import fetch_ohlcv_df,save_df_to_db
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.ai.ml.models.model_trainer import train_model, save_model, prepare_predictions
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger 
from TradeX.utils.data.data_cleaner import resample_ohlcv
import os
from TradeX.backtest.newbacktest import HighPerfBacktest
logger = get_logger("model_main")


import os
from datetime import datetime

def save_ledger_to_csv(ledger_df: pd.DataFrame, clf_name: str, folder: str = "ledgers"):
    """
    Save ledger DataFrame as CSV with filename:
    YYYYMMDD_HHMMSS_clf_name.csv
    """

    if ledger_df.empty:
        print("Ledger is empty. Nothing saved.")
        return

    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Current datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean classifier name (optional safety)
    clf_name_clean = clf_name.replace(" ", "_")

    # File name
    filename = f"{timestamp}_{clf_name_clean}.csv"
    filepath = os.path.join(folder, filename)

    # Save CSV
    ledger_df.to_csv(filepath, index=False)

    logger.info(f"Ledger saved to: {filepath}")


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def generate_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    df = df.copy()
    for ind in indicators:
        try:
            if ind in ["RSI", "EMA", "SMA", "MOM"]:  # single-series indicators
                values, window = call_indicator(ind, df["close"].values, timeperiod=14)
                df[f"{ind}_{window}"] = values

            elif ind in ["ATR", "ADX", "CCI"]:  # require high, low, close
                values, window = call_indicator(
                    ind,
                    high=df["high"].values,
                    low=df["low"].values,
                    close=df["close"].values,
                    timeperiod=14
                )
                df[f"{ind}_{window}"] = values

            elif ind == "MACD":  # MACD needs only close
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

            elif ind == "BBANDS":  # BBANDS needs only close
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



# ----------------------------
# TARGET CREATION
# ----------------------------
def create_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df


def create_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] - df["close"]) / df["close"]
    df.dropna(inplace=True)
    return df


# ----------------------------
# DATASET PREPARATION WITH LOG-DIFF SCALING
# ----------------------------
def prepare_ml_data(df: pd.DataFrame):
    df = df.copy()
    drop_cols = ["datetime", "future_close"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.dropna()

    y = df["target"]
    X = df.drop("target", axis=1)

    # Apply log-difference scaling to positive columns
    X_scaled = X.apply(lambda col: np.log(col).diff() if np.all(col > 0) else col)
    X_scaled = X_scaled.dropna()
    y = y.loc[X_scaled.index]

    return X_scaled, y

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ml_config_path = os.path.join(current_dir, "config.yml")
    config = read_config(ml_config_path)

    start_date = config.get("start_date")
    end_date = config.get("end_date")
    split_date = config.get("split_date")

    symbols = ["btc"]
    timehorizon = config.get("timehorizon", "1h")
    indicators_config = config.get("indicators", {})
    classifiers_config = config.get("classifiers", {})
    regressors_config = config.get("regressors", {})

    # Optional hyperparameters for XGBoost models
    xgb_params_clf = config.get("xgboost_classifier_params", {})
    xgb_params_reg = config.get("xgboost_regressor_params", {})

    active_indicators = [ind for ind, active in indicators_config.items() if active]

    logger.info(f"Config loaded | symbols={symbols} | timehorizon={timehorizon}")

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol} from database...")
        df_1m = fetch_ohlcv_df(
            table_name=f"{symbol}_1m",
            schema="data_binance",
            time_column="datetime",
            start_date=start_date,
            end_date=end_date
        )

        if df_1m.empty:
            logger.info(f"No data found for {symbol}. Skipping.")
            continue

        # Resample to desired timeframe
        df_1h = resample_ohlcv(df_1m, timehorizon)

        # Feature Engineering
        logger.info(f"Generating indicators for {symbol}...")
        df_gf = generate_features(df_1h, active_indicators)

        # ----------------------------
        # Train Classifiers
        # ----------------------------
        for clf_name, is_active in classifiers_config.items():
            if not is_active:
                continue

            logger.info(f"Training classifier: {clf_name} for {symbol}")
            try:
                df_clf = create_classification_target(df_gf)
                X, y = prepare_ml_data(df_clf)

                # Pass XGBoost params dynamically
                kwargs = xgb_params_clf if clf_name.lower() == "xgboost" else {}

                model, preds, test_index = train_model(
                    model_type="classifier",
                    model_name=clf_name,
                    df=df_clf,
                    target_col="target",
                    split_date=split_date,
                    **kwargs
                )
                df_predictions = prepare_predictions(df_clf,preds,test_index,model_type="classifier")
                df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'], utc=True)
                bt = HighPerfBacktest(
                    df_1m,
                    df_predictions,
                    take_profit=3,
                    stop_loss=1
                )
                ledger, final_balance, pnl = bt.run()
                save_df_to_db(
                    df=ledger,
                    schema="models",
                    table_name=f"{clf_name}_clf_{timestamp}",
                    time_column="datetime",
                    is_timeseries=True,
                    enforce_unique_time=False,
                    use_on_conflict=False
                )
                save_ledger_to_csv(ledger,f"{clf_name}_clf")
                logger.info(f"Final Balance: {final_balance}")
                logger.info(f"Cummulative PnL: {pnl}")
                save_model(
                    model,
                    X.columns.tolist(),
                    symbol,
                    f"{clf_name}_classifier"
                )
            except Exception as e:
                logger.error(f"Classifier {clf_name} failed for {symbol}: {e}")

        # ----------------------------
        # Train Regressors
        # ----------------------------
        for reg_name, is_active in regressors_config.items():
            if not is_active:
                continue

            logger.info(f"Training regressor: {reg_name} for {symbol}")
            try:
                df_reg = create_regression_target(df_gf)
                X, y = prepare_ml_data(df_reg)

                # Pass XGBoost params dynamically
                kwargs = xgb_params_reg if reg_name.lower() == "xgboost" else {}

                model, preds, test_index = train_model(
                    model_type="regressor",
                    model_name=reg_name,
                    df=df_reg,
                    target_col="target",
                    split_date=split_date,
                    **kwargs
                )
                df_predictions = prepare_predictions(df_reg,preds,test_index,model_type="regressor")
                df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'], utc=True)
                bt = HighPerfBacktest(
                    df_1m,
                    df_predictions,
                    take_profit=3,
                    stop_loss=1
                )
                ledger, final_balance, pnl = bt.run()
                save_ledger_to_csv(ledger,f"{reg_name}_reg")
                save_df_to_db(
                    df=ledger,
                    schema="models",
                    table_name=f"{reg_name}_reg_{timestamp}",
                    time_column="datetime",
                    is_timeseries=True,
                    enforce_unique_time=False,
                    use_on_conflict=False
                )
                logger.info(f"Final Balance: {final_balance}")
                logger.info(f"Cummulative PnL: {pnl}")
                save_model(
                    model,
                    X.columns.tolist(),
                    symbol,
                    f"{reg_name}_regressor"
                )
            except Exception as e:
                logger.error(f"Regressor {reg_name} failed for {symbol}: {e}")

        logger.info(f"Model training complete for {symbol}.")


# ============================================================
if __name__ == "__main__":
    main()
