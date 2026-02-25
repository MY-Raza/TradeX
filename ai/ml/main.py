import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from TradeX.utils.db.utils import fetch_ohlcv_df,save_df_to_db
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.ai.ml.models.model_trainer import train_model, save_model
from TradeX.ai.ml.utils import prepare_predictions, pnl_permutation_importance
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger 
from TradeX.utils.data.data_cleaner import resample_ohlcv
import os
from TradeX.backtest.backtest import BackTest
logger = get_logger("model_main")
import os
from datetime import datetime
# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def generate_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    volume = df["volume"].values

    for ind in indicators:
        try:
            # =========================
            # Single-series indicators
            # =========================
            if ind in {
                "RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "TRIMA",
                "KAMA", "T3", "MOM", "ROC", "ROCP", "ROCR", "ROCR100",
                "LINEARREG", "LINEARREG_SLOPE", "LINEARREG_ANGLE",
                "LINEARREG_INTERCEPT", "STDDEV", "VAR", "TSF"
            }:
                values, window = call_indicator(ind, close, timeperiod=14)
                df[f"{ind}_{window}"] = values

            # =========================
            # High / Low / Close
            # =========================
            elif ind in {
                "ATR", "NATR", "TRANGE",
                "ADX", "ADXR", "DX",
                "CCI",
                "PLUS_DI", "MINUS_DI",
                "PLUS_DM", "MINUS_DM",
                "WILLR"
            }:
                values, window = call_indicator(
                    ind,
                    high=high,
                    low=low,
                    close=close,
                    timeperiod=14
                )
                df[f"{ind}_{window}"] = values

            # =========================
            # MACD family
            # =========================
            elif ind in {"MACD", "MACDEXT", "PPO", "APO", "TRIX"}:
                out = call_indicator(ind, close)
                for i, arr in enumerate(out[0]):
                    df[f"{ind}_{i}"] = arr

            # =========================
            # Bollinger Bands
            # =========================
            elif ind == "BBANDS":
                upper, mid, lower = call_indicator("BBANDS", close, timeperiod=20)[0]
                df["BB_UPPER"] = upper
                df["BB_MIDDLE"] = mid
                df["BB_LOWER"] = lower

            # =========================
            # Stochastic family
            # =========================
            elif ind in {"STOCH", "STOCHF", "STOCHRSI"}:
                slowk, slowd = call_indicator(
                    ind,
                    high=high,
                    low=low,
                    close=close
                )[0]
                df[f"{ind}_K"] = slowk
                df[f"{ind}_D"] = slowd

            # =========================
            # Volume indicators
            # =========================
            elif ind in {"OBV", "AD"}:
                df[ind] = call_indicator(ind, close, volume)[0]

            elif ind == "ADOSC":
                df["ADOSC"] = call_indicator(
                    "ADOSC",
                    high=high,
                    low=low,
                    close=close,
                    volume=volume
                )[0]

            elif ind == "MFI":
                df["MFI_14"] = call_indicator(
                    "MFI",
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    timeperiod=14
                )[0]

            # =========================
            # Aroon
            # =========================
            elif ind in {"AROON", "AROONOSC"}:
                out = call_indicator(
                    ind,
                    high=high,
                    low=low,
                    timeperiod=14
                )[0]
                if ind == "AROON":
                    df["AROON_UP"], df["AROON_DOWN"] = out
                else:
                    df["AROONOSC"] = out

            # =========================
            # SAR
            # =========================
            elif ind in {"SAR", "SAREXT"}:
                df[ind] = call_indicator(
                    ind,
                    high=high,
                    low=low
                )[0]

            # =========================
            # Price transforms
            # =========================
            elif ind in {"AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"}:
                df[ind] = call_indicator(
                    ind,
                    open=open_,
                    high=high,
                    low=low,
                    close=close
                )[0]

            # =========================
            # Hilbert Transform (Cycle)
            # =========================
            elif ind.startswith("HT_"):
                df[ind] = call_indicator(ind, close)[0]

            # =========================
            # Candlestick patterns
            # =========================
            elif ind.startswith("CDL"):
                df[ind] = call_indicator(
                    ind,
                    open=open_,
                    high=high,
                    low=low,
                    close=close
                )[0]

            else:
                logger.warning(f"Unsupported indicator: {ind}")

        except Exception as e:
            logger.error(f"Indicator {ind} failed: {e}")

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

                model, preds, test_index, X_test = train_model(
                    model_type="classifier",
                    model_name=clf_name,
                    df=df_clf,
                    target_col="target",
                    split_date=split_date,
                    n_trails=10,
                    df_1m=df_1m
                )
                df_predictions = prepare_predictions(df_clf,preds,test_index,model_type="classifier")
                df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'], utc=True)
                bt = BackTest(
                    df_1m,
                    df_predictions,
                    take_profit=3,
                    stop_loss=1
                )
                ledger, final_balance, pnl = bt.run()
                pnl_importance_df = pnl_permutation_importance(
                    model=model,
                    X_test=X_test,
                    df=df_clf,
                    df_1m=df_1m,
                    base_pnl=pnl,
                    model_type="classifier",
                    k=0.5,
                    n_repeats=3
                )
                print(pnl_importance_df.head())
                save_df_to_db(
                    df=ledger,
                    schema="models",
                    table_name=f"{clf_name}_clf_{timestamp}",
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
                    f"{clf_name}_classifier_{timestamp}"
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

                model, preds, test_index, X_test = train_model(
                    model_type="regressor",
                    model_name=reg_name,
                    df=df_reg,
                    df_1m=df_1m,
                    target_col="target",
                    split_date=split_date,
                    n_trails=10
                )
                df_predictions = prepare_predictions(df_reg,preds,test_index,model_type="regressor")
                df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'], utc=True)
                bt = BackTest(
                    df_1m,
                    df_predictions,
                    take_profit=3,
                    stop_loss=1
                )
                ledger, final_balance, pnl = bt.run()
                pnl_importance_df = pnl_permutation_importance(
                    model=model,
                    X_test=X_test,
                    df=df_reg,
                    df_1m=df_1m,
                    base_pnl=pnl,
                    model_type="regressor",
                    k=0.5,
                    n_repeats=3
                )
                print(pnl_importance_df.head())
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
                    f"{reg_name}_regressor_{timestamp}"
                )
            except Exception as e:
                logger.error(f"Regressor {reg_name} failed for {symbol}: {e}")

        logger.info(f"Model training complete for {symbol}.")


# ============================================================
if __name__ == "__main__":
    main()
