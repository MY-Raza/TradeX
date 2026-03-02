import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from TradeX.utils.db.utils import fetch_ohlcv_df,save_df_to_db
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.ai.ml.models.model_trainer import train_model, save_model
from TradeX.ai.ml.utils import prepare_predictions, pnl_permutation_importance,extract_important_features, compute_trade_statistics
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
    """
    Generate technical indicator features for a DataFrame.
    Supports price, volume, momentum, volatility, cycle, and candlestick indicators.

    Args:
        df (pd.DataFrame): Must contain columns ['open', 'high', 'low', 'close', 'volume']
        indicators (list[str]): List of indicator names to compute

    Returns:
        pd.DataFrame: Original DataFrame with new indicator columns added
    """
    df = df.copy()
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    volume = df["volume"].values

    # =========================
    # Indicator categories
    # =========================
    single_series = {
        "RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "TRIMA",
        "KAMA", "T3", "MOM", "ROC", "ROCP", "ROCR", "ROCR100",
        "LINEARREG", "LINEARREG_SLOPE", "LINEARREG_ANGLE",
        "LINEARREG_INTERCEPT", "STDDEV", "VAR", "TSF",
        "MA","CMO"
    }

    hlc_series = {
        "ATR", "NATR", "TRANGE", "ADX", "ADXR", "DX", "CCI",
        "PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM", "WILLR"
    }

    macd_series = {"MACD", "MACDEXT", "PPO", "APO", "TRIX"}
    bband_series = {"BBANDS"}
    stochastic_series = {"STOCH", "STOCHF", "STOCHRSI"}
    volume_series = {"OBV", "AD", "ADOSC", "MFI"}
    aroon_series = {"AROON", "AROONOSC"}
    sar_series = {"SAR", "SAREXT"}
    price_transforms = {"AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"}
    cycle_series = {ind for ind in indicators if ind.startswith("HT_")}
    candle_patterns = {ind for ind in indicators if ind.startswith("CDL")}

    for ind in indicators:
        try:
            # =========================
            # Single-series indicators
            # =========================
            if ind in single_series:
                values, window = call_indicator(ind, close, timeperiod=14)
                df[f"{ind}_{window}"] = values

            # =========================
            # MAMA special case
            # =========================
            elif ind == "MAMA":
                close_arr = np.asarray(close, dtype=np.float64)
                out = call_indicator("MAMA", close, fastlimit=0.5, slowlimit=0.05)
                mama = np.ravel(out[0])  # flatten to 1D
                fama = np.ravel(out[1])  # flatten to 1D

                # Pad with NaN if needed
                if len(mama) != len(df):
                    if len(mama) > len(df):
                        mama =  mama[-len(df):]
                    else:
                        pad_len = len(df) - len(mama)
                        mama = np.concatenate([np.full(pad_len, np.nan), mama])
                        fama = np.concatenate([np.full(pad_len, np.nan), fama])

                df["MAMA"] = mama
                df["FAMA"] = fama

            # =========================
            # MIDPOINT / MIDPRICE
            # =========================
            elif ind == "MIDPOINT":
                df["MIDPOINT_14"] = call_indicator("MIDPOINT", close, timeperiod=14)[0]
            elif ind == "MIDPRICE":
                midprice = call_indicator("MIDPRICE", high, low, timeperiod=14)[0]
                df["MIDPRICE_14"] = midprice
            elif ind == "BOP":
                df["BOP"] = call_indicator(
                                    "BOP",
                                    open=open_,
                                    high=high,
                                    low=low,
                                    close=close
                                )[0]

            # =========================
            # High / Low / Close indicators
            # =========================
            elif ind in hlc_series:
                if ind == "MINUS_DM":
                    df[ind] = call_indicator("MINUS_DM", high=high, low=low, timeperiod=14)[0]
                elif ind == "PLUS_DM":
                    df[ind] = call_indicator("PLUS_DM", high=high, low=low, timeperiod=14)[0]
                else:
                    values, window = call_indicator(ind, high=high, low=low, close=close, timeperiod=14)
                    df[f"{ind}_{window}"] = values

            # =========================
            # MACD family
            # =========================
            elif ind in macd_series:
                out = call_indicator(ind, close)
                for i, arr in enumerate(out[0]):
                    df[f"{ind}_{i}"] = arr

            # =========================
            # Bollinger Bands
            # =========================
            elif ind in bband_series:
                upper, mid, lower = call_indicator("BBANDS", close, timeperiod=20)[0]
                df["BB_UPPER"], df["BB_MIDDLE"], df["BB_LOWER"] = upper, mid, lower

            # =========================
            # Stochastic family
            # =========================
            elif ind in stochastic_series:
                if ind == "STOCHRSI":
                    slowk, slowd = call_indicator(ind, close)[0]
                else:
                    slowk, slowd = call_indicator(ind, high=high, low=low, close=close)[0]

                df[f"{ind}_K"], df[f"{ind}_D"] = slowk, slowd

            # =========================
            # Volume indicators
            # =========================
            elif ind in volume_series:
                if ind in {"OBV", "AD"}:
                    df[ind] = call_indicator(ind, close, volume)[0]
                elif ind == "ADOSC":
                    df[ind] = call_indicator("ADOSC", high=high, low=low, close=close, volume=volume)[0]
                elif ind == "MFI":
                    df[f"{ind}_14"] = call_indicator("MFI", high=high, low=low, close=close, volume=volume, timeperiod=14)[0]

            # =========================
            # Aroon
            # =========================
            elif ind in aroon_series:
                out = call_indicator(ind, high=high, low=low, timeperiod=14)[0]
                if ind == "AROON":
                    df["AROON_UP"], df["AROON_DOWN"] = out
                else:
                    df["AROONOSC"] = out

            # =========================
            # SAR
            # =========================
            elif ind in sar_series:
                df[ind] = call_indicator(ind, high=high, low=low)[0]

            # =========================
            # Price transforms
            # =========================
            elif ind in price_transforms:
                df[ind] = call_indicator(ind, open=open_, high=high, low=low, close=close)[0]

            # =========================
            # Hilbert Transform (Cycle)
            # =========================
            elif ind in cycle_series:
                df[ind] = call_indicator(ind, close)[0]

            # =========================
            # Candlestick patterns
            # =========================
            elif ind in candle_patterns:
                df[ind] = call_indicator(ind, open=open_, high=high, low=low, close=close)[0]

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
    df["target"] = ((df["future_close"] - df["close"]) / df["close"]) * 1000
    df.dropna(inplace=True)
    return df

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

                # Pass XGBoost params dynamically
                kwargs = xgb_params_clf if clf_name.lower() == "xgboost" else {}

                model, preds, test_index, X_test = train_model(
                    model_type="classifier",
                    model_name=clf_name,
                    df=df_clf,
                    target_col="target",
                    split_date=split_date,
                    n_trails=2,
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
                pnl_importance_wide = pnl_importance_df.set_index('feature').T.drop(columns=['feature'], errors='ignore')
                pnl_importance_wide.insert(0, "pnl", pnl)
                table_name_clf = f"{clf_name}_clf_{timestamp}"
                important_features_df_clf = extract_important_features(
                                        pnl_importance_wide,
                                        table_name_clf
                                        )
                table_name_clf = f"{clf_name}_clf_{timestamp}"
                save_df_to_db(
                    df=important_features_df_clf,
                    table_name="best_features",
                    schema= "ml_features",
                    time_column= None,
                    is_timeseries=False
                )
                stats_df = compute_trade_statistics(ledger)
                stats_df.insert(0, "pnl", pnl)
                stats_df.insert(0,"model_name",table_name_clf)
                save_df_to_db(
                    df=stats_df,
                    table_name="ml_results",
                    schema= "model_stats",
                    time_column= None,
                    is_timeseries=False
                )
                save_model(
                    model,
                    X_test.columns.tolist(),
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

                # Pass XGBoost params dynamically
                kwargs = xgb_params_reg if reg_name.lower() == "xgboost" else {}

                model, preds, test_index, X_test = train_model(
                    model_type="regressor",
                    model_name=reg_name,
                    df=df_reg,
                    df_1m=df_1m,
                    target_col="target",
                    split_date=split_date,
                    n_trails=2
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
                pnl_importance_wide = pnl_importance_df.set_index('feature').T.drop(columns=['feature'], errors='ignore')
                pnl_importance_wide.insert(0, "pnl", pnl)
                table_name_reg = f"{reg_name}_reg_{timestamp}"
                important_features_df_reg = extract_important_features(
                                        pnl_importance_wide,
                                        table_name_reg
                                        )  
                save_df_to_db(
                    df=important_features_df_reg,
                    table_name="best_features",
                    schema= "ml_features",
                    time_column= None,
                    is_timeseries=False
                )
                stats_df = compute_trade_statistics(ledger)
                stats_df.insert(0, "pnl", pnl)
                stats_df.insert(0,"model_name",table_name_reg)
                save_df_to_db(
                    df=stats_df,
                    table_name="ml_results",
                    schema= "model_stats",
                    time_column= None,
                    is_timeseries=False
                )
                save_model(
                    model,
                    X_test.columns.tolist(),
                    symbol,
                    f"{reg_name}_regressor_{timestamp}"
                )
            except Exception as e:
                logger.error(f"Regressor {reg_name} failed for {symbol}: {e}")

        logger.info(f"Model training complete for {symbol}.")


# ============================================================
if __name__ == "__main__":
    main()
