import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from TradeX.utils.db.utils import fetch_ohlcv_df, save_df_to_db
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.ai.dl.models.model_trainer import train_model, save_model
from TradeX.ai.ml.utils import prepare_predictions, pnl_permutation_importance, extract_important_features, compute_trade_statistics
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger 
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.backtest import BackTest
from datetime import datetime
import os

logger = get_logger("dl_model_main")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------
# FEATURE ENGINEERING (same as ML)
# ----------------------------
def _arr(x) -> np.ndarray:
    """Ensure x is a 1D float64 numpy array."""
    return np.asarray(x, dtype=np.float64).ravel()


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

    close  = df["close"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    open_  = df["open"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # =========================
    # Indicator categories
    # =========================
    single_series = {
        "RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "TRIMA",
        "KAMA", "T3", "MOM", "ROC", "ROCP", "ROCR", "ROCR100",
        "LINEARREG", "LINEARREG_SLOPE", "LINEARREG_ANGLE",
        "LINEARREG_INTERCEPT", "STDDEV", "VAR", "TSF",
        "MA", "CMO"
    }

    hlc_series = {
        "ATR", "NATR", "ADX", "ADXR", "DX", "CCI",
        "PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM", "WILLR"
    }

    macd_series      = {"MACD", "MACDEXT", "PPO", "APO", "TRIX"}
    bband_series     = {"BBANDS"}
    stochastic_series = {"STOCH", "STOCHF", "STOCHRSI"}
    volume_series    = {"OBV", "AD", "ADOSC", "MFI"}
    aroon_series     = {"AROON", "AROONOSC"}
    sar_series       = {"SAR", "SAREXT"}
    price_transforms = {"AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"}
    cycle_series     = {ind for ind in indicators if ind.startswith("HT_")}
    candle_patterns  = {ind for ind in indicators if ind.startswith("CDL")}

    # Accumulate all new columns; assign once at the end via pd.concat
    new_cols = {}

    for ind in indicators:
        try:
            # =========================
            # Single-series indicators
            # =========================
            if ind in single_series:
                values, window = call_indicator(ind, close, timeperiod=14)
                new_cols[f"{ind}_{window}"] = _arr(values)

            # =========================
            # MAMA special case
            # =========================
            elif ind == "MAMA":
                (mama_raw, fama_raw), _ = call_indicator("MAMA", close, fastlimit=0.5, slowlimit=0.05)
                mama = _arr(mama_raw)
                fama = _arr(fama_raw)
                if len(mama) != len(df):
                    pad = len(df) - len(mama)
                    mama = np.concatenate([np.full(pad, np.nan), mama]) if pad > 0 else mama[-len(df):]
                    fama = np.concatenate([np.full(pad, np.nan), fama]) if pad > 0 else fama[-len(df):]
                new_cols["MAMA"] = mama
                new_cols["FAMA"] = fama

            # =========================
            # MIDPOINT / MIDPRICE / BOP / TRANGE
            # =========================
            elif ind == "MIDPOINT":
                new_cols["MIDPOINT_14"] = _arr(call_indicator("MIDPOINT", close, timeperiod=14)[0])

            elif ind == "MIDPRICE":
                new_cols["MIDPRICE_14"] = _arr(call_indicator("MIDPRICE", high, low, timeperiod=14)[0])

            elif ind == "BOP":
                new_cols["BOP"] = _arr(call_indicator("BOP", open=open_, high=high, low=low, close=close)[0])

            elif ind == "TRANGE":
                new_cols["TRANGE"] = _arr(call_indicator("TRANGE", high=high, low=low, close=close)[0])

            # =========================
            # High / Low / Close indicators
            # =========================
            elif ind in hlc_series:
                if ind in {"MINUS_DM", "PLUS_DM"}:
                    new_cols[ind] = _arr(call_indicator(ind, high=high, low=low, timeperiod=14)[0])
                else:
                    values, window = call_indicator(ind, high=high, low=low, close=close, timeperiod=14)
                    new_cols[f"{ind}_{window}"] = _arr(values)

            # =========================
            # MACD family
            # MACD/MACDEXT -> tuple of arrays (macd, signal, hist)
            # APO/PPO/TRIX  -> single 1D array
            # =========================
            elif ind in macd_series:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        new_cols[f"{ind}_{i}"] = _arr(arr)
                else:
                    new_cols[f"{ind}_0"] = _arr(result)

            # =========================
            # Bollinger Bands
            # =========================
            elif ind in bband_series:
                (upper, mid, lower), _ = call_indicator("BBANDS", close, timeperiod=20)
                new_cols["BB_UPPER"]  = _arr(upper)
                new_cols["BB_MIDDLE"] = _arr(mid)
                new_cols["BB_LOWER"]  = _arr(lower)

            # =========================
            # Stochastic family
            # =========================
            elif ind in stochastic_series:
                if ind == "STOCHRSI":
                    (slowk, slowd), _ = call_indicator(ind, close)
                else:
                    (slowk, slowd), _ = call_indicator(ind, high=high, low=low, close=close)
                new_cols[f"{ind}_K"] = _arr(slowk)
                new_cols[f"{ind}_D"] = _arr(slowd)

            # =========================
            # Volume indicators
            # =========================
            elif ind in volume_series:
                if ind == "OBV":
                    new_cols[ind] = _arr(call_indicator(ind, close, volume)[0])
                elif ind == "AD":
                    new_cols[ind] = _arr(call_indicator(ind, high=high, low=low, close=close, volume=volume)[0])
                elif ind == "ADOSC":
                    new_cols[ind] = _arr(call_indicator("ADOSC", high=high, low=low, close=close, volume=volume)[0])
                elif ind == "MFI":
                    new_cols[f"{ind}_14"] = _arr(call_indicator("MFI", high=high, low=low, close=close, volume=volume, timeperiod=14)[0])

            # =========================
            # Aroon
            # =========================
            elif ind in aroon_series:
                if ind == "AROON":
                    (aroon_up, aroon_down), _ = call_indicator(ind, high=high, low=low, timeperiod=14)
                    new_cols["AROON_UP"]   = _arr(aroon_up)
                    new_cols["AROON_DOWN"] = _arr(aroon_down)
                else:
                    new_cols["AROONOSC"] = _arr(call_indicator(ind, high=high, low=low, timeperiod=14)[0])

            # =========================
            # SAR
            # =========================
            elif ind in sar_series:
                new_cols[ind] = _arr(call_indicator(ind, high=high, low=low)[0])

            # =========================
            # Price transforms
            # =========================
            elif ind == "AVGPRICE":
                new_cols[ind] = _arr(call_indicator(ind, open=open_, high=high, low=low, close=close)[0])
            elif ind == "MEDPRICE":
                new_cols[ind] = _arr(call_indicator(ind, high=high, low=low)[0])
            elif ind in {"TYPPRICE", "WCLPRICE"}:
                new_cols[ind] = _arr(call_indicator(ind, high=high, low=low, close=close)[0])

            # =========================
            # Hilbert Transform (Cycle)
            # HT_PHASOR → (inphase, quadrature)
            # HT_SINE   → (sine, leadsine)
            # others    → single array
            # =========================
            elif ind in cycle_series:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        new_cols[f"{ind}_{i}"] = _arr(arr)
                else:
                    new_cols[ind] = _arr(result)

            # =========================
            # Candlestick patterns
            # =========================
            elif ind in candle_patterns:
                new_cols[ind] = _arr(call_indicator(ind, open=open_, high=high, low=low, close=close)[0])

            else:
                logger.warning(f"Unsupported indicator: {ind}")

        except Exception as e:
            logger.error(f"Indicator {ind} failed: {e}")

    # Single concat — avoids repeated DataFrame copy overhead
    if new_cols:
        n = len(df)
        safe = {}
        for k, v in new_cols.items():
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.shape == (n,):
                safe[k] = arr
            else:
                logger.warning(f"Skipping column {k}: expected length {n}, got {arr.shape}")
        new_df = pd.DataFrame(safe, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


# ----------------------------
# MAIN PIPELINE FOR DL
# ----------------------------
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dl_config_path = os.path.join(current_dir, "config_dl.yml")
    config = read_config(dl_config_path)

    start_date = config.get("start_date")
    end_date = config.get("end_date")
    split_date = config.get("split_date")
    symbols = config.get("symbols", ["btc"])
    timehorizon = config.get("timehorizon", "1h")
    indicators_config = config.get("indicators", {})
    dl_models_config = config.get("dl_models", {})

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
        df_tf = resample_ohlcv(df_1m, timehorizon)

        # Feature Engineering
        logger.info(f"Generating indicators for {symbol}...")
        df_gf = generate_features(df_tf, active_indicators)

        # ----------------------------
        # Train DL Models
        # ----------------------------
        for model_name, is_active in dl_models_config.items():
            if not is_active:
                continue

            logger.info(f"Training DL model: {model_name} for {symbol}")
            try:
                model, preds, test_index, df_test = train_model(
                    model_type="dl",
                    model_name=model_name,
                    df=df_gf,
                    df_1m=df_1m,
                    split_date=split_date,
                    lookback=config.get("lookback", 24),
                    epochs=config.get("epochs", 50),
                    batch_size=config.get("batch_size", 32)
                )

                # Convert preds to DataFrame if needed for BackTest
                df_predictions = prepare_predictions(df_gf, preds, test_index, model_type="dl")
                df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'], utc=True)

                # Backtest
                bt = BackTest(
                    df_1m,
                    df_predictions,
                    take_profit=3,
                    stop_loss=1
                )
                ledger, final_balance, pnl = bt.run()

                # Feature importance (DL may use permutation importance)
                pnl_importance_df = pnl_permutation_importance(
                    model=model,
                    X_test=df_test,
                    df=df_gf,
                    df_1m=df_1m,
                    base_pnl=pnl,
                    model_type="dl",
                    k=0.5,
                    n_repeats=3
                )
                pnl_importance_wide = pnl_importance_df.set_index('feature').T.drop(columns=['feature'], errors='ignore')
                pnl_importance_wide.insert(0, "pnl", pnl)

                table_name_dl = f"{model_name}_dl_{timestamp}"
                important_features_df = extract_important_features(
                    pnl_importance_wide,
                    table_name_dl
                )
                save_df_to_db(
                    df=important_features_df,
                    table_name="best_features",
                    schema="ml_features",
                    time_column=None,
                    is_timeseries=False
                )

                stats_df = compute_trade_statistics(ledger)
                stats_df.insert(0, "pnl", pnl)
                stats_df.insert(0, "model_name", table_name_dl)
                save_df_to_db(
                    df=stats_df,
                    table_name="ml_results",
                    schema="model_stats",
                    time_column=None,
                    is_timeseries=False
                )

                # Save model
                save_model(model, feature_columns=df_gf.columns.tolist(), symbol=symbol, model_name=model_name)

            except Exception as e:
                logger.error(f"DL model {model_name} failed for {symbol}: {e}")

        logger.info(f"DL model training complete for {symbol}.")

# ============================================================
if __name__ == "__main__":
    main()