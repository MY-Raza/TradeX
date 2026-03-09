import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from TradeX.utils.db.utils import fetch_ohlcv_df, save_df_to_db
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.ai.dl.models.model_trainer import train_model, save_model
from TradeX.ai.ml.utils import (
    prepare_predictions,
    pnl_permutation_importance,
    extract_important_features,
    compute_trade_statistics,
)
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.backtest import BackTest
from datetime import datetime
import os

logger    = get_logger("dl_model_main")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr(x) -> np.ndarray:
    """Return x as a flat float64 numpy array."""
    return np.asarray(x, dtype=np.float64).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

# Indicator category sets — defined once at module level so they are not
# rebuilt on every call to generate_features (was a hidden O(n) cost).
_SINGLE_SERIES = frozenset({
    "RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "TRIMA",
    "KAMA", "T3", "MOM", "ROC", "ROCP", "ROCR", "ROCR100",
    "LINEARREG", "LINEARREG_SLOPE", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "STDDEV", "VAR", "TSF",
    "MA", "CMO",
})
_HLC_SERIES       = frozenset({"ATR", "NATR", "ADX", "ADXR", "DX", "CCI",
                                "PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM", "WILLR"})
_MACD_SERIES      = frozenset({"MACD", "MACDEXT", "PPO", "APO", "TRIX"})
_BBAND_SERIES     = frozenset({"BBANDS"})
_STOCH_SERIES     = frozenset({"STOCH", "STOCHF", "STOCHRSI"})
_VOLUME_SERIES    = frozenset({"OBV", "AD", "ADOSC", "MFI"})
_AROON_SERIES     = frozenset({"AROON", "AROONOSC"})
_SAR_SERIES       = frozenset({"SAR", "SAREXT"})
_PRICE_TRANSFORMS = frozenset({"AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"})


def generate_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    Generate technical-indicator features and append them to *df*.

    Optimisations vs original:
    - Category sets are now module-level frozensets (computed once).
    - `cycle_series` and `candle_patterns` are computed once per call with
      set-comprehensions instead of per-indicator checks in the loop.
    - All new columns are collected in a plain dict and concatenated with a
      single `pd.concat` at the end (the original already did this, preserved).
    - Avoids rebuilding numpy raw arrays from df columns inside the loop;
      they are extracted once before the loop.

    Bug-fixes:
    - MAMA length-padding logic had an off-by-one: `pad > 0` check was
      correct but `else` branch used `[-len(df):]` which clips correctly only
      when len(mama) > len(df); the symmetric case was fine, but the comment
      was misleading.  No behavioural change; code is now clearer.
    - `new_cols` keys could theoretically collide (e.g. two indicators both
      writing "ATR_14"); last-writer wins silently.  A warning is now emitted
      on collision.
    """
    n = len(df)

    # Extract raw arrays once — avoids repeated .values + astype in the loop
    close  = df["close"].to_numpy(dtype=np.float64)
    high   = df["high"].to_numpy(dtype=np.float64)
    low    = df["low"].to_numpy(dtype=np.float64)
    open_  = df["open"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    # Compute per-call dynamic sets once
    indicator_set  = set(indicators)
    cycle_series   = frozenset(ind for ind in indicator_set if ind.startswith("HT_"))
    candle_patterns = frozenset(ind for ind in indicator_set if ind.startswith("CDL"))

    new_cols: dict[str, np.ndarray] = {}

    def _store(key: str, arr):
        """Store a computed array, warning on key collision."""
        if key in new_cols:
            logger.warning(f"Indicator key collision: '{key}' will be overwritten.")
        new_cols[key] = _arr(arr)

    for ind in indicators:
        try:
            # ── Single-series ──────────────────────────────────────────────
            if ind in _SINGLE_SERIES:
                values, window = call_indicator(ind, close, timeperiod=14)
                _store(f"{ind}_{window}", values)

            # ── MAMA ───────────────────────────────────────────────────────
            elif ind == "MAMA":
                (mama_raw, fama_raw), _ = call_indicator(
                    "MAMA", close, fastlimit=0.5, slowlimit=0.05
                )
                mama = _arr(mama_raw)
                fama = _arr(fama_raw)
                if len(mama) != n:
                    pad  = n - len(mama)
                    if pad > 0:
                        mama = np.concatenate([np.full(pad, np.nan), mama])
                        fama = np.concatenate([np.full(pad, np.nan), fama])
                    else:
                        mama = mama[-n:]
                        fama = fama[-n:]
                _store("MAMA", mama)
                _store("FAMA", fama)

            # ── MIDPOINT / MIDPRICE / BOP / TRANGE ────────────────────────
            elif ind == "MIDPOINT":
                _store("MIDPOINT_14", call_indicator("MIDPOINT", close, timeperiod=14)[0])

            elif ind == "MIDPRICE":
                _store("MIDPRICE_14", call_indicator("MIDPRICE", high, low, timeperiod=14)[0])

            elif ind == "BOP":
                _store("BOP", call_indicator("BOP", open=open_, high=high, low=low, close=close)[0])

            elif ind == "TRANGE":
                _store("TRANGE", call_indicator("TRANGE", high=high, low=low, close=close)[0])

            # ── High/Low/Close ─────────────────────────────────────────────
            elif ind in _HLC_SERIES:
                if ind in {"MINUS_DM", "PLUS_DM"}:
                    _store(ind, call_indicator(ind, high=high, low=low, timeperiod=14)[0])
                else:
                    values, window = call_indicator(ind, high=high, low=low, close=close, timeperiod=14)
                    _store(f"{ind}_{window}", values)

            # ── MACD family ────────────────────────────────────────────────
            elif ind in _MACD_SERIES:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        _store(f"{ind}_{i}", arr)
                else:
                    _store(f"{ind}_0", result)

            # ── Bollinger Bands ────────────────────────────────────────────
            elif ind in _BBAND_SERIES:
                (upper, mid, lower), _ = call_indicator("BBANDS", close, timeperiod=20)
                _store("BB_UPPER",  upper)
                _store("BB_MIDDLE", mid)
                _store("BB_LOWER",  lower)

            # ── Stochastic ─────────────────────────────────────────────────
            elif ind in _STOCH_SERIES:
                if ind == "STOCHRSI":
                    (slowk, slowd), _ = call_indicator(ind, close)
                else:
                    (slowk, slowd), _ = call_indicator(ind, high=high, low=low, close=close)
                _store(f"{ind}_K", slowk)
                _store(f"{ind}_D", slowd)

            # ── Volume ─────────────────────────────────────────────────────
            elif ind in _VOLUME_SERIES:
                if ind == "OBV":
                    _store(ind, call_indicator(ind, close, volume)[0])
                elif ind == "AD":
                    _store(ind, call_indicator(ind, high=high, low=low, close=close, volume=volume)[0])
                elif ind == "ADOSC":
                    _store(ind, call_indicator("ADOSC", high=high, low=low, close=close, volume=volume)[0])
                elif ind == "MFI":
                    _store(f"{ind}_14", call_indicator("MFI", high=high, low=low, close=close, volume=volume, timeperiod=14)[0])

            # ── Aroon ──────────────────────────────────────────────────────
            elif ind in _AROON_SERIES:
                if ind == "AROON":
                    (aroon_up, aroon_down), _ = call_indicator(ind, high=high, low=low, timeperiod=14)
                    _store("AROON_UP",   aroon_up)
                    _store("AROON_DOWN", aroon_down)
                else:
                    _store("AROONOSC", call_indicator(ind, high=high, low=low, timeperiod=14)[0])

            # ── SAR ────────────────────────────────────────────────────────
            elif ind in _SAR_SERIES:
                _store(ind, call_indicator(ind, high=high, low=low)[0])

            # ── Price transforms ───────────────────────────────────────────
            elif ind == "AVGPRICE":
                _store(ind, call_indicator(ind, open=open_, high=high, low=low, close=close)[0])
            elif ind == "MEDPRICE":
                _store(ind, call_indicator(ind, high=high, low=low)[0])
            elif ind in {"TYPPRICE", "WCLPRICE"}:
                _store(ind, call_indicator(ind, high=high, low=low, close=close)[0])

            # ── Hilbert Transform (Cycle) ──────────────────────────────────
            elif ind in cycle_series:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        _store(f"{ind}_{i}", arr)
                else:
                    _store(ind, result)

            # ── Candlestick patterns ───────────────────────────────────────
            elif ind in candle_patterns:
                _store(ind, call_indicator(ind, open=open_, high=high, low=low, close=close)[0])

            else:
                logger.warning(f"Unsupported indicator: {ind}")

        except Exception as e:
            logger.error(f"Indicator '{ind}' failed: {e}")

    # ── Single concat — avoids repeated DataFrame copy overhead ───────────
    if new_cols:
        safe = {}
        for k, v in new_cols.items():
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.shape == (n,):
                safe[k] = arr
            else:
                logger.warning(f"Skipping column '{k}': expected length {n}, got {arr.shape}")
        if safe:
            df = pd.concat([df, pd.DataFrame(safe, index=df.index)], axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    current_dir    = os.path.dirname(os.path.abspath(__file__))
    dl_config_path = os.path.join(current_dir, "config.yml")
    config         = read_config(dl_config_path)

    start_date        = config.get("start_date")
    end_date          = config.get("end_date")
    split_date        = config.get("split_date")
    symbols           = ["btc"]
    timehorizon       = config.get("timehorizon", "1h")
    indicators_config = config.get("indicators", {})
    dl_models_config  = config.get("forecasting_models", {})

    active_indicators = [ind for ind, active in indicators_config.items() if active]

    logger.info(f"Config loaded | symbols={symbols} | timehorizon={timehorizon}")

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol} …")
        df_1m = fetch_ohlcv_df(
            table_name=f"{symbol}_1m",
            schema="data_binance",
            time_column="datetime",
            start_date=start_date,
            end_date=end_date,
        )

        if df_1m.empty:
            logger.warning(f"No data found for {symbol}. Skipping.")
            continue

        # Resample to desired timeframe
        df_tf = resample_ohlcv(df_1m, timehorizon)

        # Feature Engineering
        logger.info(f"Generating indicators for {symbol} …")
        df_gf = generate_features(df_tf, active_indicators)

        # ── Train DL Models ───────────────────────────────────────────────
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
                    batch_size=config.get("batch_size", 32),
                )

                df_predictions = prepare_predictions(df_gf, preds, test_index, model_type="dl")
                df_predictions["datetime"] = pd.to_datetime(df_predictions["datetime"], utc=True)

                # Backtest
                bt = BackTest(df_1m, df_predictions, take_profit=3, stop_loss=1)
                ledger, final_balance, pnl = bt.run()

                # Feature importance
                pnl_importance_df = pnl_permutation_importance(
                    model=model,
                    X_test=df_test,
                    df=df_gf,
                    df_1m=df_1m,
                    base_pnl=pnl,
                    model_type="dl",
                    k=0.5,
                    n_repeats=3,
                )
                pnl_importance_wide = (
                    pnl_importance_df
                    .set_index("feature")
                    .T
                    .drop(columns=["feature"], errors="ignore")
                )
                pnl_importance_wide.insert(0, "pnl", pnl)

                table_name_dl = f"{model_name}_dl_{timestamp}"
                important_features_df = extract_important_features(pnl_importance_wide, table_name_dl)
                save_df_to_db(
                    df=important_features_df,
                    table_name="best_features",
                    schema="ml_features",
                    time_column=None,
                    is_timeseries=False,
                )

                stats_df = compute_trade_statistics(ledger)
                stats_df.insert(0, "pnl", pnl)
                stats_df.insert(0, "model_name", table_name_dl)
                save_df_to_db(
                    df=stats_df,
                    table_name="ml_results",
                    schema="model_stats",
                    time_column=None,
                    is_timeseries=False,
                )

                # BUG-FIX: save_model now re-raises on failure; wrap it so
                # a failed save does not abort the entire symbol loop.
                try:
                    save_model(
                        model,
                        feature_columns=df_gf.columns.tolist(),
                        symbol=symbol,
                        model_name=model_name,
                    )
                except Exception as save_err:
                    logger.error(f"Model save failed for {model_name}/{symbol}: {save_err}")

            except Exception as e:
                logger.error(f"DL model {model_name} failed for {symbol}: {e}")

        logger.info(f"DL model training complete for {symbol}.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()