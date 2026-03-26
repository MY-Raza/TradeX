from __future__ import annotations

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from TradeX.ai.dl.models.model_trainer import train_model, save_model
from TradeX.ai.dl.optuna_tuner import tune_all_models
from TradeX.ai.ml.utils import (
    compute_trade_statistics,
    extract_important_features,
    pnl_permutation_importance,
    prepare_predictions,
)
from TradeX.backtest.backtest import BackTest
from TradeX.indicators.talib.indicators import call_indicator
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.db.utils import fetch_ohlcv_df, save_df_to_db

logger = get_logger("dl_model_main")

# Raw OHLCV + target columns that should NOT be passed as "features" to save_model.
_NON_FEATURE_COLS = frozenset({
    "open", "high", "low", "close", "volume",
    "datetime", "log_return", "timestamp",
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr(x) -> np.ndarray:
    """Return *x* as a flat float64 numpy array."""
    return np.asarray(x, dtype=np.float64).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

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

# ----------------------------
# SPLIT DATE RESOLVER
# ----------------------------
def resolve_split_date(df: pd.DataFrame, config: dict) -> str:
    """
    Determine the train/test split date from config.

    Priority:
      1. train_ratio (float 0-1) — computes the split from the data itself
         so the boundary always falls at exactly that percentage of rows.
         The resolved date is logged for reproducibility.
      2. split_date (str 'YYYY-MM-DD') — hard calendar boundary, used
         only when train_ratio is null / absent.

    Args:
        df     : The feature DataFrame that will be split (must have a
                 'datetime' column or a DatetimeIndex).
        config : Loaded config dict.

    Returns:
        ISO date string 'YYYY-MM-DD HH:MM:SS' ready to pass to train_model.

    Raises:
        ValueError : If neither key is present in config.
        ValueError : If train_ratio is not in (0, 1).
    """
    train_ratio = config.get("train_ratio")

    if train_ratio is not None:
        train_ratio = float(train_ratio)
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(
                f"train_ratio must be between 0 and 1 (exclusive), got {train_ratio}"
            )

        # Resolve the datetime series regardless of whether it is a column
        # or the index so the helper works with both layouts.
        if "datetime" in df.columns:
            dt_series = pd.to_datetime(df["datetime"], utc=True).sort_values()
        elif isinstance(df.index, pd.DatetimeIndex):
            dt_series = df.index.sort_values().to_series()
        else:
            raise ValueError(
                "resolve_split_date: DataFrame must have a 'datetime' column "
                "or a DatetimeIndex to use train_ratio."
            )

        n_train    = int(len(dt_series) * train_ratio)
        split_ts   = dt_series.iloc[n_train]
        split_date = split_ts.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            f"train_ratio={train_ratio} → split at row {n_train}/{len(dt_series)} "
            f"→ split_date='{split_date}' "
            f"({train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% test)"
        )
        return split_date

    # Fallback: hard calendar split_date from config
    split_date = config.get("split_date")
    if split_date is None:
        raise ValueError(
            "Config must contain either 'train_ratio' or 'split_date'."
        )
    logger.info(f"Using fixed split_date='{split_date}' from config.")
    return str(split_date)

def generate_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    Compute technical indicators and append them as new columns.

    All indicator arrays are collected into a single dict and concatenated
    once at the end to avoid repeated DataFrame copies.
    """
    n = len(df)

    close  = df["close"].to_numpy(dtype=np.float64)
    high   = df["high"].to_numpy(dtype=np.float64)
    low    = df["low"].to_numpy(dtype=np.float64)
    open_  = df["open"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    indicator_set   = set(indicators)
    cycle_series    = frozenset(i for i in indicator_set if i.startswith("HT_"))
    candle_patterns = frozenset(i for i in indicator_set if i.startswith("CDL"))

    new_cols: dict[str, np.ndarray] = {}

    def _store(key: str, arr):
        if key in new_cols:
            logger.warning(f"Indicator key collision: '{key}' will be overwritten.")
        new_cols[key] = _arr(arr)

    for ind in indicators:
        try:
            if ind in _SINGLE_SERIES:
                values, window = call_indicator(ind, close, timeperiod=14)
                _store(f"{ind}_{window}", values)

            elif ind == "MAMA":
                (mama_raw, fama_raw), _ = call_indicator(
                    "MAMA", close, fastlimit=0.5, slowlimit=0.05
                )
                mama = _arr(mama_raw)
                fama = _arr(fama_raw)
                if len(mama) != n:
                    pad = n - len(mama)
                    if pad > 0:
                        mama = np.concatenate([np.full(pad, np.nan), mama])
                        fama = np.concatenate([np.full(pad, np.nan), fama])
                    else:
                        mama = mama[-n:]
                        fama = fama[-n:]
                _store("MAMA", mama)
                _store("FAMA", fama)

            elif ind == "MIDPOINT":
                _store("MIDPOINT_14", call_indicator("MIDPOINT", close, timeperiod=14)[0])

            elif ind == "MIDPRICE":
                _store("MIDPRICE_14", call_indicator("MIDPRICE", high, low, timeperiod=14)[0])

            elif ind == "BOP":
                _store("BOP", call_indicator("BOP", open=open_, high=high, low=low, close=close)[0])

            elif ind == "TRANGE":
                _store("TRANGE", call_indicator("TRANGE", high=high, low=low, close=close)[0])

            elif ind in _HLC_SERIES:
                if ind in {"MINUS_DM", "PLUS_DM"}:
                    _store(ind, call_indicator(ind, high=high, low=low, timeperiod=14)[0])
                else:
                    values, window = call_indicator(ind, high=high, low=low, close=close, timeperiod=14)
                    _store(f"{ind}_{window}", values)

            elif ind in _MACD_SERIES:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        _store(f"{ind}_{i}", arr)
                else:
                    _store(f"{ind}_0", result)

            elif ind in _BBAND_SERIES:
                (upper, mid, lower), _ = call_indicator("BBANDS", close, timeperiod=20)
                _store("BB_UPPER",  upper)
                _store("BB_MIDDLE", mid)
                _store("BB_LOWER",  lower)

            elif ind in _STOCH_SERIES:
                if ind == "STOCHRSI":
                    (slowk, slowd), _ = call_indicator(ind, close)
                else:
                    (slowk, slowd), _ = call_indicator(ind, high=high, low=low, close=close)
                _store(f"{ind}_K", slowk)
                _store(f"{ind}_D", slowd)

            elif ind in _VOLUME_SERIES:
                if ind == "OBV":
                    _store(ind, call_indicator(ind, close, volume)[0])
                elif ind == "AD":
                    _store(ind, call_indicator(ind, high=high, low=low, close=close, volume=volume)[0])
                elif ind == "ADOSC":
                    _store(ind, call_indicator("ADOSC", high=high, low=low, close=close, volume=volume)[0])
                elif ind == "MFI":
                    _store(f"{ind}_14", call_indicator("MFI", high=high, low=low, close=close, volume=volume, timeperiod=14)[0])

            elif ind in _AROON_SERIES:
                if ind == "AROON":
                    (aroon_up, aroon_down), _ = call_indicator(ind, high=high, low=low, timeperiod=14)
                    _store("AROON_UP",   aroon_up)
                    _store("AROON_DOWN", aroon_down)
                else:
                    _store("AROONOSC", call_indicator(ind, high=high, low=low, timeperiod=14)[0])

            elif ind in _SAR_SERIES:
                _store(ind, call_indicator(ind, high=high, low=low)[0])

            elif ind == "AVGPRICE":
                _store(ind, call_indicator(ind, open=open_, high=high, low=low, close=close)[0])
            elif ind == "MEDPRICE":
                _store(ind, call_indicator(ind, high=high, low=low)[0])
            elif ind in {"TYPPRICE", "WCLPRICE"}:
                _store(ind, call_indicator(ind, high=high, low=low, close=close)[0])

            elif ind in cycle_series:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        _store(f"{ind}_{i}", arr)
                else:
                    _store(ind, result)

            elif ind in candle_patterns:
                _store(ind, call_indicator(ind, open=open_, high=high, low=low, close=close)[0])

            else:
                logger.warning(f"Unsupported indicator: {ind}")

        except Exception as exc:
            logger.error(f"Indicator '{ind}' failed: {exc}")

    if new_cols:
        safe = {
            k: v for k, v in (
                (k, np.asarray(v, dtype=np.float64).ravel()) for k, v in new_cols.items()
            )
            if v.shape == (n,)
        }
        skipped = set(new_cols) - set(safe)
        for k in skipped:
            logger.warning(f"Skipping column '{k}': length mismatch.")
        if safe:
            df = pd.concat([df, pd.DataFrame(safe, index=df.index)], axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    current_dir    = os.path.dirname(os.path.abspath(__file__))
    dl_config_path = os.path.join(current_dir, "config.yml")
    config         = read_config(dl_config_path)

    start_date        = config.get("start_date")
    end_date          = config.get("end_date")
    symbols           = ["btc"]  # BUG-FIX: was hardcoded to ["btc"]
    timehorizon       = config.get("timehorizon", "1h")
    indicators_config = config.get("indicators", {})
    dl_models_config  = config.get("forecasting_models", {})
    training_cfg      = config.get("training", {})

    # BUG-FIX (main): training_cfg has no "lookback"/"epochs"/"batch_size" keys
    # in config.yml — those live inside model-specific param dicts. Using
    # training_cfg.get() with wrong keys always returned the fallback values
    # (24/50/32), silently overriding whatever was in nbeats_params etc.
    # Fix: pass None so model_trainer uses the model-specific config params.
    lookback   = training_cfg.get("lookback",    None)
    epochs     = training_cfg.get("epochs",      None)
    batch_size = training_cfg.get("batch_size",  None)

    model_params_map: dict[str, dict] = {
        "arima":       config.get("arima_params",       {}),
        "varima":      config.get("varima_params",      {}),
        "nbeats":      config.get("nbeats_params",      {}),
        "transformer": config.get("transformer_params", {}),
    }

    active_indicators = [ind for ind, active in indicators_config.items() if active]

    logger.info(f"Config loaded | symbols={symbols} | timehorizon={timehorizon}")

    # ── Optuna tuning (optional) ──────────────────────────────────────────────
    # Set run_optuna=True to search for best hyperparameters before training.
    # Results are saved to SQLite so the search can be resumed if interrupted.
    run_optuna   = training_cfg.get("run_optuna", False)
    optuna_trials = int(training_cfg.get("optuna_trials", 30))
    optuna_timeout = training_cfg.get("optuna_timeout_per_model", None)
    # Will hold {model_name: best_params} populated during the first symbol run.
    _tuned_params: dict = {}

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol} …")
        df_1m = fetch_ohlcv_df(
            table_name=f"{symbol}_1m",
            schema="data_binance",
            time_column="datetime",
            start_date=start_date,
            end_date=end_date,
        )

        if df_1m is None or df_1m.empty:
            logger.warning(f"No data found for {symbol}. Skipping.")
            continue

        df_tf = resample_ohlcv(df_1m, timehorizon)
        logger.info(f"Generating indicators for {symbol} …")
        df_gf = generate_features(df_tf, active_indicators)
        split_date = resolve_split_date(df_gf, config)

        # BUG-FIX (main): derive feature columns once, outside the model loop,
        # so save_model receives indicator columns only (not raw OHLCV/target).
        feature_columns = [
            c for c in df_gf.columns if c not in _NON_FEATURE_COLS
        ]

        # ── Optuna: run search on first symbol only, reuse for subsequent ────
        active_models = [m for m, active in dl_models_config.items() if active]
        if run_optuna and not _tuned_params:
            logger.info("Running Optuna hyperparameter search …")
            _tuned_params = tune_all_models(
                df          = df_gf,
                df_1m       = df_1m,
                split_date  = split_date,
                models      = active_models,
                n_trials    = optuna_trials,
                timeout_per_model = optuna_timeout,
                high_performance  = training_cfg.get("high_performance", True),
            )
            logger.info(f"Optuna tuning complete. Best params: {_tuned_params}")

        for model_name, is_active in dl_models_config.items():
            if not is_active:
                continue

            # Merge: config params < tuned params (tuned wins when run_optuna=True)
            base_params  = model_params_map.get(model_name, {})
            tuned        = _tuned_params.get(model_name, {})
            final_params = {**base_params, **tuned}   # tuned overrides config

            logger.info(f"Training DL model: {model_name} for {symbol}")
            try:
                model, preds, test_index, df_test = train_model(
                    model_type="dl",
                    model_name=model_name,
                    df=df_gf,
                    df_1m=df_1m,
                    split_date=split_date,
                    lookback=lookback,
                    epochs=epochs,
                    batch_size=batch_size,
                    high_performance=training_cfg.get("high_performance", True),
                    model_params=final_params,
                )
                print(preds.head())
                # Pass model's signal_threshold as the `threshold` arg so
                # prepare_predictions applies the dead-band correctly inside
                # the dl_darts branch before signals are binarised to {-1,0,1}.
                # BackTest receives a clean ['datetime','signals'] DataFrame
                # with no extra columns and no API change needed.
                sig_thresh = getattr(model, "signal_threshold", 3e-4)

                df_predictions = prepare_predictions(
                    df_gf, preds, test_index,
                    model_type="dl_darts",
                    threshold=sig_thresh,
                )

                # Normalise datetime column to UTC-aware
                if "datetime" in df_predictions.columns:
                    dt_col = pd.to_datetime(df_predictions["datetime"])
                    if dt_col.dt.tz is None:
                        dt_col = dt_col.dt.tz_localize("UTC")
                    df_predictions["datetime"] = dt_col

                if df_1m is not None and not df_1m.empty:
                    bt = BackTest(
                        df_1m,
                        df_predictions,
                        take_profit=2,
                        stop_loss=1,
                    )
                    ledger, final_balance, pnl = bt.run()
                    print(ledger.head())
                else:
                    logger.warning(
                        f"df_1m unavailable for {symbol}; skipping backtest."
                    )
                    continue

                # test_feature_df kept for future permutation-importance support.
                test_feature_df = df_gf.iloc[test_index]    # noqa: F841

                table_name_dl = f"{model_name}_dl_{timestamp}"
                stats_df = compute_trade_statistics(ledger)
                stats_df.insert(0, "pnl",        pnl)
                stats_df.insert(0, "model_name", table_name_dl)
                save_df_to_db(
                    df=stats_df,
                    table_name="dl_results",
                    schema="model_stats",
                    time_column=None,
                    is_timeseries=False,
                )
            except Exception as exc:
                logger.error(f"DL model {model_name} failed for {symbol}: {exc}")

        logger.info(f"DL model training complete for {symbol}.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()