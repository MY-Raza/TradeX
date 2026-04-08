from __future__ import annotations

import os
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

from TradeX.ai.darts.models.model_trainer import train_model, save_model
from TradeX.ai.darts.optuna_tuner import tune_all_models
from TradeX.ai.ml.utils import (
    compute_trade_statistics,
    prepare_predictions,
)
from TradeX.backtest.backtest import BackTest
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger
from TradeX.utils.db.utils import save_df_to_db

# ── All data operations come from data_pipeline ───────────────────────────────
from TradeX.ai.data.data_pipeline import fetch_raw_data,resample_data,prepare_features,resolve_split_date

logger = get_logger("dl_model_main")

# Raw OHLCV + target columns that should NOT be passed as features to save_model.
_NON_FEATURE_COLS = frozenset({
    "open", "high", "low", "close", "volume",
    "datetime", "log_return", "timestamp",
})


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    current_dir    = os.path.dirname(os.path.abspath(__file__))
    dl_config_path = os.path.join(current_dir, "config.yml")
    config         = read_config(dl_config_path)

    start_date        = config.get("start_date")
    end_date          = config.get("end_date")
    symbols           = ["btc"]
    timehorizon       = config.get("timehorizon", "1h")
    indicators_config = config.get("indicators", {})
    dl_models_config  = config.get("forecasting_models", {})
    training_cfg      = config.get("training", {})
    adf_significance  = 0.05
    max_diffs         = 2

    # training_cfg has no "lookback"/"epochs"/"batch_size" keys — those live
    # inside model-specific param dicts. Pass None so model_trainer uses its
    # own model-specific config params.
    lookback   = training_cfg.get("lookback",   None)
    epochs     = training_cfg.get("epochs",     None)
    batch_size = training_cfg.get("batch_size", None)

    model_params_map: dict[str, dict] = {
        "arima":       config.get("arima_params",       {}),
        "varima":      config.get("varima_params",      {}),
        "nbeats":      config.get("nbeats_params",      {}),
        "transformer": config.get("transformer_params", {}),
    }

    active_indicators = [ind for ind, active in indicators_config.items() if active]

    logger.info(f"Config loaded | symbols={symbols} | timehorizon={timehorizon}")

    # Optuna config
    run_optuna     = False
    optuna_trials  = 30
    optuna_timeout = None
    # Populated during the first symbol run and reused for subsequent symbols.
    _tuned_params: dict = {}

    # ------------------------------------------------------------------
    # Per-symbol loop
    # ------------------------------------------------------------------
    for symbol in symbols:
        logger.info(f"=== Processing symbol: {symbol} ===")

        # ── 1. Fetch raw 1-minute data ──────────────────────────────
        try:
            df_1m = fetch_raw_data(
                symbol=symbol,
                schema="data_binance",
                start_date=start_date,
                end_date=end_date,
            )
        except ValueError as exc:
            logger.warning(f"Skipping {symbol}: {exc}")
            continue

        # ── 2. Resample to target timeframe ──────────────────────────
        df_tf = resample_data(df_1m, timehorizon)

        # ── 3. Feature engineering + stationarity fixing ─────────────
        #   prepare_features returns a fully stationary feature DataFrame
        #   and a diff-count audit dict (logged inside data_pipeline).
        logger.info(f"Generating indicators and checking stationarity for {symbol} …")
        df_gf, diff_counts = prepare_features(
            df=df_tf,
            active_indicators=active_indicators,
            adf_significance=adf_significance,
            max_diffs=max_diffs,
        )

        # ── 4. Resolve train/test split date ─────────────────────────
        #   Done after feature engineering so train_ratio operates on
        #   the actual stationary rows seen by the trainers.
        split_date = resolve_split_date(df_gf, config)

        # Derive feature columns once, outside the model loop, so
        # save_model receives indicator columns only (not raw OHLCV/target).
        feature_columns = [c for c in df_gf.columns if c not in _NON_FEATURE_COLS]

        # ── 5. Optuna tuning (optional, first symbol only) ───────────
        active_models = [m for m, active in dl_models_config.items() if active]
        if run_optuna and not _tuned_params:
            logger.info("Running Optuna hyperparameter search …")
            _tuned_params = tune_all_models(
                df=df_gf,
                df_1m=df_1m,
                split_date=split_date,
                models=active_models,
                n_trials=optuna_trials,
                timeout_per_model=optuna_timeout,
                high_performance=training_cfg.get("high_performance", True),
            )
            logger.info(f"Optuna tuning complete. Best params: {_tuned_params}")

        # ==============================================================
        # DL MODEL LOOP
        # ==============================================================
        for model_name, is_active in dl_models_config.items():
            if not is_active:
                continue

            # Merge: config params < tuned params (tuned wins when run_optuna=True)
            base_params  = model_params_map.get(model_name, {})
            tuned        = _tuned_params.get(model_name, {})
            final_params = {**base_params, **tuned}

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

                # Apply model's signal threshold inside the dl_darts branch
                # so prepare_predictions correctly binarises to {-1, 0, 1}.
                sig_thresh = getattr(model, "signal_threshold", 3e-4)

                df_predictions = prepare_predictions(
                    df_gf, preds, test_index,
                    model_type="dl_darts",
                    threshold=sig_thresh,
                )

                # Normalise datetime to UTC-aware
                if "datetime" in df_predictions.columns:
                    dt_col = pd.to_datetime(df_predictions["datetime"])
                    if dt_col.dt.tz is None:
                        dt_col = dt_col.dt.tz_localize("UTC")
                    df_predictions["datetime"] = dt_col

                # Backtest
                bt = BackTest(
                    df_1m,
                    df_predictions,
                    take_profit=2,
                    stop_loss=1,
                )
                ledger, final_balance, pnl = bt.run()
                logger.info(
                    f"[{model_name}] Balance={final_balance:.2f}  PnL={pnl:.4f}"
                )

                # Persist trade statistics
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


# ===========================================================================
if __name__ == "__main__":
    main()