from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datetime import datetime

from TradeX.utils.db.utils import save_df_to_db
from TradeX.ai.dl.models.model_trainer import train_model, save_model
from TradeX.ai.ml.utils import (
    prepare_predictions,
    pnl_permutation_importance,
    extract_important_features,
    compute_trade_statistics,
)
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger
from TradeX.backtest.backtest import BackTest

# Reuse the same data_pipeline — no changes needed
from TradeX.ai.data.data_pipeline import (
    fetch_raw_data,
    resample_data,
    prepare_features,
    build_classification_df,
    build_regression_df,
    resolve_split_date,
)

logger = get_logger("main_dl")


# ===========================================================================
# HELPERS  (identical to main.py)
# ===========================================================================

def _run_backtest(
    df_1m: pd.DataFrame,
    model,
    preds,
    test_index,
    df_test_norm: pd.DataFrame,
    model_type: str,
    k: float = 0.5,
) -> tuple:
    """Prepare predictions and run a BackTest. Returns (ledger, balance, pnl).

    ``df_test_norm`` must be the normalised (reset-index) test slice returned
    by each model's ``train()`` function.  Its RangeIndex matches the integer
    labels in ``test_index``, so ``prepare_predictions`` can look up datetimes
    correctly without any index translation.

    All three arrays — preds, test_index, and df_test_norm — are trimmed to
    the shortest common length so ``prepare_predictions`` never receives
    mismatched inputs.
    """
    import numpy as np

    preds_np = np.asarray(preds)
    idx_arr  = np.asarray(test_index)

    # Triple-align: preds, index positions, and the datetime lookup frame
    # must all have the same length before entering prepare_predictions.
    min_len      = min(len(preds_np), len(idx_arr), len(df_test_norm))
    preds_np     = preds_np[-min_len:]
    idx_arr      = np.arange(min_len)          # always 0..min_len-1 for iloc safety
    df_test_norm = df_test_norm.iloc[-min_len:].reset_index(drop=True)

    df_predictions = prepare_predictions(
        df_test_norm, preds_np, idx_arr, model_type=model_type, k=k
    )
    df_predictions["datetime"] = pd.to_datetime(df_predictions["datetime"], utc=True)

    bt = BackTest(df_1m, df_predictions, take_profit=3, stop_loss=1)
    return bt.run()


def _compute_and_save_importance(
    model,
    X_test: pd.DataFrame,
    df_test_norm: pd.DataFrame,
    df_1m: pd.DataFrame,
    pnl: float,
    model_type: str,
    table_name: str,
) -> None:
    """Compute PnL-permutation importance and persist to DB.

    ``df_test_norm`` must be the length-aligned DataFrame returned by
    ``train_model`` (i.e. ``df_for_backtest`` with RangeIndex 0..P-1).
    Using the full ``df_clf`` / ``df_reg`` here caused the
    "All arrays must be of the same length" crash because its row count
    did not match the (seq_len-shortened) prediction array.
    """
    pnl_importance_df = pnl_permutation_importance(
        model=model,
        X_test=X_test,
        df=df_test_norm,
        df_1m=df_1m,
        base_pnl=pnl,
        model_type=model_type,
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

    important_features_df = extract_important_features(pnl_importance_wide, table_name)
    save_df_to_db(
        df=important_features_df,
        table_name="best_features",
        schema="ml_features",
        time_column=None,
        is_timeseries=False,
    )


def _save_trade_stats(
    ledger: pd.DataFrame,
    pnl: float,
    model_name: str,
) -> None:
    """Compute trade statistics and persist to DB."""
    stats_df = compute_trade_statistics(ledger)
    stats_df.insert(0, "pnl", pnl)
    stats_df.insert(0, "model_name", model_name)
    save_df_to_db(
        df=stats_df,
        table_name="ml_results",
        schema="model_stats",
        time_column=None,
        is_timeseries=False,
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    current_dir    = os.path.dirname(os.path.abspath(__file__))
    ml_config_path = os.path.join(current_dir, "config.yml")
    config         = read_config(ml_config_path)

    start_date        = config.get("start_date")
    end_date          = config.get("end_date")
    symbols           = ["btc"]
    timehorizon       = config.get("timehorizon", "1h")
    indicators_config = config.get("indicators", {})
    adf_significance  = 0.05
    max_diffs         = 2

    # Read DL-specific config; fall back to empty dicts so it's optional
    dl_classifiers   = config.get("classifiers", {})
    dl_regressors    = config.get("regressors", {})
    dl_n_trials      = int(config.get("dl_n_trials", 10))

    active_indicators = [ind for ind, active in indicators_config.items() if active]

    logger.info(
        f"DL Config loaded | symbols={symbols} | timehorizon={timehorizon} | "
        f"n_trials={dl_n_trials}"
    )

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

        # ── 2. Resample ──────────────────────────────────────────────
        df_resampled = resample_data(df_1m, timehorizon)

        # ── 3. Feature engineering + stationarity fixing ─────────────
        df_gf, diff_counts = prepare_features(
            df=df_resampled,
            active_indicators=active_indicators,
            adf_significance=adf_significance,
            max_diffs=max_diffs,
        )

        # ── 4. Resolve train/test split date ─────────────────────────
        split_date = resolve_split_date(df_gf, config)

        # ==============================================================
        # DL CLASSIFIERS
        # ==============================================================
        for clf_name, is_active in dl_classifiers.items():
            if not is_active:
                continue

            logger.info(f"Training DL classifier: {clf_name} for {symbol}")
            try:
                df_clf = build_classification_df(df_gf)

                model, preds, test_index, X_test, df_test_norm = train_model(
                    model_type="classifier",
                    model_name=clf_name,
                    df=df_clf,
                    target_col="target",
                    split_date=split_date,
                    n_trails=dl_n_trials,
                    df_1m=df_1m,
                )

                # Dry-run sanity check
                # Use seq_len + 5 rows so at least one sequence can be built.
                try:
                    dry_run_rows = model.seq_len + 5
                    sample_preds = model.predict(X_test.head(dry_run_rows))
                    logger.info(
                        f"[Dry-run] {clf_name} predictions on first samples:\n"
                        f"{sample_preds[:5]}"
                    )
                except Exception as exc:
                    logger.error(f"[Dry-run] Failed for {clf_name}: {exc}")

                # Backtest
                ledger, final_balance, pnl = _run_backtest(
                    df_1m, model, preds, test_index, df_test_norm, "classifier", k=0.5
                )
                logger.info(f"[{clf_name}] Balance={final_balance:.2f}  PnL={pnl:.4f}")

                # Importance + stats
                # Pass df_test_norm (length-aligned, RangeIndex 0..P-1) so
                # pnl_permutation_importance receives a df whose row count
                # matches the (seq_len-shortened) prediction array exactly.
                table_name_clf = f"{clf_name}_dl_clf_{timestamp}"
                _compute_and_save_importance(
                    model, X_test, df_test_norm, df_1m, pnl, "classifier", table_name_clf
                )
                _save_trade_stats(ledger, pnl, table_name_clf)

                # Persist
                save_model(
                    model,
                    X_test.columns.tolist(),
                    symbol,
                    f"{clf_name}_dl_classifier_{timestamp}",
                )

            except Exception as exc:
                logger.error(f"DL Classifier {clf_name} failed for {symbol}: {exc}")

        # ==============================================================
        # DL REGRESSORS
        # ==============================================================
        for reg_name, is_active in dl_regressors.items():
            if not is_active:
                continue

            logger.info(f"Training DL regressor: {reg_name} for {symbol}")
            try:
                df_reg = build_regression_df(df_gf)

                model, preds, test_index, X_test, df_test_norm = train_model(
                    model_type="regressor",
                    model_name=reg_name,
                    df=df_reg,
                    target_col="target",
                    split_date=split_date,
                    n_trails=dl_n_trials,
                    df_1m=df_1m,
                )

                # Dry-run sanity check
                # Use seq_len + 5 rows so at least one sequence can be built.
                try:
                    dry_run_rows = model.seq_len + 5
                    sample_preds = model.predict(X_test.head(dry_run_rows))
                    logger.info(
                        f"[Dry-run] {reg_name} predictions on first samples:\n"
                        f"{sample_preds[:5]}"
                    )
                except Exception as exc:
                    logger.error(f"[Dry-run] Failed for {reg_name}: {exc}")

                # Backtest
                ledger, final_balance, pnl = _run_backtest(
                    df_1m, model, preds, test_index, df_test_norm, "regressor", k=0.5
                )
                logger.info(f"[{reg_name}] Balance={final_balance:.2f}  PnL={pnl:.4f}")

                # Importance + stats
                # Pass df_test_norm (length-aligned, RangeIndex 0..P-1) so
                # pnl_permutation_importance receives a df whose row count
                # matches the (seq_len-shortened) prediction array exactly.
                table_name_reg = f"{reg_name}_dl_reg_{timestamp}"
                _compute_and_save_importance(
                    model, X_test, df_test_norm, df_1m, pnl, "regressor", table_name_reg
                )
                _save_trade_stats(ledger, pnl, table_name_reg)

                # Persist
                save_model(
                    model,
                    X_test.columns.tolist(),
                    symbol,
                    f"{reg_name}_dl_regressor_{timestamp}",
                )

            except Exception as exc:
                logger.error(f"DL Regressor {reg_name} failed for {symbol}: {exc}")

        logger.info(f"DL model training complete for {symbol}.")


# ===========================================================================
if __name__ == "__main__":
    main()