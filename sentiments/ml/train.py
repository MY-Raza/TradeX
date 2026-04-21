from __future__ import annotations

import sys
import traceback
from datetime import datetime

import pandas as pd

from TradeX.utils.common.logs import get_logger
from TradeX.utils.db.utils import save_df_to_db

# ── Pipeline modules ──────────────────────────────────────
from TradeX.sentiments.ml.data.data_loader      import load_features_from_db,load_price_data
from TradeX.sentiments.ml.data.preprocessing    import split_data_timewise, prepare_features_sentiments
from TradeX.sentiments.ml.model           import train_classification_model, train_regression_model, evaluate_models
from TradeX.sentiments.ml.backtesting.signals          import generate_signals
from TradeX.sentiments.ml.backtesting.backtest_runner  import run_backtest
from TradeX.indicators.talib.indicators import call_indicator, ALL_INDICATORS, TA_DEFAULT_WINDOWS
from TradeX.ai.data.data_pipeline import generate_features

# ── Config ────────────────────────────────────────────────
from config import (
    DB_SCHEMA_OUTPUT,
    TABLE_ML_PREDICTIONS,
    TABLE_BACKTEST_RESULTS,
    TABLE_BACKTEST_SUMMARY,
    DATETIME_COL,
)

logger = get_logger("train")


# =========================================================
# OPTIONAL: FEATURE IMPORTANCE VISUALISATION (BONUS)
# =========================================================

def plot_feature_importances(
    clf_importances: pd.Series,
    reg_importances: pd.Series,
    top_n: int = 20,
) -> None:
    """
    Save feature-importance bar charts to disk.
    Requires matplotlib — silently skips if not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping importance plots.")
        return

    for importances, title, fname in [
        (clf_importances, "Classifier Feature Importances", "clf_importances.png"),
        (reg_importances, "Regressor Feature Importances",  "reg_importances.png"),
    ]:
        top = importances.head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
        top[::-1].plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close(fig)
        logger.info(f"Saved importance plot → {fname}")


# =========================================================
# DB SAVE HELPERS
# =========================================================

def _save_predictions(df_signals: pd.DataFrame) -> None:
    logger.info(f"Saving predictions to {DB_SCHEMA_OUTPUT}.{TABLE_ML_PREDICTIONS} …")
    save_df_to_db(
        df_signals[[DATETIME_COL, "signals"]].copy(),
        table_name=TABLE_ML_PREDICTIONS,
        schema=DB_SCHEMA_OUTPUT,
        time_column=DATETIME_COL,
        is_timeseries=True,
    )
    logger.info(f"  Saved {len(df_signals)} signal rows.")


def _save_backtest_results(ledger: pd.DataFrame) -> None:
    if ledger.empty:
        logger.warning("Ledger is empty — skipping backtest_results save.")
        return
    logger.info(f"Saving ledger to {DB_SCHEMA_OUTPUT}.{TABLE_BACKTEST_RESULTS} …")
    save_df_to_db(
        ledger.copy(),
        table_name=TABLE_BACKTEST_RESULTS,
        schema=DB_SCHEMA_OUTPUT,
        time_column="datetime",
        is_timeseries=False,
        enforce_unique_time=False
    )
    logger.info(f"  Saved {len(ledger)} ledger rows.")


def _save_backtest_summary(summary: dict) -> None:
    logger.info(f"Saving summary to {DB_SCHEMA_OUTPUT}.{TABLE_BACKTEST_SUMMARY} …")
    summary_df = pd.DataFrame([summary])
    summary_df["run_at"] = pd.Timestamp.utcnow()
    save_df_to_db(
        summary_df,
        table_name=TABLE_BACKTEST_SUMMARY,
        schema=DB_SCHEMA_OUTPUT,
        time_column="run_at",
        is_timeseries=False,
        enforce_unique_time=False,
    )
    logger.info("  Summary saved.")


# =========================================================
# PIPELINE ORCHESTRATOR
# =========================================================

def run_pipeline(save_to_db: bool = True, plot: bool = True) -> None:
    """
    Execute the full ML trading pipeline end-to-end.

    Parameters
    ----------
    save_to_db : bool
        Set False to skip all DB writes (useful for dry runs / unit tests).
    plot : bool
        Set False to skip feature importance plots.
    """
    start_time = datetime.utcnow()
    logger.info("=" * 72)
    logger.info("PIPELINE START")
    logger.info("=" * 72)

    # ----------------------------------------------------------
    # STEP 1 — Load feature data
    # ----------------------------------------------------------
    logger.info("--- STEP 1: Load features ---")
    df_features = load_features_from_db()

    df_price_raw = load_price_data()
    

    indicator_df = generate_features(df_price_raw,ALL_INDICATORS)
    if DATETIME_COL in indicator_df.columns:
        indicator_df = indicator_df.drop(columns=[DATETIME_COL])

    df_features = df_features.set_index(DATETIME_COL)
    df_features = df_features.join(indicator_df, how="left")
    print(df_features.columns)
    columns_list = df_features.columns.tolist()
    df_features[indicator_df.columns] = df_features[indicator_df.columns].fillna(0.0)
    df_features = df_features.reset_index()

    # ----------------------------------------------------------
    # STEP 2 — Time-based split
    # ----------------------------------------------------------
    logger.info("--- STEP 2: Split data ---")
    df_train, df_val, df_test = split_data_timewise(df_features)

    # ----------------------------------------------------------
    # STEP 3 — Scale features
    # ----------------------------------------------------------
    logger.info("--- STEP 3: Prepare / scale features ---")
    prepared = prepare_features_sentiments(df_train, df_val, df_test)

    # ----------------------------------------------------------
    # STEP 4 — Train models
    # ----------------------------------------------------------
    logger.info("--- STEP 4: Train models ---")
    clf = train_classification_model(prepared)
    reg = train_regression_model(prepared)

    # ----------------------------------------------------------
    # STEP 5 — Evaluate models
    # ----------------------------------------------------------
    logger.info("--- STEP 5: Evaluate models ---")
    bundle = evaluate_models(clf, reg, prepared)

    # ----------------------------------------------------------
    # STEP 6 — Generate signals
    # ----------------------------------------------------------
    logger.info("--- STEP 6: Generate signals ---")
    df_signals = generate_signals(bundle, prepared)
    df_signals.to_csv("signals.csv", index=False)

    # ----------------------------------------------------------
    # STEP 7 — Run backtest
    # ----------------------------------------------------------
    logger.info("--- STEP 7: Run backtest ---")
    result = run_backtest(df_signals)

    # ----------------------------------------------------------
    # STEP 8 — Save outputs
    # ----------------------------------------------------------
    if save_to_db:
        logger.info("--- STEP 8: Save outputs to DB ---")
        _save_predictions(df_signals)
        _save_backtest_results(result.ledger)
        _save_backtest_summary(result.summary)
    else:
        logger.info("--- STEP 8: DB write SKIPPED (save_to_db=False) ---")

    # ----------------------------------------------------------
    # STEP 9 — (Bonus) Plot feature importances
    # ----------------------------------------------------------
    if plot:
        logger.info("--- STEP 9: Plot feature importances ---")
        plot_feature_importances(
            bundle.clf_importances,
            bundle.reg_importances,
        )

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------
    elapsed = (datetime.utcnow() - start_time).total_seconds()

    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Elapsed          : {elapsed:.1f} s")
    logger.info(f"  Starting balance : {result.summary['starting_balance']:.2f}")
    logger.info(f"  Final balance    : {result.final_balance:.2f}")
    logger.info(f"  Total PnL %%      : {result.total_pnl_pct:.2f}%%")
    logger.info(f"  Trades executed  : {result.summary['number_of_trades']}")
    if "win_rate" in result.summary:
        logger.info(f"  Win rate         : {result.summary['win_rate']:.1%}")
    if "max_drawdown" in result.summary:
        logger.info(f"  Max drawdown     : {result.summary['max_drawdown']:.2f}%%")
    logger.info("=" * 72)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    try:
        run_pipeline(save_to_db=True, plot=True)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception:
        logger.error("Pipeline failed with unhandled exception:")
        logger.error(traceback.format_exc())
        sys.exit(1)