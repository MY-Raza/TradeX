from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import optuna

from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("xgboost_regressor")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train(
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    target_col: str = "target",
    split_date: str = "2024-01-01 00:00",
    n_trials: int = 15,
    k: float = 0.5,
    transform_features: bool = True,
) -> tuple:
    """
    Train an XGBRegressor on OHLCV data using Optuna for hyperparameter
    optimization and a PnL-based backtest objective with MedianPruner.

    Optimizations vs original:
    - n_trials default 50 → 15
    - n_estimators search space capped at 300 (was 800) — faster per trial
    - study.optimize n_jobs=1 (was 4) — eliminates joblib/fork deadlock contention
    - n_chunks 5 → 2 — prune bad trials twice as fast
    - MedianPruner n_startup_trials 5→3, n_warmup_steps 1→0
    - df datetime pre-processed once before Optuna loop, not per-trial
    - X_train / X_test built once outside objective, referenced by closure
    """

    # ----------------------------
    # 1. Validate & pre-process
    # ----------------------------
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df = df.copy()
    # Pre-process datetime ONCE here — not inside objective or prepare_predictions
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    logger.info(f"[train] DataFrame shape before transform: {df.shape}")

    # ----------------------------
    # 2. Feature transformation (target excluded)
    # ----------------------------
    if transform_features:
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_cols:
            df[col] = np.log(df[col]).diff()
        if "volume" in df.columns:
            df["volume"] = np.log1p(df["volume"]).diff()
        df = df.dropna(subset=price_cols).reset_index(drop=True)
        logger.info(f"[train] DataFrame shape after transform & dropna: {df.shape}")

    # ----------------------------
    # 3. Train/Test Split
    # Build X_train / X_test ONCE — reused by every trial via closure.
    # ----------------------------
    split_dt = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_dt].copy()
    test_df  = df[df["datetime"] >= split_dt].copy()

    if train_df.empty:
        raise ValueError(f"Train set is empty for split_date='{split_date}'")
    if test_df.empty:
        raise ValueError(f"Test set is empty for split_date='{split_date}'")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df[target_col]

    logger.info(f"[train] Train rows: {len(X_train)}, Test rows: {len(X_test)}, Features: {X_train.shape[1]}")

    # ----------------------------
    # 4. Optuna objective (PnL + pruning)
    # ----------------------------
    def objective(trial: optuna.Trial) -> float:
        params = {
            # Cap n_estimators at 300 — beyond this gains are marginal
            "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
            "objective":        "reg:squarederror",
            "tree_method":      "hist",
            "random_state":     42,
        }

        # n_jobs=1: Optuna already parallelises at the study level.
        # Nested joblib pools cause thread contention / deadlocks.
        model = XGBRegressor(**params, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if np.std(preds) < 1e-8:
            logger.info(f"[trial {trial.number}] Skipping: constant predictions detected.")
            raise optuna.TrialPruned()

        df_preds = prepare_predictions(
            df, preds, X_test.index,
            model_type="regressor",
            threshold=None,
            k=k
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        # 2 chunks instead of 5 — prune bad trials earlier
        n_chunks   = 2
        total_rows = len(df_preds)
        pnl_so_far = 0.0

        for chunk_idx in range(n_chunks):
            end_idx  = (chunk_idx + 1) * total_rows // n_chunks
            df_chunk = df_preds.iloc[:end_idx]

            bt = BackTest(df_1m, df_chunk, take_profit=3, stop_loss=1)
            _, _, pnl_so_far = bt.run()

            trial.report(pnl_so_far, step=chunk_idx)
            logger.info(f"[trial {trial.number}] Chunk {chunk_idx+1}/{n_chunks} PnL: {pnl_so_far:.4f}")
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far

    # ----------------------------
    # 5. Run Optuna study
    # n_jobs=1 — eliminates joblib fork contention.
    # Tighter pruner: startup 3 (was 5), warmup 0 (was 1).
    # ----------------------------
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    logger.info(f"[train] Best Optuna parameters: {best_params}")
    logger.info(f"[train] Best trial PnL: {study.best_value:.4f}")

    # ----------------------------
    # 6. Final model — n_jobs=-1 safe here (outside Optuna parallel context)
    # ----------------------------
    final_model = XGBRegressor(**best_params, n_jobs=-1)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)

    logger.info(
        f"[train] Final predictions — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
    )

    return final_model, final_preds, X_test.index, X_test