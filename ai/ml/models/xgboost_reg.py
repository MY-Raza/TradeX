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
    n_trials: int = 50,
    k: float = 0.5,
    transform_features: bool = True,
) -> tuple:
    """
    Train XGBRegressor using Optuna PnL-based optimisation.

    FIX BN2: study.optimize called with n_jobs=-1 (parallel trials).
    FIX BN1: single df.copy() at the top.
    """

    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df = df.copy()  # single copy
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"[train] DataFrame shape before transform: {df.shape}")

    if transform_features:
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_cols:
            df[col] = np.log(df[col]).diff()
        if "volume" in df.columns:
            df["volume"] = np.log1p(df["volume"]).diff()
        df = df.dropna(subset=price_cols).reset_index(drop=True)
        logger.info(f"[train] DataFrame shape after transform & dropna: {df.shape}")

    split_dt   = pd.to_datetime(split_date, utc=True)
    train_mask = df["datetime"] < split_dt

    if train_mask.sum() == 0:
        raise ValueError(f"Train set is empty for split_date='{split_date}'")
    if (~train_mask).sum() == 0:
        raise ValueError(f"Test set is empty for split_date='{split_date}'")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]
    X_train = df.loc[train_mask].drop(columns=drop_cols)
    y_train = df.loc[train_mask, target_col]
    X_test  = df.loc[~train_mask].drop(columns=drop_cols)

    logger.info(
        f"[train] Train rows: {len(X_train)}, Test rows: {len(X_test)}, "
        f"Features: {X_train.shape[1]}"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 800),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":           trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 5.0),
            "objective":       "reg:squarederror",
            "tree_method":     "hist",
            "n_jobs":          1,
            "random_state":    42,
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if np.std(preds) < 1e-8:
            raise optuna.TrialPruned()

        df_preds = prepare_predictions(
            df, preds, X_test.index,
            model_type="regressor", threshold=None, k=k,
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        n_chunks   = 5
        total_rows = len(df_preds)
        pnl_so_far = 0.0

        for chunk_idx in range(n_chunks):
            end_idx = (chunk_idx + 1) * total_rows // n_chunks
            bt = BackTest(df_1m, df_preds.iloc[:end_idx], take_profit=3, stop_loss=1)
            _, _, pnl_so_far = bt.run()
            trial.report(pnl_so_far, step=chunk_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far

    # FIX BN2: parallel Optuna trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_params = study.best_params
    logger.info(f"[train] Best Optuna parameters: {best_params}")
    logger.info(f"[train] Best trial PnL: {study.best_value:.4f}")

    final_model = XGBRegressor(**best_params, n_jobs=-1)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)

    logger.info(
        f"[train] Final predictions — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
    )

    return final_model, final_preds, X_test.index, X_test