from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import optuna

from TradeX.ai.ml.models.train_utils import validate_and_sort,apply_log_diff_transform,split_features_labels,run_chunked_backtest
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
    """Train XGBRegressor using PnL-based Optuna optimization."""

    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Starting Optuna study ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
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
        model = XGBRegressor(**params, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if np.std(preds) < 1e-8:
            logger.info(f"[trial {trial.number}] Skipping: constant predictions.")
            raise optuna.TrialPruned()

        return run_chunked_backtest(
            trial, df, preds, X_test.index, df_1m,
            model_type="regressor", k=k,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    logger.info(f"[train] Best params: {best_params} | Best PnL: {study.best_value:.4f}")

    final_model = XGBRegressor(**best_params, n_jobs=-1)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)

    logger.info(
        f"[train] Final preds — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
    )
    return final_model, final_preds, X_test.index, X_test