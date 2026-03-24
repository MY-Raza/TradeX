from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import optuna

from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.models.train_utils import validate_and_sort,apply_log_diff_transform,split_features_labels,run_chunked_backtest
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("randomforest_classifier")
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
    """Train RandomForestClassifier using PnL-based Optuna optimization."""

    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Starting Optuna study ({n_trials} trials)...")

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
            "criterion":         trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
        model = RandomForestClassifier(random_state=42, n_jobs=1, **params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        return run_chunked_backtest(
            trial, df, probs, X_test.index, df_1m,
            model_type="classifier", k=k,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    logger.info(f"[train] Best params: {best_params} | Best PnL: {study.best_value:.4f}")

    final_model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    final_model.fit(X_train, y_train)
    final_probs = final_model.predict_proba(X_test)[:, 1]

    logger.info(
        f"[train] Final preds — min: {final_probs.min():.4f}, "
        f"max: {final_probs.max():.4f}, mean: {final_probs.mean():.4f}"
    )
    return final_model, final_probs, X_test.index, X_test