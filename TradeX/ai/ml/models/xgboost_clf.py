from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import optuna

from TradeX.ai.ml.models.train_utils import validate_and_sort,apply_log_diff_transform,split_features_labels,run_chunked_backtest
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("xgboost_classifier")
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
    """Train XGBClassifier using PnL-based Optuna optimization."""

    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    # Label encoding done ONCE outside objective — XGBoost requires [0, 1, 2, ...]
    le = LabelEncoder()
    y_train_enc   = le.fit_transform(y_train)
    y_test_enc    = le.transform(y_test)
    pos_class_idx = list(le.classes_).index(1)

    logger.info(f"[train] Starting Optuna study ({n_trials} trials)...")

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
            "objective":        "multi:softprob",
            "num_class":        len(le.classes_),
            "tree_method":      "hist",
            "eval_metric":      "mlogloss",
        }
        model = XGBClassifier(**params, n_jobs=1)
        model.fit(X_train, y_train_enc)
        probs = model.predict_proba(X_test)[:, pos_class_idx]

        return run_chunked_backtest(
            trial, df, probs, X_test.index, df_1m,
            model_type="classifier", k=k,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    logger.info(f"[train] Best params: {best_params} | Best PnL: {study.best_value:.4f}")

    final_params = {
        **best_params,
        "objective":   "multi:softprob",
        "num_class":   len(le.classes_),
        "eval_metric": "mlogloss",
    }
    final_model = XGBClassifier(**final_params, n_jobs=-1)
    final_model.fit(X_train, y_train_enc)
    final_probs = final_model.predict_proba(X_test)[:, pos_class_idx]

    logger.info(
        f"[train] Final preds — min: {final_probs.min():.4f}, "
        f"max: {final_probs.max():.4f}, mean: {final_probs.mean():.4f}"
    )
    return final_model, final_probs, X_test.index, X_test