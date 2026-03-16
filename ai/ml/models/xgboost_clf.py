from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import optuna
from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("xgboost_classifier")
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
    Train XGBClassifier using PnL-based Optuna optimization with pruning.
    Signals are generated via `prepare_predictions` for consistency with other models.

    Returns:
        final_model, final_probs, test_index, X_test
    """

    # ------------------ #
    # 1. Validate & sort #
    # ------------------ #
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain 'datetime'")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ------------------ #
    # 2. Feature transform #
    # ------------------ #
    if transform_features:
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_cols:
            df[col] = np.log(df[col]).diff()

        if "volume" in df.columns:
            df["volume"] = np.log1p(df["volume"]).diff()

        df = df.dropna(subset=price_cols).reset_index(drop=True)
        logger.info(f"DF shape after log-diff transform: {df.shape}")

    # ------------------ #
    # 3. Train/Test Split #
    # ------------------ #
    split_dt = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_dt].copy()
    test_df  = df[df["datetime"] >= split_dt].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Empty train/test split — adjust split_date")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df[target_col]

    logger.info(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Features: {X_train.shape[1]}")
    logger.info(f"[train] NaNs in X_train: {X_train.isna().sum().sum()} | X_test: {X_test.isna().sum().sum()}")
    logger.info(f"[train] NaNs in y_train: {y_train.isna().sum()} | y_test: {y_test.isna().sum()}")

    # XGBoost requires classes [0, 1, 2] — encode [-1, 0, 1] -> [0, 1, 2]
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    # Index of the positive class (original label = 1) in encoded space
    pos_class_idx = list(le.classes_).index(1)

    logger.info(f"[train] Starting Optuna study ({n_trials} trials)...")

    # ------------------ #
    # 4. Optuna Objective #
    # ------------------ #
    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "objective": "multi:softprob",
            "num_class": len(le.classes_),
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        }

        model = XGBClassifier(**params, n_jobs=1)
        logger.info(f"[trial {trial.number}] Fitting model on {len(X_train)} rows...")
        model.fit(X_train, y_train_enc)

        # Extract probability of the positive class (original label = 1)
        probs = model.predict_proba(X_test)[:, pos_class_idx]
        logger.info(f"[trial {trial.number}] Fit done. Running backtest chunks...")

        df_preds = prepare_predictions(
            df,
            probs,
            X_test.index,
            model_type="classifier",
            k=k
        )

        # ------------------ #
        # Chunked backtest for pruning
        # ------------------ #
        total_rows = len(df_preds)
        n_chunks = 5
        pnl_so_far = 0.0

        for i in range(n_chunks):
            end_idx = (i + 1) * total_rows // n_chunks
            df_chunk = df_preds.iloc[:end_idx]

            bt = BackTest(df_1m, df_chunk, take_profit=3, stop_loss=1)
            _, _, pnl_so_far = bt.run()

            trial.report(pnl_so_far, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far

    # ------------------ #
    # 5. Run Optuna study #
    # ------------------ #
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=4)

    best_params = study.best_params
    logger.info(f"[train] Best Optuna parameters: {best_params}")
    logger.info(f"[train] Best trial PnL: {study.best_value:.4f}")

    # ------------------ #
    # 6. Final model training #
    # ------------------ #
    final_params = {
        **best_params,
        "objective": "multi:softprob",
        "num_class": len(le.classes_),
        "eval_metric": "mlogloss",
    }
    final_model = XGBClassifier(**final_params, n_jobs=-1)
    logger.info("[train] Fitting final model...")
    final_model.fit(X_train, y_train_enc)
    final_probs = final_model.predict_proba(X_test)[:, pos_class_idx]
    logger.info(f"[train] Final predictions — min: {final_probs.min():.4f}, "
                f"max: {final_probs.max():.4f}, mean: {final_probs.mean():.4f}")

    return final_model, final_probs, X_test.index, X_test