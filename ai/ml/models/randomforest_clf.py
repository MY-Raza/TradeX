from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import optuna

from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("randomforest_classifier")
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
    Train RandomForestClassifier using PnL-based Optuna optimization with pruning.
    Signals are generated via `prepare_predictions` for consistency with other models.
    """

    # ------------------ #
    # 1. Validate & sort #
    # ------------------ #
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

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
    # 3. Train/Test split #
    # ------------------ #
    split_dt = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_dt].copy()
    test_df  = df[df["datetime"] >= split_dt].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/Test split resulted in empty dataset.")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df[target_col]

    logger.info(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Features: {X_train.shape[1]}")

    # ------------------ #
    # 4. Optuna Objective #
    # ------------------ #
    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }

        model = RandomForestClassifier(random_state=42, n_jobs=1, **params)
        model.fit(X_train, y_train)

        # Use predict_proba[:,1] for signal generation
        probs = model.predict_proba(X_test)[:, 1]

        # Generate signals using prepare_predictions
        df_preds = prepare_predictions(
            df,
            probs,
            X_test.index,
            model_type="classifier",
            k=k
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        # Chunked backtest for pruning
        total_rows = len(df_preds)
        n_chunks = 5
        pnl_so_far = 0.0

        for i in range(n_chunks):
            end_idx = (i + 1) * total_rows // n_chunks
            df_chunk = df_preds.iloc[:end_idx]

            bt = BackTest(df_1m, df_chunk, take_profit=3, stop_loss=1)
            _, _, pnl_chunk = bt.run()
            pnl_so_far = pnl_chunk

            trial.report(pnl_so_far, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far

    # ------------------ #
    # 5. Run Optuna study #
    # ------------------ #
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best Optuna params: {best_params} | Best PnL: {study.best_value:.4f}")

    # ------------------ #
    # 6. Final model training #
    # ------------------ #
    final_model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    final_model.fit(X_train, y_train)

    final_probs = final_model.predict_proba(X_test)[:, 1]

    return final_model, final_probs, X_test.index, X_test