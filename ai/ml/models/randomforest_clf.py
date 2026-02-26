from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import optuna

from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions


def train(
    df,
    df_1m,
    target_col="target",
    split_date="2024-01-01 00:00",
    n_trials=50,
    k=0.5
):
    """
    Train RandomForestClassifier using PnL-based Optuna optimization.
    Includes log-diff feature transformation.
    """

    # ==================================================
    # VALIDATION & SORTING
    # ==================================================
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have a 'datetime' column.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ==================================================
    # LOG-DIFF TRANSFORMATION (Stationarity)
    # ==================================================
    price_cols = ["open", "high", "low", "close"]

    for col in price_cols:
        if col in df.columns:
            df[col] = np.log(df[col]).diff()

    if "volume" in df.columns:
        df["volume"] = np.log1p(df["volume"]).diff()

    df = df.dropna().reset_index(drop=True)

    # ==================================================
    # TRAIN / TEST SPLIT
    # ==================================================
    split_date = pd.to_datetime(split_date, utc=True)

    train_df = df[df["datetime"] < split_date]
    test_df  = df[df["datetime"] >= split_date]

    if train_df.empty or test_df.empty:
        raise ValueError("Train/Test split resulted in empty dataset.")

    # ==================================================
    # FEATURE SELECTION
    # ==================================================
    drop_cols = [target_col, "datetime", "future_close"]

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = train_df[target_col]
    y_test  = test_df[target_col]

    # ==================================================
    # OPTUNA OBJECTIVE (PnL Optimization)
    # ==================================================
    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }

        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Convert predictions → trading signals
        df_preds = prepare_predictions(
            df,
            preds,
            X_test.index,
            model_type="classifier",
            k=k
        )

        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        # Backtest
        bt = BackTest(
            df_1m,
            df_preds,
            take_profit=3,
            stop_loss=1
        )

        _, _, pnl = bt.run()

        return pnl  # 🔥 maximize profit

    # ==================================================
    # RUN OPTUNA
    # ==================================================
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best Optuna Parameters (PnL): {best_params}")

    # ==================================================
    # FINAL MODEL TRAINING
    # ==================================================
    final_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)

    return final_model, final_preds, X_test.index, X_test