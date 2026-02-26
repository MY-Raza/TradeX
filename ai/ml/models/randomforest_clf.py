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
    Train RandomForestClassifier using PnL-based Optuna optimization with pruning.
    Includes log-diff feature transformation.
    """

    # ----------------------------
    # Validation & Sorting
    # ----------------------------
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have a 'datetime' column.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ----------------------------
    # Log-Diff Transformation
    # ----------------------------
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = np.log(df[col]).diff()
    if "volume" in df.columns:
        df["volume"] = np.log1p(df["volume"]).diff()

    df = df.dropna().reset_index(drop=True)

    # ----------------------------
    # Train/Test Split
    # ----------------------------
    split_date = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_date]
    test_df  = df[df["datetime"] >= split_date]

    if train_df.empty or test_df.empty:
        raise ValueError("Train/Test split resulted in empty dataset.")

    drop_cols = [target_col, "datetime", "future_close"]

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test = test_df[target_col]

    # ----------------------------
    # Optuna Objective (PnL Optimization with Pruning)
    # ----------------------------
    def objective(trial):

        # Suggest hyperparameters
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

        # Convert predictions → signals
        df_preds = prepare_predictions(
            df,
            preds,
            X_test.index,
            model_type="classifier",
            k=k
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        # ----------------------------
        # Backtest in chunks for pruning
        # ----------------------------
        total_rows = len(df_preds)
        n_chunks = 5  # Split test set into 5 parts for intermediate reporting

        pnl_so_far = 0
        for i in range(n_chunks):
            start_idx = i * total_rows // n_chunks
            end_idx   = (i + 1) * total_rows // n_chunks
            df_chunk = df_preds.iloc[:end_idx]  # cumulative for realistic backtest

            bt = BackTest(df_1m, df_chunk, take_profit=3, stop_loss=1)
            _, _, pnl_chunk = bt.run()
            pnl_so_far = pnl_chunk

            # Report intermediate PnL to Optuna
            trial.report(pnl_so_far, step=i)

            # Prune if trial is not promising
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far  # maximize profit

    # ----------------------------
    # Run Optuna with Pruner
    # ----------------------------
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best Optuna Parameters: {best_params}")

    # ----------------------------
    # Final Model Training
    # ----------------------------
    final_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)

    return final_model, final_preds, X_test.index, X_test