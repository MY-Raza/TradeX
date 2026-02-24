from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import optuna
from TradeX.backtest.backtest import BackTest


def train(df,
          target_col="target",
          split_date="2024-01-01 00:00",
          n_trials=50,
          df_ohlcv_1m=None,     # <-- pass 1m data
          take_profit=3,
          stop_loss=1):
    """
    Train XGBClassifier optimizing for Backtest PnL.
    """

    # ----------------------------
    # Ensure datetime column
    # ----------------------------
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have a 'datetime' column.")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    split_date = pd.to_datetime(split_date, utc=True)

    train_df = df[df["datetime"] < split_date]
    test_df = df[df["datetime"] >= split_date]

    X_train = train_df.drop(columns=[target_col, "datetime"])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, "datetime"])
    y_test = test_df[target_col]

    # ==================================================
    # OPTUNA OBJECTIVE — MAXIMIZE PnL
    # ==================================================
    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # ----------------------------
        # Build prediction dataframe
        # ----------------------------
        df_predictions = pd.DataFrame({
            "datetime": test_df["datetime"].values,
            "signals": preds
        })

        # ----------------------------
        # Run Backtest
        # ----------------------------
        bt = BackTest(
            df_ohlcv_1m,
            df_predictions,
            take_profit=take_profit,
            stop_loss=stop_loss
        )

        _, _, pnl = bt.run()

        return pnl  # maximize pnl

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best PnL Parameters: {best_params}")

    # ==================================================
    # FINAL TRAINING
    # ==================================================
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return model, preds, X_test.index