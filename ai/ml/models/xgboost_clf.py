from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import accuracy_score

def train(df, target_col="target", split_date="2024-01-01 00:00", n_trials=50):
    """
    Train XGBClassifier using time-based split with Optuna hyperparameter optimization.

    Args:
        df (pd.DataFrame): Input dataframe with features and target
        target_col (str): Name of target column
        split_date (str): Date string to split train/test
        n_trials (int): Number of Optuna trials for hyperparameter search
        **xgb_params: Extra XGBClassifier hyperparameters to override

    Returns:
        model: trained XGBClassifier
        preds: predictions on test set
        test_index: indices of test set
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
    # OPTUNA SECTION
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
            "eval_metric": "logloss"
        }


        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        return acc  # maximize accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best Optuna Parameters: {best_params}")

    # ==================================================
    # FINAL TRAINING
    # ==================================================

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return model, preds, X_test.index