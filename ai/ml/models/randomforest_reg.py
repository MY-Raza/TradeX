from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import optuna


def train(df, target_col="target", split_date="2024-01-01 00:00", n_trials=50):
    """
    Train RandomForestRegressor using time-based split with Optuna hyperparameter optimization.

    Args:
        df (pd.DataFrame): Input dataframe with features and target
        target_col (str): Name of the target column
        split_date (str): Date string to split train/test
        n_trials (int): Number of Optuna trials for hyperparameter search
        **model_params: any extra sklearn RandomForestRegressor params to override

    Returns:
        model: trained RandomForestRegressor
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
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse  # minimize RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best Optuna Parameters: {best_params}")

    # ==================================================
    # FINAL TRAINING
    # ==================================================

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return model, preds, X_test.index