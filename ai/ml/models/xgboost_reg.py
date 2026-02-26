from xgboost import XGBRegressor
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
    k=0.5,
    pnl_drop_threshold=7.0  # % PnL drop threshold for pruning features
):
    """
    Train XGBRegressor using PnL-based Optuna optimization with data pruning.
    Includes log-diff feature transformation.

    Args:
        df (pd.DataFrame): Feature dataframe
        df_1m (pd.DataFrame): 1-minute OHLCV dataframe for backtesting
        target_col (str): Target column name
        split_date (str): Date string to split train/test
        n_trials (int): Number of Optuna trials
        k (float): Threshold for converting predictions to trading signals
        pnl_drop_threshold (float): Minimum PnL drop (%) to keep a feature

    Returns:
        final_model: trained XGBRegressor
        final_preds: predictions on test set
        X_test.index: indices of test set
        X_test: test features after pruning
    """

    # ----------------------------
    # Ensure datetime column & sort
    # ----------------------------
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have a 'datetime' column.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ----------------------------
    # LOG-DIFF TRANSFORMATION
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
    # OPTUNA OBJECTIVE (PnL Optimization)
    # ----------------------------
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Convert predictions → trading signals
        df_preds = prepare_predictions(
            df,
            preds,
            X_test.index,
            model_type="regressor",
            threshold=None,
            k=k
        )
        df_preds['datetime'] = pd.to_datetime(df_preds['datetime'], utc=True)

        # Run Backtest
        bt = BackTest(df_1m, df_preds, take_profit=3, stop_loss=1)
        _, _, pnl = bt.run()
        return pnl  # maximize profit

    # ----------------------------
    # Run Optuna
    # ----------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best Optuna Parameters: {best_params}")

    # ----------------------------
    # DATA PRUNING BASED ON PNL DROP
    # ----------------------------
    base_model = XGBRegressor(**best_params)
    base_model.fit(X_train, y_train)
    base_preds = base_model.predict(X_test)
    df_base_preds = prepare_predictions(df, base_preds, X_test.index, model_type="regressor", k=k)
    bt_base = BackTest(df_1m, df_base_preds, take_profit=3, stop_loss=1)
    _, _, base_pnl = bt_base.run()

    pnl_drop_list = []
    for col in X_train.columns:
        X_train_temp = X_train.drop(columns=[col])
        X_test_temp = X_test.drop(columns=[col])
        temp_model = XGBRegressor(**best_params)
        temp_model.fit(X_train_temp, y_train)
        preds_temp = temp_model.predict(X_test_temp)
        df_preds_temp = prepare_predictions(df, preds_temp, X_test_temp.index, model_type="regressor", k=k)
        bt_temp = BackTest(df_1m, df_preds_temp, take_profit=3, stop_loss=1)
        _, _, pnl_temp = bt_temp.run()
        pnl_drop_percent = 100 * (base_pnl - pnl_temp) / abs(base_pnl)
        pnl_drop_list.append((col, pnl_drop_percent))

    keep_features = [f for f, drop in pnl_drop_list if drop >= pnl_drop_threshold]
    print("Features kept after data pruning:", keep_features)

    X_train = X_train[keep_features]
    X_test  = X_test[keep_features]

    # ----------------------------
    # FINAL MODEL TRAINING
    # ----------------------------
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)

    return final_model, final_preds, X_test.index, X_test