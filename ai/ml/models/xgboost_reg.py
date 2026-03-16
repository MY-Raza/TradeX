from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import optuna

from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.config_loader import get_logger 

logger = get_logger("xgboost_regressor")

# Suppress Optuna verbose logging
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
    Train an XGBRegressor on OHLCV data using Optuna for hyperparameter
    optimization and a PnL-based backtest objective with MedianPruner.

    Features (open/high/low/close/volume) are optionally log-diff transformed
    for numeric stability. The target column remains in absolute price units.

    Returns:
        final_model  (XGBRegressor): Best model re-trained on the full train set.
        final_preds  (np.ndarray):   Predictions on the test set in absolute price units.
        test_index   (pd.Index):     Row indices of the test set.
        X_test       (pd.DataFrame): Feature matrix of the test set.
    """

    # ----------------------------
    # 1. Validate & pre-process
    # ----------------------------
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    logger.info(f"[train] DataFrame shape before transform: {df.shape}")

    # ----------------------------
    # 2. Feature transformation (target excluded)
    # ----------------------------
    if transform_features:
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_cols:
            df[col] = np.log(df[col]).diff()
        if "volume" in df.columns:
            df["volume"] = np.log1p(df["volume"]).diff()
        df = df.dropna(subset=price_cols).reset_index(drop=True)
        logger.info(f"[train] DataFrame shape after transform & dropna: {df.shape}")

    # ----------------------------
    # 3. Train/Test Split
    # ----------------------------
    split_dt = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_dt].copy()
    test_df  = df[df["datetime"] >= split_dt].copy()

    if train_df.empty:
        raise ValueError(f"Train set is empty for split_date='{split_date}'")
    if test_df.empty:
        raise ValueError(f"Test set is empty for split_date='{split_date}'")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df[target_col]

    logger.info(f"[train] Train rows: {len(X_train)}, Test rows: {len(X_test)}, Features: {X_train.shape[1]}")

    # ----------------------------
    # 4. Optuna objective (PnL + pruning)
    # ----------------------------
    def objective(trial: optuna.Trial) -> float:
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
            "random_state": 42,
        }

        logger.info(f"[trial {trial.number}] Params: {params}", )

        model = XGBRegressor(**params, n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if np.std(preds) < 1e-8:
            logger.info(f"[trial {trial.number}] Skipping trial: constant predictions detected.")
            raise optuna.TrialPruned()

        df_preds = prepare_predictions(
            df,
            preds,
            X_test.index,
            model_type="regressor",
            threshold=None,
            k=k
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        # Chunked backtest
        n_chunks   = 5
        total_rows = len(df_preds)
        pnl_so_far = 0.0

        for chunk_idx in range(n_chunks):
            end_idx = (chunk_idx + 1) * total_rows // n_chunks
            df_chunk = df_preds.iloc[:end_idx]

            bt = BackTest(df_1m, df_chunk, take_profit=3, stop_loss=1)
            _, _, pnl_so_far = bt.run()

            trial.report(pnl_so_far, step=chunk_idx)
            logger.info(f"[trial {trial.number}] Chunk {chunk_idx+1}/{n_chunks} PnL: {pnl_so_far:.4f}", )
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far

    # ----------------------------
    # 5. Run Optuna study
    # ----------------------------
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"[train] Best Optuna parameters: {best_params}")
    logger.info(f"[train] Best trial PnL: {study.best_value:.4f}")

    # ----------------------------
    # 6. Final model (full train)
    # ----------------------------
    final_model = XGBRegressor(**best_params, n_jobs=-1)
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)  # absolute price predictions

    logger.info(f"[train] Final predictions — min: {final_preds.min():.4f}, max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}")

    return final_model, final_preds, X_test.index, X_test