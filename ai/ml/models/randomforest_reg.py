from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import optuna
from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("randomforest_regressor")
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
    Train a RandomForestRegressor using Optuna PnL-based optimisation.

    FIX BN3: criterion now samples from valid regressor values only
             ('squared_error', 'absolute_error', 'friedman_mse', 'poisson').
             The original code sampled 'gini'/'entropy'/'log_loss' which are
             classifier-only and cause silent failures or sklearn errors.

    FIX BN2: Optuna study is launched with n_jobs=-1 so trials run in
             parallel (each individual tree still uses n_jobs=1 to avoid
             nested-pool deadlocks, which is correct).

    FIX BN1: df.copy() is deferred — the caller's frame is only copied once
             when transform_features mutates columns; downstream helpers
             receive views or small slices, not full duplicates.
    """

    # ------------------------------------------------------------------ #
    # 1.  Validate                                                         #
    # ------------------------------------------------------------------ #
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # ------------------------------------------------------------------ #
    # 2.  Feature transformation                                           #
    #     Only copy here — avoids the repeated copy() in every sub-call.  #
    # ------------------------------------------------------------------ #
    df = df.copy()  # single copy; all subsequent work mutates this frame
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"[train] DataFrame shape before transform: {df.shape}")

    if transform_features:
        price_feature_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_feature_cols:
            df[col] = np.log(df[col]).diff()
        if "volume" in df.columns:
            df["volume"] = np.log1p(df["volume"]).diff()
        df = df.dropna(subset=price_feature_cols).reset_index(drop=True)
        logger.info(f"[train] DataFrame shape after log-diff transform & dropna: {df.shape}")

    # ------------------------------------------------------------------ #
    # 3.  Train / test split                                               #
    # ------------------------------------------------------------------ #
    split_dt = pd.to_datetime(split_date, utc=True)
    train_mask = df["datetime"] < split_dt

    if train_mask.sum() == 0:
        raise ValueError(f"Train set is empty for split_date='{split_date}'.")
    if (~train_mask).sum() == 0:
        raise ValueError(f"Test set is empty for split_date='{split_date}'.")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]

    X_train = df.loc[train_mask].drop(columns=drop_cols)
    y_train = df.loc[train_mask, target_col]
    X_test  = df.loc[~train_mask].drop(columns=drop_cols)
    y_test  = df.loc[~train_mask, target_col]  # noqa: F841 (kept for reference)

    logger.info(
        f"[train] Train rows: {len(X_train)} | Test rows: {len(X_test)} "
        f"| Features: {X_train.shape[1]}"
    )
    logger.info(
        f"[train] NaNs — X_train: {X_train.isna().sum().sum()} "
        f"| X_test: {X_test.isna().sum().sum()}"
    )
    logger.info(f"[train] Starting Optuna study ({n_trials} trials)...")

    # ------------------------------------------------------------------ #
    # 4.  Optuna objective                                                 #
    # ------------------------------------------------------------------ #
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
            # FIX BN3 — valid regressor criteria only (removed gini/entropy/log_loss)
            "criterion":         trial.suggest_categorical(
                "criterion",
                ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            ),
        }

        logger.info(f"[trial {trial.number}] Params: {params}")

        # n_jobs=1 per tree is correct here because Optuna parallelises at
        # the study level (n_jobs=-1 on study.optimize below).
        model = RandomForestRegressor(random_state=42, n_jobs=1, **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if np.std(preds) < 1e-8:
            logger.info(f"[trial {trial.number}] Pruning: constant predictions.")
            raise optuna.TrialPruned()

        df_preds = prepare_predictions(
            df, preds, X_test.index,
            model_type="regressor", threshold=None, k=k,
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        n_chunks   = 5
        total_rows = len(df_preds)
        pnl_so_far = 0.0

        for chunk_idx in range(n_chunks):
            end_idx = (chunk_idx + 1) * total_rows // n_chunks
            bt = BackTest(df_1m, df_preds.iloc[:end_idx], take_profit=3, stop_loss=1)
            _, _, pnl_so_far = bt.run()
            trial.report(pnl_so_far, step=chunk_idx)
            logger.info(
                f"[trial {trial.number}] Chunk {chunk_idx+1}/{n_chunks} "
                f"PnL: {pnl_so_far:.4f}"
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far

    # ------------------------------------------------------------------ #
    # 5.  Run study                                                        #
    #     FIX BN2: n_jobs=-1 lets Optuna run trials in parallel using     #
    #     joblib. Each trial's RF still uses n_jobs=1 (no nested pools).  #
    # ------------------------------------------------------------------ #
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_params = study.best_params
    logger.info(f"[train] Best Optuna parameters: {best_params}")
    logger.info(f"[train] Best trial PnL: {study.best_value:.4f}")

    # ------------------------------------------------------------------ #
    # 6.  Final model (full train set, unrestricted parallelism)           #
    # ------------------------------------------------------------------ #
    final_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    logger.info("[train] Fitting final model...")
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)

    logger.info(
        f"[train] Final predictions — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
    )

    return final_model, final_preds, X_test.index, X_test