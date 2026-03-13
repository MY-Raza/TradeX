from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import optuna
from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("randomforest_regressor")

# Suppress Optuna's verbose per-trial logging
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
    Train a RandomForestRegressor on OHLCV data using Optuna for hyperparameter
    optimization and a PnL-based backtest objective with MedianPruner.

    Features (open/high/low/close/volume) are optionally log-diff transformed for
    numeric stability. The target column is intentionally left untransformed so that
    model predictions are in absolute price units (e.g. 33 358.6) rather than
    log-return scale.

    Args:
        df               (pd.DataFrame): Feature dataframe; must contain a 'datetime' column.
        df_1m            (pd.DataFrame): 1-minute OHLCV dataframe used by the backtester.
        target_col       (str):          Name of the target column (absolute price).
        split_date       (str):          ISO date string that separates train from test.
        n_trials         (int):          Number of Optuna optimisation trials.
        k                (float):        Threshold multiplier for converting regressor
                                         output to long/short/flat signals.
        transform_features (bool):       When True, apply log-diff to OHLCV feature
                                         columns for numeric stability.

    Returns:
        final_model  (RandomForestRegressor): Best model re-trained on the full train set.
        final_preds  (np.ndarray):            Predictions on the test set in absolute price units.
        test_index   (pd.Index):              Row indices of the test set.
        X_test       (pd.DataFrame):         Feature matrix of the test set.

    Raises:
        ValueError: If 'datetime' column is missing, or the split yields an empty partition.
    """

    # ------------------------------------------------------------------ #
    # 1.  Validate & pre-process                                           #
    # ------------------------------------------------------------------ #
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"[train] DataFrame shape before transform: {df.shape}")

    # ------------------------------------------------------------------ #
    # 2.  Feature transformation (target is excluded)                      #
    # ------------------------------------------------------------------ #
    if transform_features:
        price_feature_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        for col in price_feature_cols:
            df[col] = np.log(df[col]).diff()

        if "volume" in df.columns:
            df["volume"] = np.log1p(df["volume"]).diff()

        # Drop rows introduced by differencing (first row per series becomes NaN)
        df = df.dropna(subset=price_feature_cols).reset_index(drop=True)
        logger.info(f"[train] DataFrame shape after log-diff transform & dropna: {df.shape}")

    # ------------------------------------------------------------------ #
    # 3.  Time-based train / test split                                    #
    # ------------------------------------------------------------------ #
    split_dt  = pd.to_datetime(split_date, utc=True)
    train_df  = df[df["datetime"] < split_dt].copy()
    test_df   = df[df["datetime"] >= split_dt].copy()

    if train_df.empty:
        raise ValueError(f"Train set is empty for split_date='{split_date}'. Adjust the split date.")
    if test_df.empty:
        raise ValueError(f"Test set is empty for split_date='{split_date}'. Adjust the split date.")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]          # absolute price units

    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df[target_col]           # kept for reference / evaluation

    logger.info(f"[train] Train rows: {len(X_train)} | Test rows: {len(X_test)} | Features: {X_train.shape[1]}", )
    logger.info(f"[train] NaNs in X_train: {X_train.isna().sum().sum()} | X_test: {X_test.isna().sum().sum()}", )
    logger.info(f"[train] NaNs in y_train: {y_train.isna().sum()} | y_test: {y_test.isna().sum()}", )
    logger.info(f"[train] Starting Optuna study ({n_trials} trials)...", )

    # ------------------------------------------------------------------ #
    # 4.  Optuna objective — maximise PnL with chunked pruning             #
    # ------------------------------------------------------------------ #
    def objective(trial: optuna.Trial) -> float:
        params = {
            # Cap n_estimators to 300 — beyond this, gains are marginal but
            # fit time grows linearly, causing silent hangs with n_jobs=-1
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        }

        logger.info(f"[trial {trial.number}] Params: {params}", )

        # Use n_jobs=1 inside Optuna trials — Optuna already parallelises
        # trials at the study level; nested joblib pools can deadlock
        model = RandomForestRegressor(random_state=42, n_jobs=1, **params)

        logger.info(f"[trial {trial.number}] Fitting model on {len(X_train)} rows...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)   # absolute price predictions

        # Guard against constant predictions (degenerate model)
        logger.info(f"[trial {trial.number}] Fit done. Running backtest chunks...")

        if np.std(preds) < 1e-8:
            logger.info(f"[trial {trial.number}] Skipping: constant predictions detected.")
            raise optuna.TrialPruned()

        df_preds = prepare_predictions(
            df,
            preds,
            X_test.index,
            model_type="regressor",
            threshold=None,
            k=k,
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        # Chunked backtest for intermediate Optuna reporting / pruning
        n_chunks   = 5
        total_rows = len(df_preds)
        pnl_so_far = 0.0

        for chunk_idx in range(n_chunks):
            end_idx    = (chunk_idx + 1) * total_rows // n_chunks
            df_chunk   = df_preds.iloc[:end_idx]

            bt = BackTest(df_1m, df_chunk, take_profit=3, stop_loss=1)
            _, _, pnl_so_far = bt.run()

            trial.report(pnl_so_far, step=chunk_idx)
            logger.info(f"[trial {trial.number}] Chunk {chunk_idx+1}/{n_chunks} PnL: {pnl_so_far:.4f}")
            if trial.should_prune():
                raise optuna.TrialPruned()

        return pnl_so_far

    # ------------------------------------------------------------------ #
    # 5.  Run Optuna study                                                 #
    # ------------------------------------------------------------------ #
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"[train] Best Optuna parameters: {best_params}")
    logger.info(f"[train] Best trial PnL: {study.best_value:.4f}")

    # ------------------------------------------------------------------ #
    # 6.  Final model — re-train on full train set with best params        #
    # ------------------------------------------------------------------ #
    # n_jobs=-1 is safe here since we're outside Optuna's parallel context
    final_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    logger.info("[train] Fitting final model...", )
    final_model.fit(X_train, y_train)
    final_preds = final_model.predict(X_test)   # absolute price units

    logger.info(f"[train] Final predictions — min: {final_preds.min():.4f}, "
          f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}")

    return final_model, final_preds, X_test.index, X_test