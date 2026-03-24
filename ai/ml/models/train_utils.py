import numpy as np
import pandas as pd
import optuna

from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.config_loader import get_logger

logger = get_logger("train_utils")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Validate & sort
# ──────────────────────────────────────────────────────────────────────────────

def validate_and_sort(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Validate required columns exist and sort the DataFrame by datetime (UTC).

    Shared by: randomforest_clf, randomforest_reg, xgboost_clf, xgboost_reg

    Args:
        df:         Raw feature DataFrame.
        target_col: Name of the target column (e.g. 'target').

    Returns:
        DataFrame with 'datetime' cast to UTC-aware Timestamps, sorted ascending.

    Raises:
        ValueError: If 'datetime' or target_col are missing.
    """
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

      
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature transformation
# ──────────────────────────────────────────────────────────────────────────────

def apply_log_diff_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log-differencing to OHLCV price/volume columns in-place (on a copy).

    Transformation applied:
        - open, high, low, close  →  log(x).diff()
        - volume                  →  log1p(x).diff()

    Rows with NaN introduced by .diff() are dropped.

    Shared by: randomforest_clf, randomforest_reg, xgboost_clf, xgboost_reg

    Args:
        df: DataFrame containing any subset of OHLCV columns.

    Returns:
        Transformed DataFrame with NaN rows removed and index reset.
    """
      
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]

    for col in price_cols:
        df[col] = np.log(df[col]).diff()
    if "volume" in df.columns:
        df["volume"] = np.log1p(df["volume"]).diff()

    df = df.dropna(subset=price_cols).reset_index(drop=True)
    logger.info(f"[apply_log_diff_transform] Shape after transform & dropna: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Train / test split + feature / label construction
# ──────────────────────────────────────────────────────────────────────────────

def split_features_labels(
    df: pd.DataFrame,
    target_col: str,
    split_date: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Temporal train/test split and X/y construction.

    Drops 'datetime', target_col, and 'future_close' from feature matrices.
    X_train / X_test are built ONCE here so every Optuna trial can reuse them
    via closure — avoiding redundant work inside the objective.

    Shared by: randomforest_clf, randomforest_reg, xgboost_clf, xgboost_reg

    Args:
        df:          Sorted, transformed DataFrame (output of validate_and_sort
                     + apply_log_diff_transform).
        target_col:  Name of the label column.
        split_date:  ISO date string marking the train/test boundary
                     (e.g. '2024-01-01 00:00').

    Returns:
        (X_train, y_train, X_test, y_test)

    Raises:
        ValueError: If either split yields an empty set.
    """
    split_dt = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_dt]  
    test_df  = df[df["datetime"] >= split_dt]  

    if train_df.empty:
        raise ValueError(f"Train set is empty for split_date='{split_date}'. Adjust the split date.")
    if test_df.empty:
        raise ValueError(f"Test set is empty for split_date='{split_date}'. Adjust the split date.")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df[target_col]

    logger.info(
        f"[split_features_labels] Train rows: {len(X_train)} | "
        f"Test rows: {len(X_test)} | Features: {X_train.shape[1]}"
    )
    logger.info(
        f"[split_features_labels] NaNs — "
        f"X_train: {X_train.isna().sum().sum()} | X_test: {X_test.isna().sum().sum()} | "
        f"y_train: {y_train.isna().sum()} | y_test: {y_test.isna().sum()}"
    )

    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 4. Chunked backtest pruning loop
# ──────────────────────────────────────────────────────────────────────────────

def run_chunked_backtest(
    trial: optuna.Trial,
    df: pd.DataFrame,
    preds: np.ndarray,
    test_index: pd.Index,
    df_1m: pd.DataFrame,
    model_type: str,
    k: float,
    n_chunks: int = 2,
    take_profit: float = 3,
    stop_loss: float = 1,
) -> float:
    """
    Evaluate a set of predictions via chunked PnL backtesting with Optuna pruning.

    Each chunk reports an intermediate PnL value to the Optuna trial. If the
    trial should be pruned (score below median of prior trials), TrialPruned is
    raised immediately — avoiding unnecessary backtest work on bad trials.

    Shared by: randomforest_clf, randomforest_reg, xgboost_clf, xgboost_reg

    Args:
        trial:        Active Optuna trial (used for report + should_prune).
        df:           Full feature DataFrame (used by prepare_predictions).
        preds:        Model predictions for the test set (probabilities or values).
        test_index:   Index of the test rows in df.
        df_1m:        1-minute OHLCV data passed to BackTest.
        model_type:   'classifier' or 'regressor' — passed to prepare_predictions.
        k:            Top-k threshold fraction for signal selection.
        n_chunks:     Number of temporal chunks to evaluate (default 2).
        take_profit:  BackTest take-profit multiplier (default 3).
        stop_loss:    BackTest stop-loss multiplier (default 1).

    Returns:
        Final cumulative PnL across all chunks.

    Raises:
        optuna.TrialPruned: If Optuna decides to prune the trial mid-evaluation.
    """
    df_preds = prepare_predictions(
        df, preds, test_index,
        model_type=model_type,
        k=k,
    )
    df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

    total_rows = len(df_preds)
    pnl_so_far = 0.0

    for chunk_idx in range(n_chunks):
        end_idx  = (chunk_idx + 1) * total_rows // n_chunks
        df_chunk = df_preds.iloc[:end_idx]

        bt = BackTest(df_1m, df_chunk, take_profit=take_profit, stop_loss=stop_loss)
        _, _, pnl_so_far = bt.run()

        trial.report(pnl_so_far, step=chunk_idx)
        logger.info(
            f"[trial {trial.number}] Chunk {chunk_idx + 1}/{n_chunks} "
            f"PnL: {pnl_so_far:.4f}"
        )
        if trial.should_prune():
            logger.info(
                f"[trial {trial.number}] Pruned at chunk {chunk_idx + 1}/{n_chunks} "
                f"— PnL: {pnl_so_far:.4f}"
            )
            raise optuna.TrialPruned()

    return pnl_so_far