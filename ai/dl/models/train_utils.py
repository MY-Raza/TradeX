from __future__ import annotations

import numpy as np
import pandas as pd
import optuna

from TradeX.backtest.backtest import BackTest
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.utils.common.logs import get_logger

logger = get_logger("dl_train_utils")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Validate & sort  (identical to ML train_utils)
# ──────────────────────────────────────────────────────────────────────────────

def validate_and_sort(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Validate required columns exist and sort the DataFrame by datetime (UTC).

    Args:
        df         : Raw feature DataFrame.
        target_col : Name of the target column (e.g. ``'target'``).

    Returns:
        DataFrame with ``'datetime'`` cast to UTC-aware Timestamps, sorted.

    Raises:
        ValueError : If ``'datetime'`` or ``target_col`` are missing.
    """
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature transformation  (identical to ML train_utils)
# ──────────────────────────────────────────────────────────────────────────────

def apply_log_diff_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log-differencing to OHLCV columns and sanitise inf/NaN.

    Transformation:
        - open, high, low, close  →  log(x).diff()
        - volume                  →  log1p(x).diff()

    After transformation all numeric columns are checked for inf / NaN and
    offending rows are dropped so downstream models never receive bad values.

    Args:
        df : DataFrame containing any subset of OHLCV columns.

    Returns:
        Transformed DataFrame with NaN/inf rows removed and index reset.
    """
    df = df.copy()

    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    for col in price_cols:
        df[col] = np.log(df[col]).diff()
    if "volume" in df.columns:
        df["volume"] = np.log1p(df["volume"]).diff()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    rows_before  = len(df)
    df           = df.dropna(subset=numeric_cols).reset_index(drop=True)
    rows_dropped = rows_before - len(df)

    if rows_dropped:
        logger.warning(
            f"[apply_log_diff_transform] Dropped {rows_dropped} rows with inf/NaN."
        )

    logger.info(f"[apply_log_diff_transform] Shape after transform: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Train / test split  (identical to ML train_utils)
# ──────────────────────────────────────────────────────────────────────────────

def split_features_labels(
    df: pd.DataFrame,
    target_col: str,
    split_date: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Temporal train/test split and X/y construction.

    Drops ``'datetime'``, ``target_col``, and ``'future_close'`` from feature
    matrices.  A final defensive pass replaces remaining inf/NaN and drops
    those rows.

    Args:
        df          : Sorted, transformed DataFrame.
        target_col  : Name of the label column.
        split_date  : ISO date string marking the train/test boundary.

    Returns:
        ``(X_train, y_train, X_test, y_test)``

    Raises:
        ValueError : If either split yields an empty set.
    """
    split_dt = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_dt]
    test_df  = df[df["datetime"] >= split_dt]

    if train_df.empty:
        raise ValueError(f"Train set is empty for split_date='{split_date}'.")
    if test_df.empty:
        raise ValueError(f"Test set is empty for split_date='{split_date}'.")

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df.columns]
    X_train   = train_df.drop(columns=drop_cols)
    y_train   = train_df[target_col]
    X_test    = test_df.drop(columns=drop_cols)
    y_test    = test_df[target_col]

    # Defensive sanitisation
    for name, X, y in (("X_train", X_train, y_train), ("X_test", X_test, y_test)):
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        bad_mask = np.isinf(X[num_cols]).any(axis=1) | X[num_cols].isna().any(axis=1)
        if bad_mask.any():
            logger.warning(
                f"[split_features_labels] {name}: dropping {bad_mask.sum()} rows "
                "with inf/NaN."
            )
            X.drop(index=X.index[bad_mask], inplace=True)
            if name == "X_train":
                y_train = y_train.loc[X_train.index]
            else:
                y_test = y_test.loc[X_test.index]

    logger.info(
        f"[split_features_labels] Train rows: {len(X_train)} | "
        f"Test rows: {len(X_test)} | Features: {X_train.shape[1]}"
    )
    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 4. Chunked backtest with Optuna pruning  (identical to ML train_utils)
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
    Evaluate predictions via chunked PnL backtesting with Optuna pruning.

    Identical to the ML module's ``run_chunked_backtest`` — DL models are
    evaluated the same way as RF / XGB models, using ``model_type='dl'`` so
    ``prepare_predictions`` applies the correct signal-generation path.

    Args:
        trial        : Active Optuna trial.
        df           : Full feature DataFrame (with ``'datetime'`` column).
        preds        : Model predictions for the test set.
        test_index   : Index of test rows (after seq_len warm-up alignment).
        df_1m        : 1-minute OHLCV data for BackTest.
        model_type   : ``'classifier'`` or ``'regressor'``.
        k            : Top-k std threshold for signal selection.
        n_chunks     : Number of temporal evaluation chunks.
        take_profit  : BackTest take-profit multiplier.
        stop_loss    : BackTest stop-loss multiplier.

    Returns:
        Final cumulative PnL across all chunks.

    Raises:
        optuna.TrialPruned : If Optuna decides to prune mid-evaluation.
    """
    # DL predictions use the 'dl' branch in prepare_predictions which handles
    # the seq_len warm-up offset automatically via lookback=0 (already aligned)
    preds_np   = np.asarray(preds)
    idx_arr    = np.asarray(test_index)

    # Guard: preds may be shorter than test_index by seq_len-1 (DL warm-up).
    # Always trim both to the shorter length before calling prepare_predictions
    # so the datetime and signal arrays are always the same size.
    min_len    = min(len(preds_np), len(idx_arr))
    preds_np   = preds_np[-min_len:]
    idx_arr    = idx_arr[-min_len:]

    df_preds = prepare_predictions(
        df, preds_np, idx_arr,
        model_type=model_type,
        k=k,
    )
    df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

    total_rows  = len(df_preds)
    pnl_so_far  = 0.0

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