from __future__ import annotations

import numpy as np
import pandas as pd

from TradeX.utils.common.logs import get_logger

# ScalerGuard lives inside the feature_pipeline module; import from there
# so we use the canonical, documented implementation.
from TradeX.sentiments.ml.feature_pipeline import ScalerGuard

from TradeX.sentiments.ml.config import (
    ALL_FEATURES,
    TARGET_CLASS_COL,
    TARGET_RETURN_COL,
    DATETIME_COL,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = get_logger("preprocessing")


# =========================================================
# SPLIT
# =========================================================

def split_data_timewise(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train / val / test without shuffling.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame, sorted ascending by DATETIME_COL.

    Returns
    -------
    df_train, df_val, df_test : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    n = len(df)
    if n < 10:
        raise ValueError(f"Dataset too small to split (n={n}).")

    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    df_train = df.iloc[:train_end].copy().reset_index(drop=True)
    df_val   = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    df_test  = df.iloc[val_end:].copy().reset_index(drop=True)

    logger.info(
        f"Dataset split (n={n}) → "
        f"train: {len(df_train)} ({TRAIN_RATIO:.0%}) | "
        f"val: {len(df_val)} ({VAL_RATIO:.0%}) | "
        f"test: {len(df_test)} ({1 - TRAIN_RATIO - VAL_RATIO:.0%})"
    )
    logger.info(f"  Train range: {df_train[DATETIME_COL].min()} → {df_train[DATETIME_COL].max()}")
    logger.info(f"  Val   range: {df_val[DATETIME_COL].min()}   → {df_val[DATETIME_COL].max()}")
    logger.info(f"  Test  range: {df_test[DATETIME_COL].min()}  → {df_test[DATETIME_COL].max()}")

    # Log class distribution
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = split[TARGET_CLASS_COL].value_counts(normalize=True).sort_index()
        logger.info(
            f"  {name} class dist — "
            + " | ".join(f"class {k}: {v:.1%}" for k, v in dist.items())
        )

    return df_train, df_val, df_test


# =========================================================
# FEATURE PREPARATION
# =========================================================

class PreparedData:
    """
    Container for scaled feature arrays and target arrays.
    Keeps datetime indices so signals can be time-aligned later.
    """

    def __init__(
        self,
        X_train:    np.ndarray,
        X_val:      np.ndarray,
        X_test:     np.ndarray,
        y_class_train: np.ndarray,
        y_class_val:   np.ndarray,
        y_class_test:  np.ndarray,
        y_reg_train:   np.ndarray,
        y_reg_val:     np.ndarray,
        y_reg_test:    np.ndarray,
        dt_train:   pd.Series,
        dt_val:     pd.Series,
        dt_test:    pd.Series,
        scaler:     ScalerGuard,
        feature_cols: list[str],
    ):
        self.X_train = X_train
        self.X_val   = X_val
        self.X_test  = X_test

        self.y_class_train = y_class_train
        self.y_class_val   = y_class_val
        self.y_class_test  = y_class_test

        self.y_reg_train = y_reg_train
        self.y_reg_val   = y_reg_val
        self.y_reg_test  = y_reg_test

        self.dt_train = dt_train
        self.dt_val   = dt_val
        self.dt_test  = dt_test

        self.scaler       = scaler
        self.feature_cols = feature_cols


def prepare_features(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> PreparedData:
    """
    Extract feature matrices and target vectors, then scale with ScalerGuard.

    Scaling is fit ONLY on df_train.  Val and test are transformed only.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Time-split DataFrames from split_data_timewise().

    Returns
    -------
    PreparedData
        Holds X_*, y_class_*, y_reg_*, dt_*, scaler, feature_cols.
    """
    # ── Determine available features ─────────────────────────
    feature_cols = [c for c in ALL_FEATURES if c in df_train.columns]
    missing = set(ALL_FEATURES) - set(feature_cols)
    if missing:
        logger.warning(f"Features absent from DataFrame (skipped): {sorted(missing)}")

    logger.info(f"Using {len(feature_cols)} features for modelling.")

    # ── Extract raw arrays ────────────────────────────────────
    def _extract(df: pd.DataFrame):
        X       = df[feature_cols]
        y_class = df[TARGET_CLASS_COL].values.astype(int)
        y_reg   = df[TARGET_RETURN_COL].values.astype(float)
        dt      = df[DATETIME_COL].reset_index(drop=True)
        return X, y_class, y_reg, dt

    X_train_raw, y_class_train, y_reg_train, dt_train = _extract(df_train)
    X_val_raw,   y_class_val,   y_reg_val,   dt_val   = _extract(df_val)
    X_test_raw,  y_class_test,  y_reg_test,  dt_test  = _extract(df_test)

    # ── Scale — FIT on train ONLY ─────────────────────────────
    guard = ScalerGuard(feature_cols=feature_cols)

    X_train = guard.fit_transform(X_train_raw)   # fit + transform
    X_val   = guard.transform(X_val_raw)          # transform only
    X_test  = guard.transform(X_test_raw)         # transform only

    logger.info(
        f"Scaling complete → "
        f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}"
    )

    return PreparedData(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_class_train=y_class_train, y_class_val=y_class_val, y_class_test=y_class_test,
        y_reg_train=y_reg_train,     y_reg_val=y_reg_val,     y_reg_test=y_reg_test,
        dt_train=dt_train, dt_val=dt_val, dt_test=dt_test,
        scaler=guard,
        feature_cols=feature_cols,
    )