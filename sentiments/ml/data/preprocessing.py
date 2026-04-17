from __future__ import annotations

import numpy as np
import pandas as pd

from TradeX.utils.common.logs import get_logger

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
# OHLCV + INDICATOR MERGE
# =========================================================

def merge_ohlcv_indicators_with_features(
    df_features:    pd.DataFrame,
    df_ohlcv_indic: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Align and merge OHLCV+indicator data with the hourly ml_features table.

    The ml_features table is hourly; OHLCV data is minute-level.
    Strategy: resample OHLCV+indicators to hourly (last value of each bar)
    then left-join onto df_features on the datetime key.

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of load_features_from_db().  Hourly, UTC-aware.
    df_ohlcv_indic : pd.DataFrame
        Output of load_ohlcv_with_indicators().  Minute-level, UTC-aware.
        Columns: datetime, open, high, low, close, volume, ta_*

    Returns
    -------
    df_merged : pd.DataFrame
        df_features rows with indicator columns appended.
        Any rows where indicator data is unavailable are zero-filled.
    indicator_cols : list[str]
        Names of the newly added indicator columns (all ta_* columns plus
        the raw OHLCV columns that were resampled).
    """
    if df_ohlcv_indic.empty:
        logger.warning("df_ohlcv_indic is empty — skipping indicator merge.")
        return df_features, []

    logger.info("Merging OHLCV+indicators with ml_features …")

    # ── 1. Identify indicator columns (ta_* + raw OHLCV) ─────
    ohlcv_raw_cols = ["open", "high", "low", "close", "volume"]
    ta_cols        = [c for c in df_ohlcv_indic.columns
                      if c.startswith("ta_")]
    indicator_cols = ohlcv_raw_cols + ta_cols

    # ── 2. Resample from minute → hourly ─────────────────────
    # For price/indicator data: take the LAST value in each hour
    # (matches the convention of using end-of-bar values as features).
    df_indic = df_ohlcv_indic[["datetime"] + indicator_cols].copy()
    df_indic = df_indic.set_index("datetime")
    df_indic.index = pd.to_datetime(df_indic.index, utc=True)

    # Resample numeric columns only
    df_hourly = (
        df_indic
        .resample("1h", closed="right", label="right")
        .last()
        .reset_index()
    )

    # Ensure UTC-aware after resample
    df_hourly["datetime"] = pd.to_datetime(df_hourly["datetime"], utc=True)

    logger.info(
        f"  Resampled OHLCV+indicators: {len(df_ohlcv_indic)} min-bars "
        f"→ {len(df_hourly)} hourly bars."
    )

    # ── 3. Left-join onto ml_features (keep all feature rows) ─
    df_features = df_features.copy()
    df_features["datetime"] = pd.to_datetime(df_features["datetime"], utc=True)

    df_merged = df_features.merge(
        df_hourly,
        on="datetime",
        how="left",
        suffixes=("", "_ohlcv"),   # avoid col name clashes
    )

    # ── 4. Resolve column conflicts (prefer original ml_features values) ──
    # Columns like 'returns', 'volume_change' exist in both — keep original.
    conflict_cols = [c for c in df_merged.columns if c.endswith("_ohlcv")]
    if conflict_cols:
        logger.info(
            f"  Dropping {len(conflict_cols)} conflicting OHLCV suffix columns: "
            f"{conflict_cols}"
        )
        df_merged = df_merged.drop(columns=conflict_cols)

    # ── 5. Zero-fill any gaps from the join ──────────────────
    added_cols = [c for c in indicator_cols if c in df_merged.columns]
    n_nulls    = df_merged[added_cols].isnull().sum().sum()
    if n_nulls:
        logger.warning(
            f"  {n_nulls} NaN cells after merge (hour gaps in OHLCV?) — zero-filled."
        )
        df_merged[added_cols] = df_merged[added_cols].fillna(0.0)

    n_missing_rows = df_merged[added_cols].isnull().any(axis=1).sum()
    logger.info(
        f"  Merge complete → df_merged shape: {df_merged.shape}  |  "
        f"indicator cols added: {len(added_cols)}  |  "
        f"rows with any NaN after fill: {n_missing_rows}"
    )

    return df_merged, added_cols


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
    extra_feature_cols: list[str] | None = None,
) -> PreparedData:
    """
    Extract feature matrices and target vectors, then scale with ScalerGuard.

    Scaling is fit ONLY on df_train.  Val and test are transformed only.

    The feature set is:  ALL_FEATURES  +  extra_feature_cols (indicator
    columns added by merge_ohlcv_indicators_with_features).  Only columns
    that are actually present in the DataFrames are used, so the function
    degrades gracefully if indicator columns are absent.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Time-split DataFrames from split_data_timewise().  May contain extra
        indicator columns appended by merge_ohlcv_indicators_with_features().
    extra_feature_cols : list[str], optional
        Indicator column names returned by merge_ohlcv_indicators_with_features().
        If None, only ALL_FEATURES are used (original behaviour).

    Returns
    -------
    PreparedData
        Holds X_*, y_class_*, y_reg_*, dt_*, scaler, feature_cols.
    """
    # ── Determine full feature set ────────────────────────────
    candidate_cols = list(ALL_FEATURES)
    if extra_feature_cols:
        # Append indicator cols that are not already in ALL_FEATURES
        for col in extra_feature_cols:
            if col not in candidate_cols:
                candidate_cols.append(col)

    # Keep only columns that exist in all three splits
    feature_cols = [
        c for c in candidate_cols
        if c in df_train.columns and c in df_val.columns and c in df_test.columns
    ]

    missing_from_config = set(ALL_FEATURES) - set(feature_cols)
    if missing_from_config:
        logger.warning(
            f"Features absent from DataFrame (skipped): {sorted(missing_from_config)}"
        )

    n_indicator_cols = len(feature_cols) - len(
        [c for c in feature_cols if c in ALL_FEATURES]
    )
    logger.info(
        f"Using {len(feature_cols)} features for modelling "
        f"({len(ALL_FEATURES)} sentiment/market/alpha + "
        f"{n_indicator_cols} OHLCV/indicator cols)."
    )

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

    X_train = guard.fit_transform(X_train_raw)
    X_val   = guard.transform(X_val_raw)
    X_test  = guard.transform(X_test_raw)

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