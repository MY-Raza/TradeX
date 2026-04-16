from __future__ import annotations

import pandas as pd
from TradeX.utils.db.utils import read_df_from_db, fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger

from TradeX.sentiments.ml.config import (
    DB_SCHEMA_FEATURES,
    DB_SCHEMA_PRICE,
    FEATURES_TABLE,
    PRICE_TABLE,
    PRICE_TIME_COLUMN,
    DATETIME_COL,
    ALL_FEATURES,
    TARGET_CLASS_COL,
    TARGET_RETURN_COL,
)

logger = get_logger("data_loader")


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _ensure_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse a datetime column to UTC-aware timestamps in-place."""
    df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _sort_and_dedup(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Sort ascending by time_col and drop exact-duplicate timestamps."""
    df = df.sort_values(time_col).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=[time_col]).reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before - after} duplicate rows on '{time_col}'")
    return df


# =========================================================
# PUBLIC API
# =========================================================

def load_features_from_db() -> pd.DataFrame:
    """
    Load the pre-engineered ML feature table from the database.

    Returns
    -------
    pd.DataFrame
        Columns: [datetime] + ALL_FEATURES + [target, target_return]
        Sorted ascending, deduplicated, missing feature values zero-filled.

    Raises
    ------
    ValueError
        If the table is empty or required columns are missing.
    """
    logger.info(f"Loading features from {DB_SCHEMA_FEATURES}.{FEATURES_TABLE} …")

    df = read_df_from_db(FEATURES_TABLE, DB_SCHEMA_FEATURES)

    if df.empty:
        raise ValueError(
            f"Feature table '{DB_SCHEMA_FEATURES}.{FEATURES_TABLE}' is empty. "
            "Run feature_pipeline.py first."
        )

    # ── Datetime ──────────────────────────────────────────────
    if DATETIME_COL not in df.columns:
        raise ValueError(f"Expected column '{DATETIME_COL}' not found in feature table.")

    df = _ensure_utc(df, DATETIME_COL)
    df = _sort_and_dedup(df, DATETIME_COL)

    # ── Validate required columns ─────────────────────────────
    required = ALL_FEATURES + [TARGET_CLASS_COL, TARGET_RETURN_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            f"{len(missing)} expected column(s) missing from feature table — "
            f"will be zero-filled: {missing}"
        )
        for col in missing:
            df[col] = 0.0

    # ── Fill NaNs in feature columns only ────────────────────
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)

    # ── Drop rows where target itself is NaN (last bar shift) ─
    n_before = len(df)
    df = df.dropna(subset=[TARGET_CLASS_COL, TARGET_RETURN_COL]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} row(s) with NaN targets (expected at tail).")

    logger.info(
        f"Features loaded → shape: {df.shape}  |  "
        f"date range: {df[DATETIME_COL].min()} → {df[DATETIME_COL].max()}"
    )
    return df


def load_price_data(
    start_date: str | pd.Timestamp | None = None,
    end_date:   str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Fetch minute-level OHLCV price data for the backtest engine.

    Parameters
    ----------
    start_date : str or pd.Timestamp, optional
        Lower bound (inclusive).  E.g. "2024-01-01" or a pd.Timestamp.
    end_date : str or pd.Timestamp, optional
        Upper bound (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns include at minimum: datetime, open, high, low, close, volume.
        Sorted ascending, UTC-aware datetime column.

    Raises
    ------
    ValueError
        If fetch returns an empty DataFrame.
    """
    logger.info(
        f"Loading OHLCV data from {DB_SCHEMA_PRICE}.{PRICE_TABLE} "
        f"[{start_date} → {end_date}] …"
    )

    df = fetch_ohlcv_df(
        table_name=PRICE_TABLE,
        schema=DB_SCHEMA_PRICE,
        time_column=PRICE_TIME_COLUMN,
        start_date=str(start_date) if start_date is not None else None,
        end_date=str(end_date)     if end_date   is not None else None,
    )

    if df.empty:
        raise ValueError(
            f"OHLCV table '{DB_SCHEMA_PRICE}.{PRICE_TABLE}' returned 0 rows "
            f"for range [{start_date} → {end_date}]."
        )

    df = _ensure_utc(df, PRICE_TIME_COLUMN)
    df = _sort_and_dedup(df, PRICE_TIME_COLUMN)

    # Rename to 'datetime' if the column is named differently
    if PRICE_TIME_COLUMN != DATETIME_COL:
        df = df.rename(columns={PRICE_TIME_COLUMN: DATETIME_COL})

    required_ohlcv = ["open", "high", "low", "close"]
    missing = [c for c in required_ohlcv if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {missing}")

    logger.info(
        f"OHLCV loaded → shape: {df.shape}  |  "
        f"date range: {df[DATETIME_COL].min()} → {df[DATETIME_COL].max()}"
    )
    return df