"""
feature_pipeline.py
====================
Production-ready ML feature engineering pipeline for BTC sentiment + OHLCV data.

Combines:
  - Reddit post sentiment   (reddit.posts_sentiment_hourly)
  - Reddit comment sentiment (reddit.comments_sentiment_hourly)
  - BTC 1-minute OHLCV      (data_binance.btc_1m)

Output:
  - final_features_df  (in-memory)
  - ml_features table  (database)
  - features.csv       (disk)
  - features.parquet   (disk)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from TradeX.utils.db.utils import read_df_from_db, save_df_to_db, fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv

logger = get_logger("feature_pipeline")

# ================================================================================
# CONFIG
# ================================================================================
POSTS_SCHEMA         = "reddit"
POSTS_TABLE          = "posts_sentiment_hourly"

COMMENTS_SCHEMA      = "reddit"
COMMENTS_TABLE       = "comments_sentiment_hourly"

OHLCV_TABLE          = "btc_1m"
OHLCV_SCHEMA         = "data_binance"
OHLCV_TIME_COLUMN    = "datetime"
OHLCV_RESAMPLE_FREQ  = "1h"

ML_FEATURES_TABLE    = "ml_features"
ML_FEATURES_SCHEMA   = "ml_features"

OUTPUT_DIR           = os.path.dirname(os.path.abspath(__file__))
CSV_PATH             = os.path.join(OUTPUT_DIR, "features.csv")
PARQUET_PATH         = os.path.join(OUTPUT_DIR, "features.parquet")

LAG_RANGE            = range(1, 6)     # lags 1–5
EMA_SPAN             = 5
VOL_WINDOW           = 10              # rolling window for volatility / spike detection
SPIKE_MULTIPLIER     = 2.0


# ================================================================================
# STEP 1 — LOAD SENTIMENT DATA
# ================================================================================

def load_sentiment_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hourly-aggregated post and comment sentiment tables.
    Ensures time_window is UTC-aware and data is sorted ascending.
    """
    logger.info("📥 Loading sentiment data…")

    posts_df    = read_df_from_db(POSTS_TABLE,    POSTS_SCHEMA)
    comments_df = read_df_from_db(COMMENTS_TABLE, COMMENTS_SCHEMA)

    for df, label in [(posts_df, "posts"), (comments_df, "comments")]:
        if "time_window" not in df.columns:
            raise ValueError(f"'{label}' table missing 'time_window' column.")

    posts_df["time_window"]    = pd.to_datetime(posts_df["time_window"],    utc=True)
    comments_df["time_window"] = pd.to_datetime(comments_df["time_window"], utc=True)

    posts_df    = posts_df.sort_values("time_window").reset_index(drop=True)
    comments_df = comments_df.sort_values("time_window").reset_index(drop=True)

    logger.info(f"  Posts rows:    {len(posts_df):,}")
    logger.info(f"  Comments rows: {len(comments_df):,}")

    return posts_df, comments_df


# ================================================================================
# STEP 2 — COMPUTE GLOBAL DATE RANGE
# ================================================================================

def compute_date_range(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns (start_date, end_date) spanning the full range of BOTH sentiment tables.
    Both timestamps are UTC-aware.
    """
    posts_min    = posts_df["time_window"].min()
    posts_max    = posts_df["time_window"].max()
    comments_min = comments_df["time_window"].min()
    comments_max = comments_df["time_window"].max()

    start_date = min(posts_min, comments_min)
    end_date   = max(posts_max, comments_max)

    logger.info(f"📅 Global date range → start: {start_date}  |  end: {end_date}")

    return start_date, end_date


# ================================================================================
# STEP 3 — FETCH OHLCV DATA
# ================================================================================

def load_ohlcv(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetches BTC 1-minute OHLCV rows for the computed date range,
    sorted ascending by datetime (UTC-aware).
    """
    logger.info(
        f"📈 Fetching OHLCV [{OHLCV_SCHEMA}.{OHLCV_TABLE}] "
        f"{start_date} → {end_date}"
    )

    ohlcv_df = fetch_ohlcv_df(
        table_name=OHLCV_TABLE,
        schema=OHLCV_SCHEMA,
        time_column=OHLCV_TIME_COLUMN,
        start_date=start_date,
        end_date=end_date,
    )

    if ohlcv_df.empty:
        raise ValueError("OHLCV fetch returned 0 rows — check date range / DB connectivity.")

    ohlcv_df[OHLCV_TIME_COLUMN] = pd.to_datetime(ohlcv_df[OHLCV_TIME_COLUMN], utc=True)
    ohlcv_df = ohlcv_df.sort_values(OHLCV_TIME_COLUMN).reset_index(drop=True)

    logger.info(f"  OHLCV rows fetched: {len(ohlcv_df):,}")

    return ohlcv_df


# ================================================================================
# STEP 4 — SENTIMENT FEATURE ENGINEERING
# ================================================================================

def _safe_volume_col(df: pd.DataFrame) -> str:
    """
    Detect the volume / count column from the aggregated sentiment table.
    Tries common naming conventions.
    """
    candidates = [c for c in df.columns if c.endswith("_count") or c == "count"]
    if candidates:
        return candidates[0]

    # Fallback: first non time_window, non-sentinel column
    fallback = [c for c in df.columns if c != "time_window"][0]
    logger.warning(f"Volume column not found by name; using '{fallback}' as proxy.")
    return fallback


def create_sentiment_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute sentiment feature columns for a single sentiment source.

    Args:
        df     : Hourly sentiment DataFrame (must contain time_window, mean_sentiment,
                 std_sentiment, sentiment_confidence_mean or similar, and a count col).
        prefix : Column prefix, e.g. "post" or "comment".

    Returns:
        DataFrame with engineered feature columns and time_window.
    """
    df = df.copy().sort_values("time_window").reset_index(drop=True)

    # ── Resolve column names defensively ────────────────────────────────────────
    mean_col       = "mean_sentiment"
    std_col        = "std_sentiment"
    conf_col       = next(
        (c for c in df.columns if "confidence" in c.lower()),
        None,
    )
    volume_col     = _safe_volume_col(df)

    if mean_col not in df.columns:
        raise ValueError(f"Expected '{mean_col}' in {prefix} sentiment DataFrame.")

    # ── Core features ────────────────────────────────────────────────────────────
    out = pd.DataFrame()
    out["time_window"] = df["time_window"]

    out[f"{prefix}_mean_sentiment"] = df[mean_col]
    out[f"{prefix}_volatility"]     = df[std_col] if std_col in df.columns else 0.0
    out[f"{prefix}_volume"]         = df[volume_col]

    # Momentum = first difference of mean sentiment (backward-looking by definition)
    out[f"{prefix}_momentum"] = df[mean_col].diff().fillna(0)

    # EMA of mean sentiment
    out[f"{prefix}_ema"] = (
        df[mean_col]
        .ewm(span=EMA_SPAN, adjust=False)
        .mean()
        .fillna(0)
    )

    # Confidence-weighted sentiment
    if conf_col:
        out[f"{prefix}_weighted"] = (df[mean_col] * df[conf_col]).fillna(0)
    else:
        logger.warning(f"[{prefix}] No confidence column found; setting weighted = mean_sentiment.")
        out[f"{prefix}_weighted"] = df[mean_col].fillna(0)

    # ── Lag features (1–5) — strictly backward-looking ──────────────────────────
    for lag in LAG_RANGE:
        out[f"{prefix}_lag_{lag}"] = df[mean_col].shift(lag).fillna(0)

    return out.reset_index(drop=True)


# ================================================================================
# STEP 5 — COMBINE POSTS + COMMENTS
# ================================================================================

def combine_sentiment(
    posts_feat: pd.DataFrame,
    comments_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outer-join post and comment features on time_window, then compute
    combined alpha features.  Missing values are filled with 0 before
    any combination arithmetic.
    """
    logger.info("🔗 Combining post + comment sentiment features…")

    merged = pd.merge(posts_feat, comments_feat, on="time_window", how="outer")
    merged = merged.sort_values("time_window").reset_index(drop=True)

    # Fill gaps introduced by the outer join
    merged = merged.fillna(0)

    # ── Combined signals ────────────────────────────────────────────────────────
    merged["sentiment_combined"] = (
        0.4 * merged["post_mean_sentiment"]
        + 0.6 * merged["comment_mean_sentiment"]
    )

    merged["sentiment_volume_total"] = (
        merged["post_volume"] + merged["comment_volume"]
    )

    merged["sentiment_disagreement"] = (
        merged["post_mean_sentiment"] - merged["comment_mean_sentiment"]
    ).abs()

    # Volume-weighted confidence average; guard against zero total volume
    total_vol = merged["sentiment_volume_total"].replace(0, np.nan)
    merged["sentiment_confidence_combined"] = (
        merged["post_weighted"] * merged["post_volume"]
        + merged["comment_weighted"] * merged["comment_volume"]
    ) / total_vol
    merged["sentiment_confidence_combined"] = (
        merged["sentiment_confidence_combined"].fillna(0)
    )

    logger.info(f"  Combined sentiment rows: {len(merged):,}")

    return merged


# ================================================================================
# STEP 6 — OHLCV FEATURE ENGINEERING
# ================================================================================

def build_ohlcv_features(ohlcv_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 1-minute OHLCV to hourly then computes market features.
    All NaNs replaced with 0.
    """
    logger.info(f"📊 Resampling OHLCV to {OHLCV_RESAMPLE_FREQ}…")

    ohlcv_h = resample_ohlcv(ohlcv_1m, OHLCV_RESAMPLE_FREQ).copy()
    ohlcv_h[OHLCV_TIME_COLUMN] = pd.to_datetime(ohlcv_h[OHLCV_TIME_COLUMN], utc=True)
    ohlcv_h = ohlcv_h.sort_values(OHLCV_TIME_COLUMN).reset_index(drop=True)

    logger.info(f"  Hourly candles: {len(ohlcv_h):,}")

    # Strictly backward-looking market features
    ohlcv_h["returns"]          = ohlcv_h["close"].pct_change().fillna(0)
    ohlcv_h["volatility"]       = (
        ohlcv_h["returns"]
        .rolling(VOL_WINDOW, min_periods=1)
        .std()
        .fillna(0)
    )
    ohlcv_h["volume_change"]    = ohlcv_h["volume"].pct_change().fillna(0)
    ohlcv_h["price_momentum"]   = ohlcv_h["close"].diff().fillna(0)

    return ohlcv_h


# ================================================================================
# STEP 7 — MERGE SENTIMENT + MARKET
# ================================================================================

def merge_sentiment_market(
    ohlcv_h: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    LEFT JOIN ohlcv_h (market anchor) with sentiment on datetime == time_window.
    Forward-fills then zero-fills any remaining NaN sentiment values.
    """
    logger.info("🔀 Merging OHLCV + sentiment…")

    merged = ohlcv_h.merge(
        sentiment_df,
        left_on=OHLCV_TIME_COLUMN,
        right_on="time_window",
        how="left",
    )

    # Drop the redundant time_window column that came from the right side
    if "time_window" in merged.columns:
        merged = merged.drop(columns=["time_window"])

    # Forward fill to propagate the last known sentiment into market gaps
    sentiment_cols = [c for c in merged.columns if c not in ohlcv_h.columns]
    merged[sentiment_cols] = merged[sentiment_cols].ffill()

    # Zero-fill any remaining NaNs (start of series with no prior sentiment)
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0)

    logger.info(f"  Merged rows: {len(merged):,}")

    return merged.reset_index(drop=True)


# ================================================================================
# STEP 8 — ALPHA FEATURES
# ================================================================================

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-domain alpha signals.  All NaNs replaced with 0.
    """
    logger.info("⚡ Building alpha features…")

    df = df.copy()

    # 1. Divergence: sentiment direction vs price momentum
    df["divergence"] = (df["sentiment_combined"] - df["price_momentum"]).fillna(0)

    # 2. Sentiment spike: volume > 2× rolling mean (boolean → int)
    rolling_mean_vol = (
        df["sentiment_volume_total"]
        .rolling(VOL_WINDOW, min_periods=1)
        .mean()
    )
    df["sentiment_spike"] = (
        df["sentiment_volume_total"] > SPIKE_MULTIPLIER * rolling_mean_vol
    ).astype(int)

    # 3. Fear / greed index
    df["fear_greed_index"] = (
        df["sentiment_combined"] * df["sentiment_volume_total"]
    ).fillna(0)

    # 4. Sentiment × price interaction (both differenced → backward-looking)
    df["sentiment_price_interaction"] = (
        df["sentiment_combined"].diff() * df["returns"]
    ).fillna(0)

    return df


# ================================================================================
# STEP 9 — TARGET VARIABLES
# ================================================================================

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds classification (target) and regression (target_return) labels.

    target        = 1 if next-bar close > current close, else 0
    target_return = next-bar pct_change of close

    The final row is dropped because its target is undefined (no future bar).
    """
    logger.info("🎯 Building target variables…")

    df = df.copy()

    df["target"]        = (df["close"].shift(-1) > df["close"]).astype(int)
    df["target_return"] = df["close"].pct_change().shift(-1)

    # Drop only the last row (NaN target from forward-shift)
    df = df.iloc[:-1].reset_index(drop=True)

    logger.info(f"  Rows after target construction: {len(df):,}")

    return df


# ================================================================================
# STEP 10 — FINAL FEATURE SELECTION
# ================================================================================

SENTIMENT_FEATURES = [
    "sentiment_combined",
    "sentiment_volume_total",
    "sentiment_disagreement",
    "post_momentum",
    "comment_momentum",
    *[f"post_lag_{i}"    for i in LAG_RANGE],
    *[f"comment_lag_{i}" for i in LAG_RANGE],
]

MARKET_FEATURES = [
    "returns",
    "volatility",
    "volume_change",
    "price_momentum",
]

ALPHA_FEATURES = [
    "divergence",
    "sentiment_spike",
    "fear_greed_index",
    "sentiment_price_interaction",
]

TARGET_COLS = ["target", "target_return"]

ALL_FEATURES = SENTIMENT_FEATURES + MARKET_FEATURES + ALPHA_FEATURES


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the final feature set + datetime + targets.
    Enforces zero-fill, deduplication, and ascending sort.
    """
    logger.info("🗂 Selecting final feature set…")

    keep_cols = [OHLCV_TIME_COLUMN] + ALL_FEATURES + TARGET_COLS
    missing   = [c for c in keep_cols if c not in df.columns]

    if missing:
        logger.warning(f"  Missing columns (will be created as 0): {missing}")
        for col in missing:
            df[col] = 0

    final = df[keep_cols].copy()

    # ── STEP 11: Final cleaning ─────────────────────────────────────────────────
    # Fill remaining NaNs (targets intentionally excluded from blanket fill)
    feature_cols = [c for c in ALL_FEATURES if c in final.columns]
    final[feature_cols] = final[feature_cols].fillna(0)

    final = final.sort_values(OHLCV_TIME_COLUMN).reset_index(drop=True)
    final = final.drop_duplicates(subset=[OHLCV_TIME_COLUMN]).reset_index(drop=True)

    logger.info(f"  Final shape: {final.shape}")

    return final


# ================================================================================
# STEP 12 — SAVE OUTPUTS
# ================================================================================

def save_outputs(final_df: pd.DataFrame) -> None:
    """
    Persists final_features_df to:
      1. Database  (ml_features.ml_features)
      2. CSV       (features.csv)
      3. Parquet   (features.parquet)
    """
    logger.info("💾 Saving outputs…")

    # DB
    save_df_to_db(
        final_df,
        table_name=ML_FEATURES_TABLE,
        schema=ML_FEATURES_SCHEMA,
        time_column=OHLCV_TIME_COLUMN,
        is_timeseries=True,
    )
    logger.info(f"  ✅ Saved to DB: {ML_FEATURES_SCHEMA}.{ML_FEATURES_TABLE}")

    # CSV
    final_df.to_csv(CSV_PATH, index=False)
    logger.info(f"  ✅ CSV: {CSV_PATH}")

    # Parquet
    final_df.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"  ✅ Parquet: {PARQUET_PATH}")


# ================================================================================
# STEP 13 — DIAGNOSTICS
# ================================================================================

def run_diagnostics(final_df: pd.DataFrame) -> None:
    """
    Prints dataset shape, missing value audit, feature-target correlations,
    and classification target class balance.
    """
    logger.info("=" * 72)
    logger.info("📋 DIAGNOSTICS")
    logger.info("=" * 72)

    # Shape
    logger.info(f"  Shape: {final_df.shape}")

    # Missing values
    total_nans = final_df[ALL_FEATURES].isna().sum().sum()
    if total_nans == 0:
        logger.info("  ✅ Missing values: 0 (feature columns)")
    else:
        nan_report = final_df[ALL_FEATURES].isna().sum()
        nan_report = nan_report[nan_report > 0]
        logger.warning(f"  ⚠️ NaN columns:\n{nan_report}")

    # Feature correlation with target
    if "target" in final_df.columns:
        corr = (
            final_df[ALL_FEATURES + ["target"]]
            .corr()["target"]
            .drop("target")
            .sort_values(key=abs, ascending=False)
        )
        logger.info("  📈 Top 10 feature correlations with 'target':")
        for feat, val in corr.head(10).items():
            logger.info(f"    {feat:<45} {val:+.4f}")

    # Class balance
    if "target" in final_df.columns:
        balance = final_df["target"].value_counts(normalize=True).sort_index()
        logger.info("  ⚖️  Class balance (target):")
        for cls, pct in balance.items():
            label = "UP  (1)" if cls == 1 else "DOWN(0)"
            logger.info(f"    {label}: {pct:.1%}")

    logger.info("=" * 72)


# ================================================================================
# MAIN
# ================================================================================

def main(save_to_database: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Returns:
        final_features_df — ML-ready DataFrame
    """
    logger.info("🚀 FEATURE PIPELINE START")

    # ── 1. Load sentiment ───────────────────────────────────────────────────────
    posts_df, comments_df = load_sentiment_data()

    # ── 2. Compute global date range ────────────────────────────────────────────
    start_date, end_date = compute_date_range(posts_df, comments_df)

    # ── 3. Fetch OHLCV ──────────────────────────────────────────────────────────
    ohlcv_1m = load_ohlcv(start_date, end_date)

    # ── 4. Sentiment feature engineering ────────────────────────────────────────
    logger.info("🧮 Engineering sentiment features…")
    posts_feat    = create_sentiment_features(posts_df,    prefix="post")
    comments_feat = create_sentiment_features(comments_df, prefix="comment")

    # ── 5. Combine posts + comments ─────────────────────────────────────────────
    sentiment_df = combine_sentiment(posts_feat, comments_feat)

    # ── 6. OHLCV feature engineering ────────────────────────────────────────────
    ohlcv_h = build_ohlcv_features(ohlcv_1m)

    # ── 7. Merge sentiment + market ─────────────────────────────────────────────
    merged_df = merge_sentiment_market(ohlcv_h, sentiment_df)

    # ── 8. Alpha features ───────────────────────────────────────────────────────
    merged_df = build_alpha_features(merged_df)

    # ── 9. Target variables ─────────────────────────────────────────────────────
    merged_df = build_targets(merged_df)

    # ── 10 + 11. Select + clean final feature set ───────────────────────────────
    final_features_df = select_final_features(merged_df)

    # ── 12. Save outputs ────────────────────────────────────────────────────────
    if save_to_database:
        save_outputs(final_features_df)

    # ── 13. Diagnostics ─────────────────────────────────────────────────────────
    run_diagnostics(final_features_df)

    logger.info("✅ FEATURE PIPELINE COMPLETE")

    return final_features_df


# ================================================================================
# RUN
# ================================================================================

if __name__ == "__main__":
    final_features_df = main()
    logger.info(final_features_df.head())