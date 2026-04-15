"""
feature_engineering.py
=======================
Production-ready ML feature engineering pipeline for TradeX.

Combines:
  - Reddit post sentiment  (hourly)
  - Reddit comment sentiment (hourly)
  - OHLCV market data      (minute → resampled to 1h)

Output: ml_features table  +  CSV  +  Parquet
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from TradeX.utils.db.utils import read_df_from_db, save_df_to_db
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.common.logs import get_logger

logger = get_logger("feature_engineering")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_REDDIT  = "reddit"
SCHEMA_MARKET  = "data_binance"
SCHEMA_OUTPUT  = "reddit"

POSTS_AGG_TABLE    = "posts_sentiment_hourly"
COMMENTS_AGG_TABLE = "comments_sentiment_hourly"
OHLCV_TABLE        = "btc_1m"
OUTPUT_TABLE       = "ml_features"

RESAMPLE_FREQ = "1h"

# Weights for combined sentiment (comments weighted higher — more signal-dense)
POST_WEIGHT    = 0.4
COMMENT_WEIGHT = 0.6

# Lag window
N_LAGS = 5

# EMA span for sentiment smoothing
EMA_SPAN = 5

# Volume spike window
SPIKE_WINDOW = 10

# Correlation columns to report in diagnostics
DIAG_FEATURES = [
    "sentiment_combined", "sentiment_volume_total", "sentiment_disagreement",
    "post_momentum", "comment_momentum",
    "returns", "volatility", "volume_change", "price_momentum",
    "divergence", "fear_greed_index", "sentiment_price_interaction",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load OHLCV, posts sentiment, and comments sentiment from DB.

    Returns:
        ohlcv_df   : 1-minute OHLCV (datetime sorted)
        posts_df   : hourly posts sentiment (time_window sorted)
        comments_df: hourly comments sentiment (time_window sorted)
    """
    logger.info("Loading data from DB…")

    ohlcv_df    = read_df_from_db(OHLCV_TABLE,        SCHEMA_MARKET)
    posts_df    = read_df_from_db(POSTS_AGG_TABLE,    SCHEMA_REDDIT)
    comments_df = read_df_from_db(COMMENTS_AGG_TABLE, SCHEMA_REDDIT)

    # ── Parse & sort datetimes ──────────────────────────────────────────────
    ohlcv_df["datetime"]      = pd.to_datetime(ohlcv_df["datetime"], utc=True)
    posts_df["time_window"]   = pd.to_datetime(posts_df["time_window"], utc=True)
    comments_df["time_window"] = pd.to_datetime(comments_df["time_window"], utc=True)

    ohlcv_df    = ohlcv_df.sort_values("datetime").reset_index(drop=True)
    posts_df    = posts_df.sort_values("time_window").reset_index(drop=True)
    comments_df = comments_df.sort_values("time_window").reset_index(drop=True)

    logger.info(
        f"Loaded  OHLCV={len(ohlcv_df):,}  "
        f"posts={len(posts_df):,}  "
        f"comments={len(comments_df):,}"
    )
    return ohlcv_df, posts_df, comments_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  — SENTIMENT FEATURE ENGINEERING (PER SOURCE)
# ─────────────────────────────────────────────────────────────────────────────

def create_sentiment_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Build per-source sentiment features from an hourly aggregated DataFrame.

    Expected input columns (from sentiment_analysis.py aggregation):
        time_window, mean_sentiment, std_sentiment, sentiment_confidence_mean,
        <id_col>_count  (num items)

    Returns a new DataFrame indexed by time_window with prefixed feature cols.

    NOTE:
      - All NaNs produced by diff / EMA / lag are filled with 0 (safe default).
      - No rows are dropped.
    """
    df = df.copy()

    # ── Resolve the "count" column (created dynamically in aggregate_sentiment_hourly)
    count_col = next(
        (c for c in df.columns if c.endswith("_count")),
        None
    )
    if count_col is None:
        logger.warning(f"[{prefix}] No count column found; defaulting num_items to 0.")
        df["_num_items"] = 0
    else:
        df["_num_items"] = df[count_col].fillna(0)

    # ── Guard: ensure required columns exist ────────────────────────────────
    for col in ("mean_sentiment", "std_sentiment", "sentiment_confidence_mean"):
        if col not in df.columns:
            logger.warning(f"[{prefix}] Missing '{col}'; defaulting to 0.")
            df[col] = 0.0

    s = df["mean_sentiment"].fillna(0)
    std = df["std_sentiment"].fillna(0)
    conf = df["sentiment_confidence_mean"].fillna(0)
    volume = df["_num_items"]

    features = pd.DataFrame(index=df.index)
    features["time_window"] = df["time_window"]

    # Core features
    features[f"{prefix}_mean_sentiment"] = s
    features[f"{prefix}_momentum"]       = s.diff().fillna(0)
    features[f"{prefix}_volatility"]     = std
    features[f"{prefix}_volume"]         = volume
    features[f"{prefix}_ema"]            = s.ewm(span=EMA_SPAN, adjust=False).mean().fillna(0)
    features[f"{prefix}_weighted"]       = (s * conf).fillna(0)

    # Lag features — shift(n) on first rows → NaN → fill with 0
    for lag in range(1, N_LAGS + 1):
        features[f"{prefix}_lag_{lag}"] = s.shift(lag).fillna(0)

    logger.info(f"[{prefix}] sentiment features built: {features.shape[1] - 1} cols")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  — COMBINE POSTS + COMMENTS
# ─────────────────────────────────────────────────────────────────────────────

def combine_sentiment(
    posts_feats: pd.DataFrame,
    comments_feats: pd.DataFrame,
    posts_raw: pd.DataFrame,
    comments_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outer-join posts and comments features on time_window, then build
    cross-source alpha signals.

    posts_raw / comments_raw supply mean_sentiment and count for the
    confidence-weighted average that requires original volume info.
    """
    # ── Merge feature sets ──────────────────────────────────────────────────
    merged = pd.merge(
        posts_feats,
        comments_feats,
        on="time_window",
        how="outer",
    ).sort_values("time_window").reset_index(drop=True)

    # Fill any gaps created by outer join with 0 (never drop rows)
    post_sent    = merged["post_mean_sentiment"].fillna(0)
    comment_sent = merged["comment_mean_sentiment"].fillna(0)
    post_vol     = merged["post_volume"].fillna(0)
    comment_vol  = merged["comment_volume"].fillna(0)

    # Weighted combined sentiment (volume-weighted for confidence avg)
    total_vol = post_vol + comment_vol

    merged["sentiment_combined"] = (
        POST_WEIGHT    * post_sent +
        COMMENT_WEIGHT * comment_sent
    )

    merged["sentiment_volume_total"] = total_vol

    merged["sentiment_disagreement"] = (post_sent - comment_sent).abs()

    # Volume-weighted confidence average; guard against zero denominator
    post_conf    = merged.get("post_weighted",    post_sent)     # proxy if missing
    comment_conf = merged.get("comment_weighted", comment_sent)

    with np.errstate(invalid="ignore", divide="ignore"):
        merged["sentiment_confidence_combined"] = np.where(
            total_vol > 0,
            (post_conf * post_vol + comment_conf * comment_vol) / total_vol,
            0.0,
        )

    # Final NaN sweep on combined features
    combined_cols = [
        "sentiment_combined",
        "sentiment_volume_total",
        "sentiment_disagreement",
        "sentiment_confidence_combined",
    ]
    merged[combined_cols] = merged[combined_cols].fillna(0)

    logger.info(f"Combined sentiment shape: {merged.shape}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  — OHLCV FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_ohlcv_features(ohlcv_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1m OHLCV to hourly and compute market features.

    All NaNs from rolling / pct_change are filled with 0.
    """
    logger.info(f"Resampling OHLCV 1m → {RESAMPLE_FREQ}…")
    df = resample_ohlcv(ohlcv_1m, RESAMPLE_FREQ).copy()

    df["returns"]        = df["close"].pct_change().fillna(0)
    df["volatility"]     = df["returns"].rolling(SPIKE_WINDOW).std().fillna(0)
    df["volume_change"]  = df["volume"].pct_change().fillna(0)
    df["price_momentum"] = df["close"].diff().fillna(0)

    logger.info(f"OHLCV features built: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  — MERGE SENTIMENT + MARKET
# ─────────────────────────────────────────────────────────────────────────────

def merge_sentiment_market(
    ohlcv_feats: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join OHLCV (master timeline) with combined sentiment.

    Missing sentiment values after merge:
      1. Forward-filled (carry last known value — no leakage, hourly cadence)
      2. Remaining leading NaNs filled with 0
    """
    merged = ohlcv_feats.merge(
        sentiment_df,
        left_on="datetime",
        right_on="time_window",
        how="left",
    )

    # Drop redundant time_window column
    merged.drop(columns=["time_window"], errors="ignore", inplace=True)

    # Forward fill then zero-fill
    sentiment_cols = [c for c in merged.columns if c not in ohlcv_feats.columns]
    merged[sentiment_cols] = (
        merged[sentiment_cols]
        .ffill()
        .fillna(0)
    )

    logger.info(f"Merged shape: {merged.shape}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  — ALPHA FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-domain alpha signals from merged market + sentiment data.

    All NaN results are filled with 0.
    """
    df = df.copy()

    # 1. Divergence: sentiment vs price momentum (normalised to similar scale)
    df["divergence"] = (
        df["sentiment_combined"] - df["price_momentum"]
    ).fillna(0)

    # 2. Sentiment spike: binary flag when volume > 2x rolling mean
    rolling_vol_mean = (
        df["sentiment_volume_total"]
        .rolling(SPIKE_WINDOW, min_periods=1)
        .mean()
        .fillna(0)
    )
    df["sentiment_spike"] = np.where(
    df["sentiment_combined"].diff() > 0.3, 1,
    np.where(df["sentiment_combined"].diff() < -0.3, -1, 0)
    )

    # 3. Fear-greed proxy: sentiment × activity
    df["fear_greed_index"] = (
        df["sentiment_combined"] * df["sentiment_volume_total"]
    ).fillna(0)

    # 4. Interaction: rate-of-change of sentiment × price returns
    df["sentiment_price_interaction"] = (
        df["sentiment_combined"].diff().fillna(0) * df["returns"]
    ).fillna(0)

    logger.info("Alpha features built.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  — TARGET VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct classification and regression targets.

    target        : 1 if next-bar close > current close, else 0
    target_return : next-bar pct_change of close

    The LAST row is dropped (no forward label available).
    No other rows are dropped.
    """
    df = df.copy()

    df["target"]        = (df["close"].shift(-1) > df["close"]).astype("Int64")
    df["target_return"] = df["close"].pct_change().shift(-1)

    # Drop only the final row (NaN target unavoidable)
    df = df.iloc[:-1].copy()

    logger.info(f"Targets added. Shape after last-row drop: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  — SELECT FINAL FEATURE SET
# ─────────────────────────────────────────────────────────────────────────────

def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the documented feature set + meta / target columns.
    """
    lag_cols = (
        [f"post_lag_{i}"    for i in range(1, N_LAGS + 1)] +
        [f"comment_lag_{i}" for i in range(1, N_LAGS + 1)]
    )

    feature_cols = [
        # ── Identifiers
        "datetime",

        # ── OHLCV passthrough (needed for targets / downstream models)
        "open", "high", "low", "close", "volume",

        # ── Sentiment
        "sentiment_combined",
        "sentiment_volume_total",
        "sentiment_disagreement",
        "post_momentum",
        "comment_momentum",
        *lag_cols,

        # ── Market
        "returns",
        "volatility",
        "volume_change",
        "price_momentum",

        # ── Alpha
        "divergence",
        "sentiment_spike",
        "fear_greed_index",
        "sentiment_price_interaction",

        # ── Targets
        "target",
        "target_return",
    ]

    # Keep only columns that actually exist (defensive)
    available = [c for c in feature_cols if c in df.columns]
    missing   = set(feature_cols) - set(available)
    if missing:
        logger.warning(f"Missing columns (skipped): {missing}")

    final = df[available].copy()
    logger.info(f"Final feature set: {final.shape}")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9  — FINAL CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Last safety pass:
      - Fill any residual NaNs (except targets) with 0
      - Sort by datetime
      - Remove duplicate timestamps
    """
    df = df.copy()

    # Preserve targets as-is (they should be clean after build_targets)
    non_target_cols = [c for c in df.columns if c not in ("target", "target_return")]
    df[non_target_cols] = df[non_target_cols].fillna(0)

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )

    logger.info(f"Final clean shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10  — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame) -> None:
    """
    Persist the final feature DataFrame to:
      1. Database (ml_features.ml_features)
      2. CSV
      3. Parquet
    """
    logger.info("Saving to DB…")
    save_df_to_db(df, OUTPUT_TABLE, SCHEMA_OUTPUT, "datetime", is_timeseries=True)


    logger.info(f"Saved → DB: {SCHEMA_OUTPUT}.{OUTPUT_TABLE}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11  — DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(df: pd.DataFrame) -> None:
    """
    Print key quality metrics without modifying the DataFrame.
    """
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING DIAGNOSTICS")
    print("=" * 60)

    print(f"\n  Dataset shape   : {df.shape}")
    print(f"  Datetime range  : {df['datetime'].min()} → {df['datetime'].max()}")

    # Missing values
    total_nan = df.drop(columns=["target", "target_return"], errors="ignore").isna().sum().sum()
    print(f"\n  Missing values (excl. targets): {total_nan}  (should be 0)")

    # Class balance
    if "target" in df.columns:
        counts = df["target"].value_counts()
        pct_up = counts.get(1, 0) / len(df) * 100
        print(f"\n  Class balance   : UP={pct_up:.1f}%  DOWN={100 - pct_up:.1f}%")

    # Feature correlations with target
    if "target" in df.columns:
        print("\n  Feature correlations with target (top 10 |r|):")
        available_diag = [c for c in DIAG_FEATURES if c in df.columns]
        corr_series = (
            df[available_diag + ["target"]]
            .corr()["target"]
            .drop("target")
            .abs()
            .sort_values(ascending=False)
        )
        print(corr_series.head(10).to_string())

    print("\n" + "=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_pipeline(save_to_db: bool = True) -> pd.DataFrame:
    """
    Orchestrate the full feature engineering pipeline.

    Args:
        save_to_db: If True, persist outputs to DB + disk.

    Returns:
        final_features_df: ML-ready DataFrame.
    """
    logger.info("══ Feature Engineering Pipeline START ══")

    # ── 1. Load ─────────────────────────────────────────────────────────────
    ohlcv_df, posts_df, comments_df = load_data()

    # ── 2. Per-source sentiment features ────────────────────────────────────
    posts_feats    = create_sentiment_features(posts_df,    prefix="post")
    comments_feats = create_sentiment_features(comments_df, prefix="comment")

    # ── 3. Combine sentiment sources ────────────────────────────────────────
    sentiment_df = combine_sentiment(
        posts_feats, comments_feats, posts_df, comments_df
    )

    # ── 4. OHLCV features ───────────────────────────────────────────────────
    ohlcv_feats = build_ohlcv_features(ohlcv_df)

    # ── 5. Merge market + sentiment ─────────────────────────────────────────
    merged_df = merge_sentiment_market(ohlcv_feats, sentiment_df)

    # ── 6. Alpha features ───────────────────────────────────────────────────
    merged_df = build_alpha_features(merged_df)

    # ── 7. Target variables ─────────────────────────────────────────────────
    merged_df = build_targets(merged_df)

    # ── 8. Select final feature set ─────────────────────────────────────────
    final_df = select_final_features(merged_df)

    # ── 9. Final cleaning ───────────────────────────────────────────────────
    final_df = final_clean(final_df)

    # ── 10. Persist ─────────────────────────────────────────────────────────
    if save_to_db:
        save_outputs(final_df)

    # ── 11. Diagnostics ─────────────────────────────────────────────────────
    run_diagnostics(final_df)

    logger.info("══ Feature Engineering Pipeline DONE ══")
    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    final_features_df = run_feature_pipeline(save_to_db=True)