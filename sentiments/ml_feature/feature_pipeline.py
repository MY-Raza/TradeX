from __future__ import annotations

import pandas as pd
import numpy as np

# -- Project imports (match existing codebase conventions) --
from TradeX.utils.db.utils import read_df_from_db, save_df_to_db
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.common.logs import get_logger

logger = get_logger("feature_pipeline")


# ===========================================================================
# STEP 1 — LOAD DATA
# ===========================================================================

def load_data(
    ohlcv_table: str,
    posts_table: str = "posts_sentiment_hourly",
    comments_table: str = "comments_sentiment_hourly",
    ohlcv_schema: str | None = "data_binance",
    sentiment_schema: str | None = "reddit",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load OHLCV and both Reddit sentiment tables from the database.

    Args:
        ohlcv_table      : Table name for raw OHLCV data.
        posts_table      : Table name for hourly post sentiment.
        comments_table   : Table name for hourly comment sentiment.
        ohlcv_schema     : DB schema for OHLCV (default: 'data_binance').
        sentiment_schema : DB schema for both sentiment tables (default: 'reddit').

    Returns:
        ohlcv_df     : minute-level OHLCV, sorted by datetime
        posts_df     : hourly post sentiment, sorted by time_window
        comments_df  : hourly comment sentiment, sorted by time_window
    """
    logger.info(
        f"Loading data from DB… "
        f"[OHLCV: {ohlcv_schema}.{ohlcv_table}] "
        f"[Posts: {sentiment_schema}.{posts_table}] "
        f"[Comments: {sentiment_schema}.{comments_table}]"
    )

    ohlcv_df    = read_df_from_db(ohlcv_table,    schema=ohlcv_schema)
    posts_df    = read_df_from_db(posts_table,    schema=sentiment_schema)
    comments_df = read_df_from_db(comments_table, schema=sentiment_schema)

    # ── Parse & sort OHLCV ──────────────────────────────────────────────────
    ohlcv_df["datetime"] = pd.to_datetime(ohlcv_df["datetime"], utc=True)
    ohlcv_df = ohlcv_df.sort_values("datetime").reset_index(drop=True)

    # ── Parse & sort sentiment frames ───────────────────────────────────────
    for df, name in [(posts_df, "posts"), (comments_df, "comments")]:
        if "time_window" not in df.columns:
            raise KeyError(f"{name} table missing 'time_window' column")
        df["time_window"] = pd.to_datetime(df["time_window"], utc=True)

    posts_df    = posts_df.sort_values("time_window").reset_index(drop=True)
    comments_df = comments_df.sort_values("time_window").reset_index(drop=True)

    logger.info(
        f"Loaded | OHLCV: {len(ohlcv_df):,} rows | "
        f"Posts: {len(posts_df):,} rows | Comments: {len(comments_df):,} rows"
    )
    return ohlcv_df, posts_df, comments_df


# ===========================================================================
# STEP 2 — SENTIMENT FEATURE ENGINEERING (REUSABLE)
# ===========================================================================

def create_sentiment_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Engineer sentiment features from a single source (posts or comments).

    Expected columns in `df`:
        time_window, mean_sentiment, std_sentiment,
        num_items, sentiment_confidence_mean

    Args:
        df     : Hourly sentiment DataFrame.
        prefix : Column name prefix, e.g. 'post' or 'comment'.

    Returns:
        DataFrame indexed by time_window with engineered feature columns.
    """
    df = df.copy()
    df = df.sort_values("time_window").reset_index(drop=True)

    # FIX: Normalise the item-count column name.
    # sentiment_analysis.py now always writes 'num_items', but tables written
    # by older runs may carry a legacy name like 'post_id_count' or
    # 'comment_id_count'.  This block handles both cases gracefully so the
    # pipeline never raises a KeyError regardless of which DB snapshot is used.
    if "num_items" not in df.columns:
        legacy_count_cols = [c for c in df.columns if c.endswith("_count")]
        if legacy_count_cols:
            df.rename(columns={legacy_count_cols[0]: "num_items"}, inplace=True)
            logger.warning(
                f"[{prefix}] Renamed '{legacy_count_cols[0]}' → 'num_items'. "
                f"Re-run sentiment_analysis.py to persist the correct column name."
            )
        else:
            logger.warning(
                f"[{prefix}] No count column found — defaulting num_items to 1."
            )
            df["num_items"] = 1

    # Core series — fill source NaNs before any computation
    mean_s  = df["mean_sentiment"].fillna(0)
    std_s   = df["std_sentiment"].fillna(0)
    num_i   = df["num_items"].fillna(0)
    conf    = df["sentiment_confidence_mean"].fillna(0)

    out = pd.DataFrame({"time_window": df["time_window"]})

    # ── Base features ────────────────────────────────────────────────────────
    out[f"{prefix}_mean_sentiment"] = mean_s
    out[f"{prefix}_volatility"]     = std_s
    out[f"{prefix}_volume"]         = num_i

    # ── Momentum (1-step diff) ───────────────────────────────────────────────
    # NaN only appears on the first row; fill with 0 (no prior info)
    out[f"{prefix}_momentum"] = mean_s.diff().fillna(0)

    # ── EMA (span=5) ────────────────────────────────────────────────────────
    # adjust=False → pure recursive EMA; first rows warm up from available data
    out[f"{prefix}_ema"] = mean_s.ewm(span=5, adjust=False).mean().fillna(0)

    # ── Confidence-weighted sentiment ────────────────────────────────────────
    out[f"{prefix}_weighted"] = (mean_s * conf).fillna(0)

    # ── Lag features (1–5) ───────────────────────────────────────────────────
    for lag in range(1, 6):
        out[f"{prefix}_lag_{lag}"] = mean_s.shift(lag).fillna(0)

    logger.info(
        f"[{prefix}] Sentiment features built | "
        f"shape: {out.shape} | NaNs: {out.isna().sum().sum()}"
    )
    return out


# ===========================================================================
# STEP 3 — COMBINE POST + COMMENT SENTIMENT
# ===========================================================================

def combine_sentiment(
    posts_feat_df: pd.DataFrame,
    comments_feat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outer-merge post and comment features on time_window, then create
    combined signals.

    Args:
        posts_feat_df    : Output of create_sentiment_features(posts_df, 'post')
        comments_feat_df : Output of create_sentiment_features(comments_df, 'comment')

    Returns:
        sentiment_df : Combined hourly sentiment DataFrame.
    """
    # Outer join — preserves rows that exist in either source
    sentiment_df = pd.merge(
        posts_feat_df, comments_feat_df,
        on="time_window", how="outer"
    ).sort_values("time_window").reset_index(drop=True)

    # Fill any gaps introduced by the outer join BEFORE combining
    post_mean    = sentiment_df["post_mean_sentiment"].fillna(0)
    comment_mean = sentiment_df["comment_mean_sentiment"].fillna(0)
    post_vol     = sentiment_df["post_volume"].fillna(0)
    comment_vol  = sentiment_df["comment_volume"].fillna(0)
    post_w       = sentiment_df["post_weighted"].fillna(0)
    comment_w    = sentiment_df["comment_weighted"].fillna(0)

    # ── Combined sentiment (volume-weighted blend) ───────────────────────────
    # Comments weighted higher (0.6) — typically more conversational signal
    sentiment_df["sentiment_combined"] = (
        0.4 * post_mean + 0.6 * comment_mean
    )

    # ── Total sentiment volume ────────────────────────────────────────────────
    sentiment_df["sentiment_volume_total"] = post_vol + comment_vol

    # ── Disagreement between sources ─────────────────────────────────────────
    sentiment_df["sentiment_disagreement"] = (
        (post_mean - comment_mean).abs()
    )

    # ── Volume-weighted confidence ────────────────────────────────────────────
    total_vol = post_vol + comment_vol
    # Guard against zero-volume rows
    safe_vol = total_vol.replace(0, np.nan)
    sentiment_df["sentiment_confidence_combined"] = (
        (post_w * post_vol + comment_w * comment_vol) / safe_vol
    ).fillna(0)

    # Final NaN sweep on all combined columns
    combined_cols = [
        "sentiment_combined", "sentiment_volume_total",
        "sentiment_disagreement", "sentiment_confidence_combined",
    ]
    sentiment_df[combined_cols] = sentiment_df[combined_cols].fillna(0)

    logger.info(
        f"Sentiment combined | shape: {sentiment_df.shape} | "
        f"NaNs: {sentiment_df.isna().sum().sum()}"
    )
    return sentiment_df


# ===========================================================================
# STEP 4 — OHLCV FEATURE ENGINEERING
# ===========================================================================

def build_ohlcv_features(
    ohlcv_df: pd.DataFrame,
    target_interval: str = "1h",
) -> pd.DataFrame:
    """
    Resample minute OHLCV to `target_interval` and compute market features.

    Args:
        ohlcv_df        : Raw 1-minute OHLCV DataFrame (datetime column).
        target_interval : Resample target, e.g. '1h'.

    Returns:
        DataFrame with datetime + OHLCV columns + market feature columns.
    """
    # Resample using the existing data_cleaner utility
    df = resample_ohlcv(ohlcv_df, target_interval).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    close  = df["close"]
    volume = df["volume"]

    # ── Returns & momentum ───────────────────────────────────────────────────
    # pct_change / diff produce NaN on row 0 only → fill with 0
    df["returns"]        = close.pct_change().fillna(0)
    df["price_momentum"] = close.diff().fillna(0)
    df["volume_change"]  = volume.pct_change().fillna(0)

    # ── Rolling volatility (10-period std of returns) ────────────────────────
    # min_periods=1 avoids NaN for the first 9 rows
    df["volatility"] = (
        df["returns"].rolling(window=10, min_periods=1).std().fillna(0)
    )

    logger.info(
        f"OHLCV features built | interval: {target_interval} | "
        f"shape: {df.shape} | NaNs: {df.isna().sum().sum()}"
    )
    return df


# ===========================================================================
# STEP 5 — MERGE SENTIMENT + MARKET
# ===========================================================================

def merge_sentiment_market(
    ohlcv_feat_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join OHLCV features with hourly sentiment on the OHLCV timeline.
    Missing sentiment rows (before sentiment data starts, or sparse hours)
    are forward-filled then zero-filled.

    Args:
        ohlcv_feat_df : Output of build_ohlcv_features.
        sentiment_df  : Output of combine_sentiment.

    Returns:
        merged_df : Full merged DataFrame on OHLCV timeline.
    """
    merged_df = ohlcv_feat_df.merge(
        sentiment_df,
        left_on="datetime",
        right_on="time_window",
        how="left",
    )

    # Drop the redundant join key
    if "time_window" in merged_df.columns:
        merged_df = merged_df.drop(columns=["time_window"])

    # Sentiment columns can be NaN where no hourly bar aligns —
    # forward-fill first (carry last known value), then fill any head NaNs
    sentiment_cols = [c for c in merged_df.columns if c not in ohlcv_feat_df.columns]
    merged_df[sentiment_cols] = (
        merged_df[sentiment_cols].ffill().fillna(0)
    )

    logger.info(
        f"Merged | shape: {merged_df.shape} | NaNs: {merged_df.isna().sum().sum()}"
    )
    return merged_df


# ===========================================================================
# STEP 6 — ALPHA FEATURES
# ===========================================================================

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-source alpha signals from merged OHLCV + sentiment data.

    All signals are strictly backward-looking (no future data).

    Args:
        df : Output of merge_sentiment_market.

    Returns:
        df with alpha feature columns appended.
    """
    df = df.copy()

    sent   = df["sentiment_combined"]
    pmom   = df["price_momentum"]
    svol   = df["sentiment_volume_total"]
    ret    = df["returns"]

    # ── 1. Divergence: how much sentiment deviates from price momentum ───────
    df["divergence"] = (sent - pmom).fillna(0)

    # ── 2. Sentiment spike: volume > 2× rolling mean (binary signal) ─────────
    rolling_mean_vol = svol.rolling(window=10, min_periods=1).mean()
    df["sentiment_spike"] = (svol > 2 * rolling_mean_vol).astype(int)

    # ── 3. Fear/Greed index: sentiment amplitude × crowd size ────────────────
    df["fear_greed_index"] = (sent * svol).fillna(0)

    # ── 4. Sentiment-price interaction: change in sentiment × price return ───
    # sent.diff() uses only past values; fill first-row NaN
    df["sentiment_price_interaction"] = (
        sent.diff().fillna(0) * ret
    ).fillna(0)

    logger.info(
        f"Alpha features built | NaNs: {df.isna().sum().sum()}"
    )
    return df


# ===========================================================================
# STEP 7 — TARGET VARIABLES
# ===========================================================================

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct classification and regression targets.

    Targets look ONE step ahead.  Only the LAST row is dropped (it has no
    future close to predict).  NaNs are NOT filled — models must be aware
    of the boundary.

    Args:
        df : Full feature DataFrame (sorted by datetime).

    Returns:
        df with 'target' and 'target_return' columns; last row dropped.
    """
    df = df.copy()

    # Classification: 1 if next close > current close
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)

    # Regression: next period's return
    df["target_return"] = df["close"].pct_change().shift(-1)

    # Drop only the last row (no future close available)
    df = df.iloc[:-1].reset_index(drop=True)

    logger.info(
        f"Targets built | shape after drop: {df.shape} | "
        f"Class balance: {df['target'].value_counts().to_dict()}"
    )
    return df


# ===========================================================================
# STEP 8 — SELECT FINAL FEATURE SET
# ===========================================================================

FEATURE_COLUMNS = [
    # ── Sentiment ────────────────────────────────────────────────────────────
    "sentiment_combined",
    "sentiment_volume_total",
    "sentiment_disagreement",
    "post_momentum",
    "comment_momentum",
    *[f"post_lag_{i}"    for i in range(1, 6)],
    *[f"comment_lag_{i}" for i in range(1, 6)],

    # ── Market ───────────────────────────────────────────────────────────────
    "returns",
    "volatility",
    "volume_change",
    "price_momentum",

    # ── Alpha ────────────────────────────────────────────────────────────────
    "divergence",
    "sentiment_spike",
    "fear_greed_index",
    "sentiment_price_interaction",
]

TARGET_COLUMNS = ["target", "target_return"]


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the designated feature + target columns plus 'datetime'.
    Missing features (if any) are zero-filled and a warning is logged.
    """
    available = set(df.columns)
    missing   = [c for c in FEATURE_COLUMNS if c not in available]

    if missing:
        logger.warning(f"Missing features — zero-filling: {missing}")
        for col in missing:
            df[col] = 0.0

    keep = ["datetime"] + FEATURE_COLUMNS + TARGET_COLUMNS
    return df[keep].copy()


# ===========================================================================
# STEP 9 — FINAL CLEANING
# ===========================================================================

def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Last safety pass:
    - Sort by datetime
    - Remove exact duplicate timestamps
    - Fill any residual NaNs with 0 (features only, not targets)
    """
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])
    df = df.reset_index(drop=True)

    # Fill feature NaNs with 0 — leave target NaNs as-is (boundary rows)
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS + ["datetime"]]
    df[feature_cols] = df[feature_cols].fillna(0)

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        logger.info(f"Residual NaNs (expected in targets only):\n{nan_counts}")

    logger.info(f"Final clean | shape: {df.shape}")
    return df


# ===========================================================================
# STEP 10 — OUTPUT
# ===========================================================================

def save_outputs(
    df: pd.DataFrame,
    db_table: str = "ml_features",
    schema: str | None = None,
) -> None:
    """
    Persist the final feature DataFrame to DB, CSV, and Parquet.
    """
    # DB
    save_df_to_db(df, db_table, schema=schema)
    logger.info(f"Saved to DB: {db_table}")



# ===========================================================================
# STEP 11 — DIAGNOSTICS
# ===========================================================================

def print_diagnostics(df: pd.DataFrame) -> None:
    """
    Print dataset health and feature-target correlation report.
    """
    separator = "─" * 60

    logger.info(f"\n{separator}")
    logger.info("DATASET DIAGNOSTICS")
    logger.info(f"{separator}")

    # Shape
    logger.info(f"\nShape:  {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Missing values
    nan_total = df[FEATURE_COLUMNS].isna().sum().sum()
    logger.info(f"NaNs in features:  {nan_total}  (should be 0)")

    nan_per_col = df[FEATURE_COLUMNS].isna().sum()
    problem_cols = nan_per_col[nan_per_col > 0]
    if not problem_cols.empty:
        logger.info(f"  ⚠ Columns with NaNs:\n{problem_cols}")

    # Class balance
    if "target" in df.columns:
        balance = df["target"].value_counts(normalize=True).mul(100).round(1)
        logger.info(f"\nClass balance (target):\n{balance.to_string()}")

    # Feature–target correlation
    if "target" in df.columns:
        corr = (
            df[FEATURE_COLUMNS + ["target"]]
            .corr()["target"]
            .drop("target")
            .abs()
            .sort_values(ascending=False)
        )
        logger.info(f"\nTop-10 features by |correlation| with target:")
        logger.info(corr.head(10).to_string())

    # Target return stats
    if "target_return" in df.columns:
        logger.info(f"\ntarget_return summary:")
        logger.info(df["target_return"].describe().to_string())

    logger.info(f"\n{separator}\n")


# ===========================================================================
# ORCHESTRATOR — run_feature_pipeline()
# ===========================================================================

def run_feature_pipeline(
    ohlcv_table: str,
    ohlcv_schema: str = "data_binance",
    sentiment_schema: str = "reddit",
    target_interval: str = "1h",
    posts_table: str = "posts_sentiment_hourly",
    comments_table: str = "comments_sentiment_hourly",
    db_output_table: str = "ml_features",
    db_output_schema: str = "data_binance",
) -> pd.DataFrame:
    """
    End-to-end orchestrator.  Runs all 11 steps in order.

    Args:
        ohlcv_table      : DB table name for raw 1-minute OHLCV.
        ohlcv_schema     : DB schema for OHLCV data (default: 'data_binance').
        sentiment_schema : DB schema for Reddit sentiment tables (default: 'reddit').
        target_interval  : Resample interval for OHLCV ('1h', '4h', ...).
        posts_table      : DB table for hourly post sentiment.
        comments_table   : DB table for hourly comment sentiment.
        db_output_table  : Table name to write final features.
        db_output_schema : Schema to write final features into (default: 'data_binance').

    Returns:
        final_features_df : ML-ready DataFrame.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    ohlcv_df, posts_df, comments_df = load_data(
        ohlcv_table, posts_table, comments_table,
        ohlcv_schema=ohlcv_schema,
        sentiment_schema=sentiment_schema,
    )

    # ── 2. Sentiment features (independently) ────────────────────────────────
    posts_feat    = create_sentiment_features(posts_df,    prefix="post")
    comments_feat = create_sentiment_features(comments_df, prefix="comment")

    # ── 3. Combine sentiment sources ─────────────────────────────────────────
    sentiment_df = combine_sentiment(posts_feat, comments_feat)

    # ── 4. OHLCV features ────────────────────────────────────────────────────
    ohlcv_feat_df = build_ohlcv_features(ohlcv_df, target_interval=target_interval)

    # ── 5. Merge ──────────────────────────────────────────────────────────────
    merged_df = merge_sentiment_market(ohlcv_feat_df, sentiment_df)

    # ── 6. Alpha features ─────────────────────────────────────────────────────
    merged_df = build_alpha_features(merged_df)

    # ── 7. Targets ────────────────────────────────────────────────────────────
    merged_df = build_targets(merged_df)

    # ── 8. Select feature set ────────────────────────────────────────────────
    final_features_df = select_final_features(merged_df)

    # ── 9. Final clean ────────────────────────────────────────────────────────
    final_features_df = final_clean(final_features_df)

    # ── 10. Save outputs ──────────────────────────────────────────────────────
    save_outputs(
        final_features_df,
        db_table=db_output_table,
        schema=db_output_schema,
    )

    # ── 11. Diagnostics ───────────────────────────────────────────────────────
    print_diagnostics(final_features_df)

    return final_features_df


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    features = run_feature_pipeline(
        ohlcv_table="btc_1m",
        ohlcv_schema="data_binance",       # OHLCV lives here
        sentiment_schema="reddit",          # both sentiment tables live here
        target_interval="1h",
        posts_table="posts_sentiment_hourly",
        comments_table="comments_sentiment_hourly",
        db_output_table="ml_features",
        db_output_schema="reddit",    # write output alongside OHLCV
    )
    logger.info(f"\nDone. Final shape: {features.shape}")