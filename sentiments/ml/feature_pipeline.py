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
SCHEMA               = "reddit"
POSTS_TABLE          = "posts_sentiment_hourly"
COMMENTS_TABLE       = "comments_sentiment_hourly"

OHLCV_TABLE          = "btc_1m"
OHLCV_SCHEMA         = "data_binance"
OHLCV_TIME_COLUMN    = "datetime"
OHLCV_RESAMPLE_FREQ  = "1h"

# FIX-1: Document and enforce resampling convention.
# resample_ohlcv MUST use label='left', closed='left' so that the timestamp
# represents the START of the 1-hour bucket, then we shift it +1h to mark the
# bar-close time (the moment when all that bar's information is known).
# If resample_ohlcv uses label='right' internally, remove the +1h shift below.
OHLCV_RESAMPLE_LABEL    = "left"   # expected label convention inside resample_ohlcv
OHLCV_RESAMPLE_SHIFT_1H = True     # set False only if resample_ohlcv already labels by close

ML_FEATURES_TABLE    = "ml_features"

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

    posts_df    = read_df_from_db(POSTS_TABLE,    SCHEMA)
    comments_df = read_df_from_db(COMMENTS_TABLE, SCHEMA)

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

    fallback = [c for c in df.columns if c != "time_window"][0]
    logger.warning(f"Volume column not found by name; using '{fallback}' as proxy.")
    return fallback


def create_sentiment_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute sentiment feature columns for a single sentiment source.

    TIME ALIGNMENT NOTE:
        time_window here is still the RAW bucket start (activity in [T, T+1h)).
        The +1h alignment shift that maps it to bar-close time is applied ONCE
        in merge_sentiment_market, NOT here.  That keeps this function clean
        and avoids double-shifting if it is ever called in isolation.

    All rolling/EMA operations are strictly backward-looking (no center=True,
    no look-ahead windows).
    """
    df = df.copy().sort_values("time_window").reset_index(drop=True)

    # ── Resolve column names defensively ────────────────────────────────────────
    mean_col   = "mean_sentiment"
    std_col    = "std_sentiment"
    conf_col   = next(
        (c for c in df.columns if "confidence" in c.lower()),
        None,
    )
    volume_col = _safe_volume_col(df)

    if mean_col not in df.columns:
        raise ValueError(f"Expected '{mean_col}' in {prefix} sentiment DataFrame.")

    # ── Core features ────────────────────────────────────────────────────────────
    out = pd.DataFrame()
    out["time_window"] = df["time_window"]

    out[f"{prefix}_mean_sentiment"] = df[mean_col]
    out[f"{prefix}_volatility"]     = df[std_col] if std_col in df.columns else 0.0
    out[f"{prefix}_volume"]         = df[volume_col]

    # Momentum = first difference; diff() is inherently backward-looking.
    out[f"{prefix}_momentum"] = df[mean_col].diff().fillna(0)

    # FIX-4: EMA — adjust=False is causal (each value only uses past data).
    # min_periods=1 avoids NaNs at series start without introducing future info.
    out[f"{prefix}_ema"] = (
        df[mean_col]
        .ewm(span=EMA_SPAN, adjust=False, min_periods=1)
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

    NOTE: time_window values here are still the RAW bucket-start timestamps.
    The leakage-preventing +1h shift happens in merge_sentiment_market so that
    ffill (applied right after the shift) propagates only into the future,
    never backward.
    """
    logger.info("🔗 Combining post + comment sentiment features…")

    merged = pd.merge(posts_feat, comments_feat, on="time_window", how="outer")
    merged = merged.sort_values("time_window").reset_index(drop=True)

    # Fill gaps introduced by the outer join with 0 BEFORE the shift.
    # This is safe: we are only filling within the sentiment domain where
    # every row's time_window is its own bucket-start — no cross-row leakage.
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

    FIX-1 — OHLCV RESAMPLING LEAKAGE
    ──────────────────────────────────
    Pandas resample with label='left', closed='left' labels each bucket by its
    START time (e.g. 13:00 for the candle covering 13:00–13:59).  At time 13:00
    a model is NOT yet allowed to see this candle (it hasn't closed).  The
    candle only becomes available at 14:00.

    Fix: after resampling, shift the datetime index +1h so the label equals
    the bar-CLOSE time.  Downstream code that merges on datetime==time_window
    will then correctly align bars to the moment they become observable.

    Assumption: resample_ohlcv uses label='left', closed='left' internally.
    If it uses label='right' the +1h shift must be removed (see OHLCV_RESAMPLE_SHIFT_1H).
    """
    logger.info(f"📊 Resampling OHLCV to {OHLCV_RESAMPLE_FREQ}…")

    ohlcv_h = resample_ohlcv(ohlcv_1m, OHLCV_RESAMPLE_FREQ).copy()
    ohlcv_h[OHLCV_TIME_COLUMN] = pd.to_datetime(ohlcv_h[OHLCV_TIME_COLUMN], utc=True)
    ohlcv_h = ohlcv_h.sort_values(OHLCV_TIME_COLUMN).reset_index(drop=True)

    # FIX-1: Shift bar label from bucket-START to bucket-CLOSE (+1 hour).
    # After this, datetime=14:00 means "the candle that closed at 14:00"
    # (i.e. it aggregated 1-min bars from 13:00 to 13:59 inclusive).
    if OHLCV_RESAMPLE_SHIFT_1H:
        ohlcv_h[OHLCV_TIME_COLUMN] = (
            ohlcv_h[OHLCV_TIME_COLUMN] + pd.Timedelta("1h")
        )
        logger.info(
            "  FIX-1 applied: resampled bar timestamps shifted +1h "
            "(label now = bar-close time, preventing future candle leakage)."
        )

    logger.info(f"  Hourly candles: {len(ohlcv_h):,}")

    # ── Market features — all strictly backward-looking ─────────────────────────

    # returns: pct_change uses close[T] and close[T-1], both already observed.
    ohlcv_h["returns"] = ohlcv_h["close"].pct_change().fillna(0)

    # FIX-5: Volatility — rolling std with no center=True; window looks only
    # backward.  The current bar T is included (it has just closed), which is
    # correct — its return is known at T.
    ohlcv_h["volatility"] = (
        ohlcv_h["returns"]
        .rolling(VOL_WINDOW, min_periods=1)
        .std()
        .fillna(0)
    )

    ohlcv_h["volume_change"] = ohlcv_h["volume"].pct_change().fillna(0)

    # FIX-10: Replace raw price diff (price_momentum) with a normalized version
    # (rate-of-change / ROC) so it is on the same dimensionless scale as
    # sentiment features and avoids scale mismatch in divergence & interactions.
    # close.pct_change() is already computed as 'returns'; reuse it.
    ohlcv_h["price_momentum"] = ohlcv_h["returns"]   # normalized ROC, not raw diff

    return ohlcv_h


# ================================================================================
# STEP 7 — MERGE SENTIMENT + MARKET
# ================================================================================

def merge_sentiment_market(
    ohlcv_h: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aligns sentiment to market bars and LEFT JOINs them.

    FIX-2 — SENTIMENT TIME ALIGNMENT
    ──────────────────────────────────
    Sentiment row with time_window=T covers Reddit activity in [T, T+1h).
    That activity is NOT fully observable until T+1h.
    To attach it to the market bar that CLOSES at T+1h, we shift:
        time_window_aligned = time_window + 1h

    After the shift, time_window_aligned=14:00 means
    "sentiment from Reddit posts created between 13:00 and 13:59"
    which is exactly the information available at 14:00 (bar close).

    FIX-3 — FORWARD FILL ORDER
    ────────────────────────────
    ffill is applied AFTER the +1h shift.  This means:
      – If sentiment for 14:00 is missing, we propagate the last known value
        from 13:00 (or earlier), which is past data.  ✅
      – We never propagate a 14:00 sentiment value backward to fill 13:00.  ✅
    """
    logger.info("🔀 Merging OHLCV + sentiment…")

    sentiment_df = sentiment_df.copy()

    # FIX-2: Shift sentiment timestamps by +1h before merge.
    # Reason: activity in [T, T+1h) becomes known at T+1h (bar close).
    sentiment_df["time_window"] = (
        sentiment_df["time_window"] + pd.Timedelta("1h")
    )
    logger.info(
        "  FIX-2 applied: sentiment time_window shifted +1h "
        "(activity in [T-1h,T) now aligns with market bar closing at T)."
    )

    # Enforce: no sentiment timestamp should exceed the latest market bar.
    # Any sentiment row that would "peer into the future" relative to OHLCV is
    # silently dropped — this prevents right-side contamination.
    max_market_ts = ohlcv_h[OHLCV_TIME_COLUMN].max()
    n_before = len(sentiment_df)
    sentiment_df = sentiment_df[sentiment_df["time_window"] <= max_market_ts]
    n_dropped = n_before - len(sentiment_df)
    if n_dropped:
        logger.warning(
            f"  FIX-2: Dropped {n_dropped} sentiment rows whose aligned "
            f"time_window exceeded latest market bar ({max_market_ts})."
        )

    # Left join: market bars are the anchor; sentiment is attached where available.
    merged = ohlcv_h.merge(
        sentiment_df,
        left_on=OHLCV_TIME_COLUMN,
        right_on="time_window",
        how="left",
    )

    if "time_window" in merged.columns:
        merged = merged.drop(columns=["time_window"])

    # FIX-3: Forward-fill AFTER the +1h shift — propagates only past sentiment.
    sentiment_cols = [c for c in merged.columns if c not in ohlcv_h.columns]
    merged[sentiment_cols] = merged[sentiment_cols].ffill()

    # Zero-fill the head of the series where no prior sentiment exists.
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0)

    logger.info(f"  Merged rows: {len(merged):,}")

    return merged.reset_index(drop=True)


# ================================================================================
# STEP 8 — ALPHA FEATURES
# ================================================================================

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-domain alpha signals.  All operations are strictly backward-
    looking relative to the merged timeline.
    """
    logger.info("⚡ Building alpha features…")

    df = df.copy()

    # FIX-6: Divergence — scale mismatch fix.
    # sentiment_combined is in [-1, +1]; returns is a small float (e.g. 0.002).
    # A raw difference is dominated by the sentiment term.
    # Normalize sentiment by its rolling std (same window as volatility) to
    # put both series on a comparable scale before differencing.
    rolling_sent_std = (
        df["sentiment_combined"]
        .rolling(VOL_WINDOW, min_periods=1)
        .std()
        .replace(0, np.nan)      # avoid div-by-zero
        .fillna(1.0)             # fallback to 1 when std is undefined
    )
    sentiment_normalized = df["sentiment_combined"] / rolling_sent_std

    rolling_ret_std = (
        df["returns"]
        .rolling(VOL_WINDOW, min_periods=1)
        .std()
        .replace(0, np.nan)
        .fillna(1.0)
    )
    returns_normalized = df["returns"] / rolling_ret_std

    # FIX-6 continued: both terms now dimensionless (z-score-like).
    df["divergence"] = (sentiment_normalized - returns_normalized).fillna(0)

    # FIX-7: Sentiment spike — use shift(1) on rolling mean so the CURRENT
    # bar's volume does not inflate (or deflate) its own spike threshold.
    # Without shift(1), the rolling mean includes the current bar, which means
    # an unusually high value at T lowers the relative spike ratio — subtle
    # leakage of T's own magnitude into T's feature.
    rolling_mean_vol = (
        df["sentiment_volume_total"]
        .shift(1)                          # FIX-7: exclude current bar
        .rolling(VOL_WINDOW, min_periods=1)
        .mean()
        .fillna(0)
    )
    df["sentiment_spike"] = (
        df["sentiment_volume_total"] > SPIKE_MULTIPLIER * rolling_mean_vol
    ).astype(int)

    # Fear / greed index — both inputs are backward-looking at this point.
    df["fear_greed_index"] = (
        df["sentiment_combined"] * df["sentiment_volume_total"]
    ).fillna(0)

    # Sentiment × price interaction.
    # .diff() on sentiment_combined and returns are both backward-looking.
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

    Both use shift(-1) on 'close', which is a PAST-only column at this stage
    (all forward-leaking operations have already been blocked upstream).

    FIX-8: target_return construction.
    Original code: df["close"].pct_change().shift(-1)
      → pct_change() gives return[T] = (close[T]-close[T-1])/close[T-1]
      → shift(-1) gives return[T+1], which is CORRECT for the next-bar return.
    However, the semantics are clearer and safer when written explicitly as
    the next bar's return = (close[T+1] - close[T]) / close[T]:
    """
    logger.info("🎯 Building target variables…")

    df = df.copy()

    # FIX-8: Explicit next-bar return avoids any ambiguity with pct_change ordering.
    df["target"]        = (df["close"].shift(-1) > df["close"]).astype(int)
    df["target_return"] = (df["close"].shift(-1) - df["close"]) / df["close"]

    # Drop the last row — it has no future bar to label.
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

    feature_cols = [c for c in ALL_FEATURES if c in final.columns]
    final[feature_cols] = final[feature_cols].fillna(0)

    final = final.sort_values(OHLCV_TIME_COLUMN).reset_index(drop=True)
    final = final.drop_duplicates(subset=[OHLCV_TIME_COLUMN]).reset_index(drop=True)

    logger.info(f"  Final shape: {final.shape}")

    return final


# ================================================================================
# STEP 11 — SCALER GUARD  (FIX-9)
# ================================================================================

class ScalerGuard:
    """
    FIX-9 — SCALING / NORMALISATION LEAKAGE PREVENTION
    ─────────────────────────────────────────────────────
    Scalers (StandardScaler, MinMaxScaler, RobustScaler …) MUST be fit ONLY
    on the training split.  Fitting on the full dataset leaks future statistics
    (mean, std, min, max) into the training window.

    Usage pattern (to be called by the model training script, NOT here):

        guard = ScalerGuard(feature_cols)
        X_train_scaled = guard.fit_transform(X_train)   # fit on train only
        X_val_scaled   = guard.transform(X_val)          # apply to val/test
        X_test_scaled  = guard.transform(X_test)

    This class is intentionally a thin wrapper that documents the contract.
    The actual scaler is sklearn-compatible (has fit / transform).
    """

    def __init__(self, feature_cols: list[str], scaler=None):
        if scaler is None:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        self.scaler       = scaler
        self.feature_cols = feature_cols
        self._fitted      = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on train split ONLY, then transform."""
        X = df[self.feature_cols].values
        result = self.scaler.fit_transform(X)
        self._fitted = True
        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform val/test using statistics from train split."""
        if not self._fitted:
            raise RuntimeError(
                "ScalerGuard: call fit_transform on the training split first."
            )
        return self.scaler.transform(df[self.feature_cols].values)


# ================================================================================
# STEP 12 — SAVE OUTPUTS
# ================================================================================

def save_outputs(final_df: pd.DataFrame) -> None:
    """
    Persists final_features_df to DB.
    """
    logger.info("💾 Saving outputs…")

    save_df_to_db(
        final_df,
        table_name=ML_FEATURES_TABLE,
        schema=SCHEMA,
        time_column=OHLCV_TIME_COLUMN,
        is_timeseries=True,
    )
    logger.info(f"  ✅ Saved to DB: {SCHEMA}.{ML_FEATURES_TABLE}")


# ================================================================================
# STEP 13 — DIAGNOSTICS
# ================================================================================

def run_diagnostics(final_df: pd.DataFrame) -> None:
    """
    Prints dataset shape, missing value audit, feature-target correlations,
    and classification target class balance.

    LEAKAGE SMOKE TEST:
      A suspiciously high feature-target correlation (|r| > 0.3 for raw features)
      is a red flag that warrants manual inspection.  Returns are logged for
      reference; they are expected to be near-zero for a properly lagged pipeline.
    """
    logger.info("=" * 72)
    logger.info("📋 DIAGNOSTICS")
    logger.info("=" * 72)

    logger.info(f"  Shape: {final_df.shape}")

    total_nans = final_df[ALL_FEATURES].isna().sum().sum()
    if total_nans == 0:
        logger.info("  ✅ Missing values: 0 (feature columns)")
    else:
        nan_report = final_df[ALL_FEATURES].isna().sum()
        nan_report = nan_report[nan_report > 0]
        logger.warning(f"  ⚠️ NaN columns:\n{nan_report}")

    if "target" in final_df.columns:
        corr = (
            final_df[ALL_FEATURES + ["target"]]
            .corr()["target"]
            .drop("target")
            .sort_values(key=abs, ascending=False)
        )
        logger.info("  📈 Top 10 feature correlations with 'target':")
        for feat, val in corr.head(10).items():
            flag = "  ⚠️ HIGH — CHECK FOR LEAKAGE" if abs(val) > 0.3 else ""
            logger.info(f"    {feat:<45} {val:+.4f}{flag}")

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
        final_features_df — ML-ready, leakage-free DataFrame.

    SCALING NOTE (FIX-9):
        This pipeline returns UN-scaled features.  Scaling must be performed
        AFTER the train/val/test split using ScalerGuard:

            guard = ScalerGuard(ALL_FEATURES)
            X_train = guard.fit_transform(train_df)
            X_val   = guard.transform(val_df)
            X_test  = guard.transform(test_df)
    """
    logger.info("🚀 FEATURE PIPELINE START")

    posts_df, comments_df = load_sentiment_data()
    start_date, end_date  = compute_date_range(posts_df, comments_df)
    ohlcv_1m              = load_ohlcv(start_date, end_date)

    logger.info("🧮 Engineering sentiment features…")
    posts_feat    = create_sentiment_features(posts_df,    prefix="post")
    comments_feat = create_sentiment_features(comments_df, prefix="comment")

    sentiment_df = combine_sentiment(posts_feat, comments_feat)
    ohlcv_h      = build_ohlcv_features(ohlcv_1m)
    merged_df    = merge_sentiment_market(ohlcv_h, sentiment_df)
    merged_df    = build_alpha_features(merged_df)
    merged_df    = build_targets(merged_df)
    final_features_df = select_final_features(merged_df)

    if save_to_database:
        save_outputs(final_features_df)

    run_diagnostics(final_features_df)

    logger.info("✅ FEATURE PIPELINE COMPLETE")
    return final_features_df


# ================================================================================
# RUN
# ================================================================================

if __name__ == "__main__":
    final_features_df = main()
    logger.info(final_features_df.head())