import pandas as pd
import numpy as np
from datetime import datetime, timezone
import torch
from transformers import pipeline
import logging
import warnings

warnings.filterwarnings("ignore")

# 🔥 IMPORT YOUR ADVANCED CLEANER
from TradeX.sentiments.data.data_cleaner import (
    apply_cleaning_to_df,
    deduplicate_exact,
    deduplicate_near
)

from TradeX.utils.db.utils import read_df_from_db, save_df_to_db, fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger

# ================================================================================
# LOGGING SETUP
# ================================================================================
logger = get_logger("reddit_sentiment_pipeline")

# ================================================================================
# CONFIG
# ================================================================================
SCHEMA = "reddit"
POSTS_TABLE = "reddit_posts"
COMMENTS_TABLE = "reddit_comments"

POSTS_SENTIMENT_TABLE = "posts_sentiment"
COMMENTS_SENTIMENT_TABLE = "comments_sentiment"
POSTS_SENTIMENT_AGG_TABLE = "posts_sentiment_hourly"
COMMENTS_SENTIMENT_AGG_TABLE = "comments_sentiment_hourly"

OHLCV_TABLE = "btc_1m"
OHLCV_TIME_COLUMN = "datetime"
OHLCV_SCHEMA = "data_binance"

EXCLUDED_SUBS = ["ethereum", "ethtrader"]
BTC_PATTERN = r"\b(btc|bitcoin)\b|\$btc"

# ✅ UPGRADED: FinBERT replaces twitter-roberta
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ UPGRADED: FinBERT uses human-readable labels directly
SENTIMENT_MAPPING = {
    "positive": 1,
    "neutral":  0,
    "negative": -1
}

# ================================================================================
# DEVICE INFO
# ================================================================================
def log_device_info():
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️ Using CPU")

# ================================================================================
# MODEL
# ================================================================================
def load_sentiment_model():
    logger.info(f"📥 Loading model: {MODEL_NAME}")

    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=0 if DEVICE == "cuda" else -1,
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=MAX_LENGTH
    )

# ================================================================================
# DATA LOADING
# ================================================================================
def load_data():
    posts_df = read_df_from_db(POSTS_TABLE, SCHEMA)
    comments_df = read_df_from_db(COMMENTS_TABLE, SCHEMA)

    logger.info(f"Posts: {len(posts_df):,}")
    logger.info(f"Comments: {len(comments_df):,}")

    return posts_df, comments_df

# ================================================================================
# DATE RANGE
# ================================================================================
def compute_date_range(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> tuple:
    """
    Derives the global start and end timestamps from both posts and comments.

    Converts post_time and comment_time to UTC-aware datetimes, then takes
    the earliest and latest timestamps across both tables.

    Returns:
        (start_date, end_date) as timezone-aware pandas Timestamps (UTC)
    """
    posts_times = pd.to_datetime(posts_df["post_time"], utc=True)
    comments_times = pd.to_datetime(comments_df["comment_time"], utc=True)

    all_times = pd.concat([posts_times, comments_times], ignore_index=True)

    start_date = all_times.min()
    end_date = all_times.max()

    logger.info(f"📅 Date range — start: {start_date}  |  end: {end_date}")

    return start_date, end_date

# ================================================================================
# OHLCV LOADING
# ================================================================================
def load_ohlcv(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fetches BTC 1-minute OHLCV data for the computed date range.

    Args:
        start_date: UTC-aware start timestamp
        end_date:   UTC-aware end timestamp

    Returns:
        ohlcv_df: DataFrame containing BTC OHLCV rows within the range
    """
    logger.info(f"📈 Fetching OHLCV from '{OHLCV_TABLE}' [{start_date} → {end_date}]")

    ohlcv_df = fetch_ohlcv_df(
        table_name=OHLCV_TABLE,
        schema=OHLCV_SCHEMA,
        time_column=OHLCV_TIME_COLUMN,
        start_date=start_date,
        end_date=end_date
    )

    logger.info(f"✅ OHLCV rows fetched: {len(ohlcv_df):,}")

    return ohlcv_df

# ================================================================================
# FILTERING
# ================================================================================
def filter_subreddits(posts_df, comments_df):
    posts_df = posts_df[~posts_df["subreddit"].str.lower().isin(EXCLUDED_SUBS)].copy()
    comments_df = comments_df[~comments_df["subreddit"].str.lower().isin(EXCLUDED_SUBS)].copy()
    return posts_df.reset_index(drop=True), comments_df.reset_index(drop=True)

def filter_btc(posts_df, comments_df):
    posts_df = posts_df[
        posts_df["title"].str.contains(BTC_PATTERN, case=False, regex=True, na=False)
    ]
    comments_df = comments_df[
        comments_df["comment_text"].str.contains(BTC_PATTERN, case=False, regex=True, na=False)
    ]
    return posts_df.reset_index(drop=True), comments_df.reset_index(drop=True)

# ================================================================================
# SENTIMENT CORE
# ================================================================================
def apply_sentiment_analysis(texts, sentiment_pipeline):
    """
    Runs FinBERT inference over a list of texts.

    Empty strings are skipped and filled with a neutral default
    {"label": "neutral", "score": 0.5} — consistent with FinBERT label format.
    """
    NEUTRAL_DEFAULT = {"label": "neutral", "score": 0.5}

    valid_texts = [t for t in texts if len(t) > 0]

    if not valid_texts:
        return [NEUTRAL_DEFAULT] * len(texts)

    results = sentiment_pipeline(valid_texts)

    # ✅ FinBERT returns lowercase labels: "positive", "neutral", "negative"
    # Normalise to lowercase defensively in case of future model variation
    for r in results:
        r["label"] = r["label"].lower()

    out = []
    idx = 0
    for t in texts:
        if len(t) > 0:
            out.append(results[idx])
            idx += 1
        else:
            out.append(NEUTRAL_DEFAULT)

    return out

# ================================================================================
# MAIN SENTIMENT FUNCTION
# ================================================================================
def add_sentiment_to_df(df, text_column, time_column,sentiment_pipeline):

    logger.info(f"🧹 Cleaning {len(df):,} rows...")
    df = df.sort_values(time_column).reset_index(drop=True)

    # 🔥 CLEAN + FEATURES
    logger.info(f"Before drop: earliest={df[time_column].min()}, latest={df[time_column].max()}")
    df = apply_cleaning_to_df(
        df,
        text_column=text_column,
        extract_features=True,
        drop_invalid=True
    )
    logger.info(f"After drop:  earliest={df[time_column].min()}, latest={df[time_column].max()}")

    logger.info(f"After cleaning: {len(df):,}")

    # 🔥 DEDUP
    df = deduplicate_exact(df)

    if len(df) < 50000:
        df = deduplicate_near(df)

    logger.info(f"After dedup: {len(df):,}")

    texts = df["cleaned_text"].tolist()

    logger.info("🔍 Running sentiment (FinBERT)...")

    results = apply_sentiment_analysis(texts, sentiment_pipeline)

    # ✅ SENTIMENT_MAPPING now keyed on FinBERT string labels
    df["sentiment_score"] = [SENTIMENT_MAPPING[r["label"]] for r in results]
    df["sentiment_confidence"] = [r["score"] for r in results]

    df["sentiment_label"] = df["sentiment_score"].map({
        -1: "negative",
        0: "neutral",
        1: "positive"
    })

    return df

# ================================================================================
# AGGREGATION (ENHANCED)
# ================================================================================
def aggregate_sentiment_hourly(df, time_column):

    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True)

    df["hour"] = df[time_column].dt.floor("1H")

    agg = df.groupby("hour").agg({
        "sentiment_score": ["mean", "std"],
        "sentiment_confidence": "mean",

        # 🔥 ALPHA FEATURES
        "emoji_count": "mean",
        "caps_ratio": "mean",
        "punct_intensity": "mean",
        "spam_score": "mean",
        "token_count": "mean",

        df.columns[0]: "count"
    }).reset_index()

    agg.columns = ["_".join(col).strip("_") for col in agg.columns]

    agg.rename(columns={
        "hour": "time_window",
        "sentiment_score_mean": "mean_sentiment",
        "sentiment_score_std": "std_sentiment"
    }, inplace=True)

    return agg

# ================================================================================
# DTYPE FIXES FOR SQL
# ================================================================================
def fix_dtypes_for_sql(df):
    df = df.copy()

    # 🔥 FIX simhash (uint64 → string)
    if "simhash" in df.columns:
        df["simhash"] = df["simhash"].astype(str)

    return df

# ================================================================================
# SAVE
# ================================================================================
def save_sentiment_to_db(posts_df, comments_df, posts_agg, comments_agg):

    posts_df = fix_dtypes_for_sql(posts_df)
    comments_df = fix_dtypes_for_sql(comments_df)
    save_df_to_db(posts_df, POSTS_SENTIMENT_TABLE, SCHEMA, "post_time", True)
    save_df_to_db(comments_df, COMMENTS_SENTIMENT_TABLE, SCHEMA, "comment_time", True)
    save_df_to_db(posts_agg, POSTS_SENTIMENT_AGG_TABLE, SCHEMA, "time_window", True)
    save_df_to_db(comments_agg, COMMENTS_SENTIMENT_AGG_TABLE, SCHEMA, "time_window", True)

# ================================================================================
# MAIN
# ================================================================================
def main(apply_btc_filter=True, save_to_database=True):

    logger.info("🚀 START PIPELINE")

    log_device_info()

    # ── 1. Load raw data ────────────────────────────────────────────────────────
    posts_df, comments_df = load_data()

    # ── 2. Compute global date range (before filtering to keep full span) ───────
    start_date, end_date = compute_date_range(posts_df, comments_df)

    # ── 3. Fetch BTC OHLCV for that range ──────────────────────────────────────
    ohlcv_df = load_ohlcv(start_date, end_date)

    # ── 4. Filter subreddits / BTC mentions ────────────────────────────────────
    posts_df, comments_df = filter_subreddits(posts_df, comments_df)

    if apply_btc_filter:
        posts_df, comments_df = filter_btc(posts_df, comments_df)

    # ── 5. Load FinBERT and run sentiment ──────────────────────────────────────
    model = load_sentiment_model()

    posts_df = add_sentiment_to_df(posts_df, "title","post_time", model)
    comments_df = add_sentiment_to_df(comments_df, "comment_text","comment_time", model)

    # ── 6. Hourly aggregation ──────────────────────────────────────────────────
    posts_agg = aggregate_sentiment_hourly(posts_df, "post_time")
    comments_agg = aggregate_sentiment_hourly(comments_df, "comment_time")

    # ── 7. Persist to DB ───────────────────────────────────────────────────────
    if save_to_database:
        save_sentiment_to_db(posts_df, comments_df, posts_agg, comments_agg)

    logger.info("✅ DONE")

    return {
        "posts": posts_df,
        "comments": comments_df,
        "posts_agg": posts_agg,
        "comments_agg": comments_agg,
        "ohlcv": ohlcv_df
    }

# ================================================================================
# RUN
# ================================================================================
if __name__ == "__main__":

    results = main()

    logger.info(results["posts"].head())
    logger.info(results["posts_agg"].head())
    logger.info(results["ohlcv"].head())