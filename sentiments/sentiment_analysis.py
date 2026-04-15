import pandas as pd
import numpy as np
from datetime import datetime
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

from TradeX.utils.db.utils import read_df_from_db, save_df_to_db
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

EXCLUDED_SUBS = ["ethereum", "ethtrader"]
BTC_PATTERN = r"\b(btc|bitcoin)\b|\$btc"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTIMENT_MAPPING = {
    "LABEL_0": -1,
    "LABEL_1": 0,
    "LABEL_2": 1
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
    valid_texts = [t for t in texts if len(t) > 0]

    if not valid_texts:
        return [{"label": "LABEL_1", "score": 0.5}] * len(texts)

    results = sentiment_pipeline(valid_texts)

    out = []
    idx = 0
    for t in texts:
        if len(t) > 0:
            out.append(results[idx])
            idx += 1
        else:
            out.append({"label": "LABEL_1", "score": 0.5})

    return out

# ================================================================================
# MAIN SENTIMENT FUNCTION (UPDATED)
# ================================================================================
def add_sentiment_to_df(df, text_column, sentiment_pipeline):

    logger.info(f"🧹 Cleaning {len(df):,} rows...")

    # 🔥 CLEAN + FEATURES
    df = apply_cleaning_to_df(
        df,
        text_column=text_column,
        extract_features=True,
        drop_invalid=True
    )

    logger.info(f"After cleaning: {len(df):,}")

    # 🔥 DEDUP
    df = deduplicate_exact(df)

    if len(df) < 50000:
        df = deduplicate_near(df)

    logger.info(f"After dedup: {len(df):,}")

    texts = df["cleaned_text"].tolist()

    logger.info("🔍 Running sentiment...")

    results = apply_sentiment_analysis(texts, sentiment_pipeline)

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
    df[time_column] = pd.to_datetime(df[time_column])

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
# SAVE
# ================================================================================
def save_sentiment_to_db(posts_df, comments_df, posts_agg, comments_agg):

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

    posts_df, comments_df = load_data()

    posts_df, comments_df = filter_subreddits(posts_df, comments_df)

    if apply_btc_filter:
        posts_df, comments_df = filter_btc(posts_df, comments_df)

    model = load_sentiment_model()

    posts_df = add_sentiment_to_df(posts_df, "title", model)
    comments_df = add_sentiment_to_df(comments_df, "comment_text", model)

    posts_agg = aggregate_sentiment_hourly(posts_df, "post_time")
    comments_agg = aggregate_sentiment_hourly(comments_df, "comment_time")

    if save_to_database:
        save_sentiment_to_db(posts_df, comments_df, posts_agg, comments_agg)

    logger.info("✅ DONE")

    return {
        "posts": posts_df,
        "comments": comments_df,
        "posts_agg": posts_agg,
        "comments_agg": comments_agg
    }

# ================================================================================
# RUN
# ================================================================================
if __name__ == "__main__":

    results = main()

    logger.info(f"{results["posts"].head()}")
    print(f"{results["posts_agg"].head()}")