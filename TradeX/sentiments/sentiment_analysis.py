from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import torch
from transformers import pipeline as hf_pipeline
import logging
import warnings

warnings.filterwarnings("ignore")

from TradeX.sentiments.data.data_cleaner import (
    apply_cleaning_to_df,
    deduplicate_exact,
    deduplicate_near,
)
from TradeX.utils.db.utils import read_df_from_db, save_df_to_db
from TradeX.utils.common.logs import get_logger

# Import coin registry from schema (single source of truth)
from app.schemas.sentiment_schema import COIN_CONFIG

# ================================================================================
# LOGGING
# ================================================================================
logger = get_logger("reddit_sentiment_pipeline")

# ================================================================================
# STATIC CONFIG
# ================================================================================
SCHEMA           = "reddit"
POSTS_TABLE      = "reddit_posts"
COMMENTS_TABLE   = "reddit_comments"

# Subreddits excluded regardless of coin
EXCLUDED_SUBS: set[str] = set()   # extend if needed per-project

MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 32
MAX_LENGTH  = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SENTIMENT_MAPPING = {
    "positive":  1,
    "neutral":   0,
    "negative": -1,
}


# ================================================================================
# HELPERS
# ================================================================================

def _table_names(coin: str) -> dict[str, str]:
    """Return all four output table names for a given coin."""
    return {
        "posts_sentiment":          f"{coin}_posts_sentiment",
        "comments_sentiment":       f"{coin}_comments_sentiment",
        "posts_sentiment_hourly":   f"{coin}_posts_sentiment_hourly",
        "comments_sentiment_hourly": f"{coin}_comments_sentiment_hourly",
    }


def log_device_info() -> None:
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️  Using CPU")


# ================================================================================
# MODEL
# ================================================================================

def load_sentiment_model():
    logger.info(f"📥 Loading FinBERT: {MODEL_NAME}")
    return hf_pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=0 if DEVICE == "cuda" else -1,
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ================================================================================
# DATA LOADING
# ================================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    posts_df    = read_df_from_db(POSTS_TABLE,    SCHEMA)
    comments_df = read_df_from_db(COMMENTS_TABLE, SCHEMA)
    logger.info(f"Posts loaded:    {len(posts_df):,}")
    logger.info(f"Comments loaded: {len(comments_df):,}")
    return posts_df, comments_df


# ================================================================================
# FILTERING  (now coin-aware)
# ================================================================================

def filter_subreddits(
    posts_df:    pd.DataFrame,
    comments_df: pd.DataFrame,
    excluded:    set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows from globally excluded subreddits."""
    excl = {s.lower() for s in (excluded or EXCLUDED_SUBS)}
    if excl:
        posts_df    = posts_df   [~posts_df   ["subreddit"].str.lower().isin(excl)].copy()
        comments_df = comments_df[~comments_df["subreddit"].str.lower().isin(excl)].copy()
    return posts_df.reset_index(drop=True), comments_df.reset_index(drop=True)


def filter_coin(
    posts_df:    pd.DataFrame,
    comments_df: pd.DataFrame,
    coin:        str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only posts / comments that mention the given coin.

    Uses COIN_CONFIG[coin]["pattern"] for text matching and
    COIN_CONFIG[coin]["native_subs"] to skip the pattern check
    for subreddits where every post is implicitly about that coin.
    """
    cfg         = COIN_CONFIG[coin]
    pattern     = cfg["pattern"]
    native_subs = {s.lower() for s in cfg["native_subs"]}

    # Posts
    native_mask_p  = posts_df["subreddit"].str.lower().isin(native_subs)
    mention_mask_p = posts_df["title"].str.contains(pattern, case=False, regex=True, na=False)
    posts_df       = posts_df[native_mask_p | mention_mask_p].reset_index(drop=True)

    # Comments
    native_mask_c  = comments_df["subreddit"].str.lower().isin(native_subs)
    mention_mask_c = comments_df["comment_text"].str.contains(
        pattern, case=False, regex=True, na=False
    )
    comments_df = comments_df[native_mask_c | mention_mask_c].reset_index(drop=True)

    logger.info(
        f"After {coin.upper()} filter — posts: {len(posts_df):,} | "
        f"comments: {len(comments_df):,}"
    )
    return posts_df, comments_df


# ================================================================================
# SENTIMENT CORE
# ================================================================================

def apply_sentiment_analysis(texts: list[str], sentiment_pipeline) -> list[dict]:
    """
    Run FinBERT inference over a list of texts.
    Empty strings are skipped; they receive a neutral default.
    """
    NEUTRAL_DEFAULT = {"label": "neutral", "score": 0.5}

    valid_texts = [t for t in texts if len(t) > 0]
    if not valid_texts:
        return [NEUTRAL_DEFAULT] * len(texts)

    results = sentiment_pipeline(valid_texts)
    for r in results:
        r["label"] = r["label"].lower()   # normalise defensively

    out, idx = [], 0
    for t in texts:
        if len(t) > 0:
            out.append(results[idx]); idx += 1
        else:
            out.append(NEUTRAL_DEFAULT)
    return out


def add_sentiment_to_df(
    df:                 pd.DataFrame,
    text_column:        str,
    time_column:        str,
    sentiment_pipeline,
) -> pd.DataFrame:
    logger.info(f"🧹 Cleaning {len(df):,} rows...")
    df = df.sort_values(time_column).reset_index(drop=True)

    df = apply_cleaning_to_df(
        df,
        text_column=text_column,
        extract_features=True,
        drop_invalid=True,
    )
    logger.info(f"After cleaning: {len(df):,}")

    df = deduplicate_exact(df)
    if len(df) < 50_000:
        df = deduplicate_near(df)
    logger.info(f"After dedup: {len(df):,}")

    texts   = df["cleaned_text"].tolist()
    logger.info("🔍 Running FinBERT inference...")
    results = apply_sentiment_analysis(texts, sentiment_pipeline)

    df["sentiment_score"]      = [SENTIMENT_MAPPING[r["label"]] for r in results]
    df["sentiment_confidence"] = [r["score"]                     for r in results]
    df["sentiment_label"]      = df["sentiment_score"].map({
        -1: "negative", 0: "neutral", 1: "positive"
    })
    return df


# ================================================================================
# AGGREGATION
# ================================================================================

def _std_pop(x: pd.Series) -> float:
    return x.std(ddof=0)

_std_pop.__name__ = "std_pop"


def aggregate_sentiment_hourly(
    df:          pd.DataFrame,
    time_column: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "time_window", "mean_sentiment", "std_sentiment",
            "sentiment_confidence_mean", "emoji_count_mean", "caps_ratio_mean",
            "punct_intensity_mean", "spam_score_mean", "token_count_mean", "post_id_count",
        ])

    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    df["hour"]      = df[time_column].dt.floor("1H")

    agg = (
        df.groupby("hour", as_index=False)
        .agg(
            mean_sentiment              =("sentiment_score",      "mean"),
            std_sentiment               =("sentiment_score",      _std_pop),
            sentiment_confidence_mean   =("sentiment_confidence", "mean"),
            emoji_count_mean            =("emoji_count",          "mean"),
            caps_ratio_mean             =("caps_ratio",           "mean"),
            punct_intensity_mean        =("punct_intensity",      "mean"),
            spam_score_mean             =("spam_score",           "mean"),
            token_count_mean            =("token_count",          "mean"),
            post_id_count               =("sentiment_score",      "count"),
        )
        .rename(columns={"hour": "time_window"})
    )
    return agg


# ================================================================================
# DTYPE FIXES
# ================================================================================

def fix_dtypes_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "simhash" in df.columns:
        df["simhash"] = df["simhash"].astype(str)
    return df


# ================================================================================
# SAVE
# ================================================================================

def save_sentiment_to_db(
    posts_df:     pd.DataFrame,
    comments_df:  pd.DataFrame,
    posts_agg:    pd.DataFrame,
    comments_agg: pd.DataFrame,
    coin:         str,
) -> None:
    tables = _table_names(coin)

    posts_df    = fix_dtypes_for_sql(posts_df)
    comments_df = fix_dtypes_for_sql(comments_df)

    save_df_to_db(posts_df,    tables["posts_sentiment"],          SCHEMA, "post_time",    True)
    save_df_to_db(comments_df, tables["comments_sentiment"],       SCHEMA, "comment_time", True)
    save_df_to_db(posts_agg,   tables["posts_sentiment_hourly"],   SCHEMA, "time_window",  True)
    save_df_to_db(comments_agg,tables["comments_sentiment_hourly"],SCHEMA, "time_window",  True)

    logger.info(f"💾 Saved sentiment tables for coin='{coin}'")


# ================================================================================
# PUBLIC ENTRY-POINT  (called by sentiment_service)
# ================================================================================

def run_pipeline(
    coin:             str  = "btc",
    apply_coin_filter: bool = True,
    save_to_database:  bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Full pipeline for a given coin.

    Args:
        coin:              Coin id from COIN_CONFIG (e.g. "btc", "eth", "sol")
        apply_coin_filter: If True, filter posts/comments to coin mentions only
        save_to_database:  If True, persist results to the DB

    Returns:
        dict with keys: posts, comments, posts_agg, comments_agg
    """
    if coin not in COIN_CONFIG:
        raise ValueError(
            f"Unknown coin '{coin}'. "
            f"Supported: {list(COIN_CONFIG.keys())}"
        )

    logger.info(f"🚀 START PIPELINE  coin={coin.upper()}")
    log_device_info()

    # 1. Raw data ----------------------------------------------------------------
    posts_df, comments_df = load_data()

    # 2. Filtering ---------------------------------------------------------------
    posts_df, comments_df = filter_subreddits(posts_df, comments_df)
    if apply_coin_filter:
        posts_df, comments_df = filter_coin(posts_df, comments_df, coin)

    if posts_df.empty and comments_df.empty:
        raise ValueError(
            f"No posts or comments found mentioning coin='{coin}' after filtering. "
            f"Try scraping more subreddits or run without coin filter."
        )

    # 3. FinBERT -----------------------------------------------------------------
    model = load_sentiment_model()

    if not posts_df.empty:
        posts_df = add_sentiment_to_df(posts_df, "title", "post_time", model)
    else:
        logger.warning(f"No posts found for coin='{coin}' — skipping post sentiment.")

    if not comments_df.empty:
        comments_df = add_sentiment_to_df(comments_df, "comment_text", "comment_time", model)
    else:
        logger.warning(f"No comments found for coin='{coin}' — skipping comment sentiment.")

    # 4. Hourly aggregation ------------------------------------------------------
    posts_agg    = aggregate_sentiment_hourly(posts_df,    "post_time")    if not posts_df.empty    else pd.DataFrame()
    comments_agg = aggregate_sentiment_hourly(comments_df, "comment_time") if not comments_df.empty else pd.DataFrame()

    # 5. Persist -----------------------------------------------------------------
    if save_to_database:
        save_sentiment_to_db(posts_df, comments_df, posts_agg, comments_agg, coin)

    logger.info(f"✅ DONE  coin={coin.upper()}")
    return {
        "posts":       posts_df,
        "comments":    comments_df,
        "posts_agg":   posts_agg,
        "comments_agg":comments_agg,
    }


# ================================================================================
# CLI  (python sentiment_analysis.py --coin eth)
# ================================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TradeX Sentiment Pipeline")
    parser.add_argument(
        "--coin", default="btc",
        help=f"Coin to analyse. Options: {list(COIN_CONFIG.keys())}",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Disable coin-specific post filtering",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip database save",
    )
    args = parser.parse_args()

    results = run_pipeline(
        coin=args.coin,
        apply_coin_filter=not args.no_filter,
        save_to_database=not args.no_save,
    )
    logger.info(results["posts"].head())
    logger.info(results["posts_agg"].head())