import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import warnings

warnings.filterwarnings("ignore")

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

# Output table names for sentiment results
POSTS_SENTIMENT_TABLE = "posts_sentiment"
COMMENTS_SENTIMENT_TABLE = "comments_sentiment"
POSTS_SENTIMENT_AGG_TABLE = "posts_sentiment_hourly"
COMMENTS_SENTIMENT_AGG_TABLE = "comments_sentiment_hourly"

# Subreddits to EXCLUDE (from filtered_data.py)
EXCLUDED_SUBS = ["ethereum", "ethtrader"]

# Optional BTC filter pattern
BTC_PATTERN = r"\b(btc|bitcoin)\b|\$btc"

# Sentiment Model Config
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
BATCH_SIZE = 32
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sentiment label mapping
SENTIMENT_MAPPING = {
    "LABEL_0": -1,  # Negative
    "LABEL_1": 0,   # Neutral
    "LABEL_2": 1    # Positive
}

# ================================================================================
# DEVICE INFO
# ================================================================================
def log_device_info():
    """Log device information for transparency."""
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("⚠️  GPU not available - using CPU (slower)")
        logger.info(f"💻 CPU Cores: {torch.get_num_threads()}")


# ================================================================================
# LOAD SENTIMENT MODEL
# ================================================================================
def load_sentiment_model():
    """
    Load pre-trained sentiment model from Hugging Face.
    
    Model: cardiffnlp/twitter-roberta-base-sentiment
    - Lightweight and fast
    - Optimized for social media text
    - No API costs
    
    Returns:
        pipeline: transformers pipeline object
    """
    logger.info(f"📥 Loading sentiment model: {MODEL_NAME}")
    logger.info(f"🔧 Using device: {DEVICE}")
    
    try:
        sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=0 if DEVICE == "cuda" else -1,  # -1 for CPU
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=MAX_LENGTH
        )
        logger.info("✅ Model loaded successfully")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# ================================================================================
# DATA LOADING
# ================================================================================
def load_data():
    """
    Load posts and comments from database.
    
    Returns:
        tuple: (posts_df, comments_df)
    """
    logger.info("📥 Loading data from database...")
    
    try:
        posts_df = read_df_from_db(
            table_name=POSTS_TABLE,
            schema=SCHEMA
        )
        
        comments_df = read_df_from_db(
            table_name=COMMENTS_TABLE,
            schema=SCHEMA
        )
        
        logger.info(f"✅ Posts loaded: {len(posts_df):,}")
        logger.info(f"✅ Comments loaded: {len(comments_df):,}")
        
        return posts_df, comments_df
    
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise


# ================================================================================
# FILTERING SUBREDDITS
# ================================================================================
def filter_subreddits(posts_df: pd.DataFrame, comments_df: pd.DataFrame):
    """
    Remove excluded subreddits from both posts and comments.
    
    Args:
        posts_df: DataFrame with posts
        comments_df: DataFrame with comments
        
    Returns:
        tuple: (filtered_posts_df, filtered_comments_df)
    """
    logger.info("🚫 Filtering excluded subreddits...")
    logger.info(f"   Excluded: {EXCLUDED_SUBS}")
    
    posts_before = len(posts_df)
    comments_before = len(comments_df)
    
    posts_filtered = posts_df[
        ~posts_df["subreddit"].str.lower().isin(EXCLUDED_SUBS)
    ].copy()
    
    comments_filtered = comments_df[
        ~comments_df["subreddit"].str.lower().isin(EXCLUDED_SUBS)
    ].copy()
    
    posts_filtered.reset_index(drop=True, inplace=True)
    comments_filtered.reset_index(drop=True, inplace=True)
    
    posts_removed = posts_before - len(posts_filtered)
    comments_removed = comments_before - len(comments_filtered)
    
    logger.info(f"✅ Posts filtered: {len(posts_filtered):,} (removed: {posts_removed:,})")
    logger.info(f"✅ Comments filtered: {len(comments_filtered):,} (removed: {comments_removed:,})")
    
    return posts_filtered, comments_filtered


# ================================================================================
# OPTIONAL BTC FILTERING
# ================================================================================
def filter_btc(posts_df: pd.DataFrame, comments_df: pd.DataFrame):
    """
    Filter to only BTC-related posts and comments.
    
    Args:
        posts_df: DataFrame with posts
        comments_df: DataFrame with comments
        
    Returns:
        tuple: (btc_posts_df, btc_comments_df)
    """
    logger.info("₿ Applying BTC filter...")
    
    posts_before = len(posts_df)
    comments_before = len(comments_df)
    
    btc_posts = posts_df[
        posts_df["title"].str.contains(BTC_PATTERN, case=False, regex=True, na=False) |
        posts_df["url"].str.contains(BTC_PATTERN, case=False, regex=True, na=False)
    ].copy()
    
    btc_comments = comments_df[
        comments_df["comment_text"].str.contains(BTC_PATTERN, case=False, regex=True, na=False)
    ].copy()
    
    btc_posts.reset_index(drop=True, inplace=True)
    btc_comments.reset_index(drop=True, inplace=True)
    
    posts_kept = len(btc_posts)
    comments_kept = len(btc_comments)
    
    logger.info(f"✅ BTC Posts: {posts_kept:,} (from {posts_before:,})")
    logger.info(f"✅ BTC Comments: {comments_kept:,} (from {comments_before:,})")
    
    return btc_posts, btc_comments


# ================================================================================
# TEXT CLEANING & VALIDATION
# ================================================================================
def clean_text(text):
    """
    Clean and validate text for sentiment analysis.
    
    Args:
        text: Input text (may be NaN or string)
        
    Returns:
        str: Cleaned text or empty string
    """
    # Handle NaN/None
    if pd.isna(text):
        return ""
    
    # Convert to string if needed
    text = str(text).strip()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Handle empty text
    if len(text) == 0:
        return ""
    
    return text


# ================================================================================
# SENTIMENT ANALYSIS (BATCHED)
# ================================================================================
def apply_sentiment_analysis(texts: list, sentiment_pipeline) -> list:
    """
    Apply sentiment analysis to a batch of texts.
    
    Uses batching for efficiency with GPU/CPU.
    Handles edge cases gracefully.
    
    Args:
        texts: List of text strings
        sentiment_pipeline: Hugging Face pipeline object
        
    Returns:
        list: List of sentiment results (dicts with 'label' and 'score')
    """
    # Filter out empty texts and track indices
    valid_indices = [i for i, text in enumerate(texts) if len(text) > 0]
    valid_texts = [texts[i] for i in valid_indices]
    
    # Handle case where all texts are empty
    if len(valid_texts) == 0:
        return [{
            "label": "LABEL_1",
            "score": 0.5,
            "is_empty": True
        } for _ in texts]
    
    try:
        # Run sentiment analysis in batches
        results_valid = sentiment_pipeline(valid_texts)
    except Exception as e:
        logger.error(f"❌ Error in sentiment analysis: {e}")
        results_valid = [
            {"label": "LABEL_1", "score": 0.5, "error": True}
            for _ in valid_texts
        ]
    
    # Reconstruct results for original text list
    results = []
    valid_idx = 0
    
    for i in range(len(texts)):
        if i in valid_indices:
            results.append(results_valid[valid_idx])
            valid_idx += 1
        else:
            # Empty text: mark as neutral
            results.append({
                "label": "LABEL_1",
                "score": 0.5,
                "is_empty": True
            })
    
    return results


# ================================================================================
# ADD SENTIMENT TO DATAFRAME
# ================================================================================
def add_sentiment_to_df(df: pd.DataFrame, text_column: str, sentiment_pipeline) -> pd.DataFrame:
    """
    Add sentiment labels and scores to a dataframe.
    
    Processes texts in batches for efficiency.
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text
        sentiment_pipeline: Hugging Face pipeline object
        
    Returns:
        pd.DataFrame: DataFrame with new sentiment columns
    """
    df = df.copy()
    
    logger.info(f"🔍 Analyzing sentiment for {len(df):,} items...")
    
    # Clean texts
    df["cleaned_text"] = df[text_column].apply(clean_text)
    
    # Process in batches
    texts = df["cleaned_text"].tolist()
    all_results = []
    
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(texts))
        
        batch_texts = texts[start_idx:end_idx]
        batch_results = apply_sentiment_analysis(batch_texts, sentiment_pipeline)
        all_results.extend(batch_results)
        
        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
            logger.info(f"   ✓ Processed {end_idx:,}/{len(texts):,} items")
    
    # Extract labels and scores
    df["sentiment_label_raw"] = [r["label"] for r in all_results]
    df["sentiment_confidence"] = [r["score"] for r in all_results]
    
    # Map labels to sentiment names and numeric scores
    df["sentiment_label"] = df["sentiment_label_raw"].map(SENTIMENT_MAPPING).map({
        -1: "negative",
        0: "neutral",
        1: "positive"
    })
    
    df["sentiment_score"] = df["sentiment_label_raw"].map(SENTIMENT_MAPPING)
    
    # Clean up temporary columns
    df.drop(columns=["cleaned_text", "sentiment_label_raw"], inplace=True)
    
    logger.info(f"✅ Sentiment analysis complete!")
    logger.info(f"   Positive: {(df['sentiment_score'] == 1).sum():,}")
    logger.info(f"   Neutral: {(df['sentiment_score'] == 0).sum():,}")
    logger.info(f"   Negative: {(df['sentiment_score'] == -1).sum():,}")
    
    return df


# ================================================================================
# AGGREGATION: HOURLY SENTIMENT
# ================================================================================
def aggregate_sentiment_hourly(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Aggregate sentiment by hour.
    
    Useful for time-series analysis and ML feature engineering.
    
    Args:
        df: DataFrame with sentiment columns and time column
        time_column: Name of timestamp column
        
    Returns:
        pd.DataFrame: Hourly aggregated sentiment
    """
    logger.info("📊 Aggregating sentiment by hour...")
    
    df = df.copy()
    
    # Ensure time column is datetime
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Round to hour
    df["hour"] = df[time_column].dt.floor("3H")
    
    # Aggregate
    agg_df = df.groupby("hour").agg({
        "sentiment_score": ["mean", "std", "count"],
        "sentiment_confidence": "mean",
        "post_id" if "post_id" in df.columns else "comment_id": "count"
    }).reset_index()
    
    # Flatten multi-level columns
    agg_df.columns = ["_".join(col).strip("_") for col in agg_df.columns.values]
    agg_df.rename(columns={
        "hour": "time_window",
        "sentiment_score_mean": "mean_sentiment",
        "sentiment_score_std": "std_sentiment",
        "sentiment_score_count": "num_items"
    }, inplace=True)
    
    logger.info(f"✅ Created {len(agg_df):,} hourly aggregates")
    
    return agg_df


# ================================================================================
# SAVE TO DATABASE
# ================================================================================
def save_sentiment_to_db(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    posts_agg_df: pd.DataFrame,
    comments_agg_df: pd.DataFrame
):
    """
    Save sentiment analysis results to database.
    
    Args:
        posts_df: Posts with sentiment
        comments_df: Comments with sentiment
        posts_agg_df: Hourly aggregated posts sentiment
        comments_agg_df: Hourly aggregated comments sentiment
    """
    logger.info("💾 Saving sentiment results to database...")
    
    try:
        # Save post-level sentiment
        save_df_to_db(
            df=posts_df,
            table_name=POSTS_SENTIMENT_TABLE,
            schema=SCHEMA,
            time_column="post_time",
            is_timeseries=True
        )
        logger.info(f"✅ Saved {len(posts_df):,} posts with sentiment")
        
        # Save comment-level sentiment
        save_df_to_db(
            df=comments_df,
            table_name=COMMENTS_SENTIMENT_TABLE,
            schema=SCHEMA,
            time_column="comment_time",
            is_timeseries=True
        )
        logger.info(f"✅ Saved {len(comments_df):,} comments with sentiment")
        
        # Save hourly aggregations
        save_df_to_db(
            df=posts_agg_df,
            table_name=POSTS_SENTIMENT_AGG_TABLE,
            schema=SCHEMA,
            time_column="time_window",
            is_timeseries=True
        )
        logger.info(f"✅ Saved {len(posts_agg_df):,} hourly post aggregates")
        
        save_df_to_db(
            df=comments_agg_df,
            table_name=COMMENTS_SENTIMENT_AGG_TABLE,
            schema=SCHEMA,
            time_column="time_window",
            is_timeseries=True
        )
        logger.info(f"✅ Saved {len(comments_agg_df):,} hourly comment aggregates")
        
        logger.info("✅ All sentiment data saved to database!")
        
    except Exception as e:
        logger.error(f"❌ Failed to save to database: {e}")
        raise


# ================================================================================
# STATISTICS SUMMARY
# ================================================================================
def log_sentiment_stats(posts_df: pd.DataFrame, comments_df: pd.DataFrame):
    """
    Log detailed sentiment statistics.
    
    Args:
        posts_df: Posts with sentiment
        comments_df: Comments with sentiment
    """
    logger.info("\n" + "="*80)
    logger.info("SENTIMENT ANALYSIS STATISTICS")
    logger.info("="*80)
    
    # Posts stats
    logger.info("\n📝 POSTS SENTIMENT:")
    logger.info(f"  Total posts: {len(posts_df):,}")
    logger.info(f"  Mean sentiment: {posts_df['sentiment_score'].mean():.4f}")
    logger.info(f"  Std sentiment: {posts_df['sentiment_score'].std():.4f}")
    logger.info(f"  Mean confidence: {posts_df['sentiment_confidence'].mean():.4f}")
    
    posts_dist = posts_df["sentiment_label"].value_counts()
    for label, count in posts_dist.items():
        pct = 100 * count / len(posts_df)
        logger.info(f"  {label.upper()}: {count:,} ({pct:.1f}%)")
    
    # Comments stats
    logger.info("\n💬 COMMENTS SENTIMENT:")
    logger.info(f"  Total comments: {len(comments_df):,}")
    logger.info(f"  Mean sentiment: {comments_df['sentiment_score'].mean():.4f}")
    logger.info(f"  Std sentiment: {comments_df['sentiment_score'].std():.4f}")
    logger.info(f"  Mean confidence: {comments_df['sentiment_confidence'].mean():.4f}")
    
    comments_dist = comments_df["sentiment_label"].value_counts()
    for label, count in comments_dist.items():
        pct = 100 * count / len(comments_df)
        logger.info(f"  {label.upper()}: {count:,} ({pct:.1f}%)")
    
    logger.info("\n" + "="*80 + "\n")


# ================================================================================
# MAIN PIPELINE
# ================================================================================
def main(apply_btc_filter: bool = False, save_to_database: bool = True):
    """
    Main sentiment analysis pipeline.
    
    Workflow:
    1. Load raw data from database
    2. Filter excluded subreddits
    3. Optionally filter for BTC mentions
    4. Load sentiment model
    5. Apply sentiment analysis (batched)
    6. Aggregate sentiment by hour
    7. Save results to database
    8. Log statistics
    
    Args:
        apply_btc_filter: If True, only analyze BTC-related posts/comments
        save_to_database: If True, save results to database
        
    Returns:
        dict: Results dictionary with posts, comments, and aggregations
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING REDDIT SENTIMENT ANALYSIS PIPELINE")
    logger.info("="*80 + "\n")
    
    start_time = datetime.now()
    log_device_info()
    
    # Step 1: Load data
    logger.info("\n[1/6] LOADING DATA")
    logger.info("-" * 80)
    posts_df, comments_df = load_data()
    
    # Step 2: Filter subreddits
    logger.info("\n[2/6] FILTERING SUBREDDITS")
    logger.info("-" * 80)
    posts_df, comments_df = filter_subreddits(posts_df, comments_df)
    
    # Step 3: Optional BTC filtering
    if apply_btc_filter:
        logger.info("\n[3/6] APPLYING BTC FILTER")
        logger.info("-" * 80)
        posts_df, comments_df = filter_btc(posts_df, comments_df)
    else:
        logger.info("\n[3/6] SKIPPING BTC FILTER (disabled)")
    
    # Step 4: Load sentiment model
    logger.info("\n[4/6] LOADING SENTIMENT MODEL")
    logger.info("-" * 80)
    sentiment_pipeline = load_sentiment_model()
    
    # Step 5: Apply sentiment analysis
    logger.info("\n[5/6] SENTIMENT ANALYSIS")
    logger.info("-" * 80)
    
    posts_df = add_sentiment_to_df(posts_df, "title", sentiment_pipeline)
    comments_df = add_sentiment_to_df(comments_df, "comment_text", sentiment_pipeline)
    
    # Step 6: Aggregate sentiment
    logger.info("\n[6/6] AGGREGATION & SAVING")
    logger.info("-" * 80)
    
    posts_agg_df = aggregate_sentiment_hourly(posts_df, "post_time")
    comments_agg_df = aggregate_sentiment_hourly(comments_df, "comment_time")
    
    # Step 7: Save to database
    if save_to_database:
        save_sentiment_to_db(posts_df, comments_df, posts_agg_df, comments_agg_df)
    else:
        logger.info("⏭️  Skipping database save (disabled)")
    
    # Step 8: Log statistics
    log_sentiment_stats(posts_df, comments_df)
    
    elapsed = datetime.now() - start_time
    logger.info(f"✅ PIPELINE COMPLETE in {elapsed.total_seconds():.1f}s")
    
    return {
        "posts_sentiment": posts_df,
        "comments_sentiment": comments_df,
        "posts_sentiment_hourly": posts_agg_df,
        "comments_sentiment_hourly": comments_agg_df
    }


# ================================================================================
# RUN SCRIPT
# ================================================================================
if __name__ == "__main__":
    # Run with BTC filter
    results = main(
        apply_btc_filter=True,
        save_to_database=True
    )
    
    # Display sample results
    logger.info("\n📊 SAMPLE RESULTS - POSTS:")
    logger.info("\n" + str(results["posts_sentiment"][
        ["title", "sentiment_label", "sentiment_score", "sentiment_confidence"]
    ].head(10)))
    
    logger.info("\n💬 SAMPLE RESULTS - COMMENTS:")
    logger.info("\n" + str(results["comments_sentiment"][
        ["comment_text", "sentiment_label", "sentiment_score", "sentiment_confidence"]
    ].head(10)))
    
    logger.info("\n📈 HOURLY AGGREGATION - POSTS:")
    logger.info("\n" + str(results["posts_sentiment_hourly"].head(10)))