import pandas as pd
from TradeX.utils.db.utils import read_df_from_db
from TradeX.utils.common.logs import get_logger

logger = get_logger("filtered_reddit_pipeline")

# =====================================
# CONFIG
# =====================================
SCHEMA = "reddit"
POSTS_TABLE = "reddit_posts"
COMMENTS_TABLE = "reddit_comments"

# Subreddits to EXCLUDE
EXCLUDED_SUBS = ["ethereum", "ethtrader"]

# Optional BTC filter pattern
BTC_PATTERN = r"\b(btc|bitcoin)\b|\$btc"


# =====================================
# LOAD DATA FROM DB
# =====================================
def load_data():
    logger.info("📥 Loading data from DB...")

    posts_df = read_df_from_db(
        table_name=POSTS_TABLE,
        schema=SCHEMA
    )

    comments_df = read_df_from_db(
        table_name=COMMENTS_TABLE,
        schema=SCHEMA
    )

    logger.info(f"✅ Posts loaded: {len(posts_df)}")
    logger.info(f"✅ Comments loaded: {len(comments_df)}")

    return posts_df, comments_df


# =====================================
# FILTER SUBREDDITS
# =====================================
def filter_subreddits(posts_df: pd.DataFrame, comments_df: pd.DataFrame):
    logger.info("🚫 Removing ethereum & ethtrader data...")

    posts_filtered = posts_df[
        ~posts_df["subreddit"].str.lower().isin(EXCLUDED_SUBS)
    ].copy()

    comments_filtered = comments_df[
        ~comments_df["subreddit"].str.lower().isin(EXCLUDED_SUBS)
    ].copy()

    posts_filtered.reset_index(drop=True, inplace=True)
    comments_filtered.reset_index(drop=True, inplace=True)

    logger.info(f"✅ Posts after filter: {len(posts_filtered)}")
    logger.info(f"✅ Comments after filter: {len(comments_filtered)}")

    return posts_filtered, comments_filtered


# =====================================
# OPTIONAL BTC FILTER
# =====================================
def filter_btc(posts_df: pd.DataFrame, comments_df: pd.DataFrame):
    logger.info("₿ Filtering BTC-related data...")

    btc_posts = posts_df[
        posts_df["title"].str.contains(BTC_PATTERN, case=False, regex=True, na=False) |
        posts_df["url"].str.contains(BTC_PATTERN, case=False, regex=True, na=False)
    ].copy()

    btc_comments = comments_df[
        comments_df["comment_text"].str.contains(BTC_PATTERN, case=False, regex=True, na=False)
    ].copy()

    btc_posts.reset_index(drop=True, inplace=True)
    btc_comments.reset_index(drop=True, inplace=True)

    logger.info(f"✅ BTC Posts: {len(btc_posts)}")
    logger.info(f"✅ BTC Comments: {len(btc_comments)}")

    return btc_posts, btc_comments


# =====================================
# SAVE TO DB (OPTIONAL)
# =====================================
def save_to_db(posts_df: pd.DataFrame, comments_df: pd.DataFrame):
    logger.info("💾 Saving filtered data to DB...")

    save_df_to_db(
        df=posts_df,
        table_name="filtered_posts",
        schema=SCHEMA,
        time_column="post_time"
    )

    save_df_to_db(
        df=comments_df,
        table_name="filtered_comments",
        schema=SCHEMA,
        time_column="comment_time"
    )

    logger.info("✅ Data saved to DB")


# =====================================
# MAIN PIPELINE
# =====================================
def main(apply_btc_filter=False, save=False):
    posts_df, comments_df = load_data()

    # Step 1: Remove unwanted subreddits
    posts_df, comments_df = filter_subreddits(posts_df, comments_df)

    # Step 2: Optional BTC filtering
    if apply_btc_filter:
        posts_df, comments_df = filter_btc(posts_df, comments_df)

    return posts_df, comments_df


# =====================================
# RUN SCRIPT
# =====================================
if __name__ == "__main__":
    posts, comments = main(
        apply_btc_filter=True,   
    )

    logger.info("\n📊 Sample Posts:")
    logger.info(posts.head())

    logger.info("\n💬 Sample Comments:")
    logger.info(comments.head())