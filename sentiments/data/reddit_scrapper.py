import praw
import pandas as pd
from datetime import datetime
from TradeX.utils.common.logs import get_logger
logger = get_logger("reddit_scraper")

# =========================
# Reddit API Initialization
# =========================
reddit = praw.Reddit(
    client_id="c4tOcIwaGed2RYnsuEEFUQ",
    client_secret="-yG8JfMUxhn9Th8fctAShn6pi_co0A",
    user_agent="Scraping"
)

# =========================
# Subreddits
# =========================
subreddits = [
    "cryptocurrency",
    "CryptoMarkets",
    "Bitcoin",
    "BitcoinMarkets",
    "ethereum",
    "ethtrader",
    "CryptoMoonShots",
    "SatoshiStreetBets",
    "CryptoCurrencyTrading",
    "CryptoNews"
]

# =========================
# Storage
# =========================
posts_data = []
comments_data = []

# =========================
# Data Collection
# =========================
for sub in subreddits:
    logger.info(f"⏳ Fetching r/{sub}...")
    subreddit = reddit.subreddit(sub)

    for post in subreddit.hot(limit=200):

        post_id = post.id
        post_time = datetime.fromtimestamp(post.created_utc)

        # ---------------------
        # Post metrics
        # ---------------------
        score = post.score
        upvote_ratio = post.upvote_ratio

        try:
            upvotes = int(score / upvote_ratio) if upvote_ratio > 0 else score
            downvotes = upvotes - score
        except:
            upvotes, downvotes = score, 0

        # ---------------------
        # Store post
        # ---------------------
        posts_data.append({
            "post_id": post_id,
            "subreddit": sub,
            "title": post.title,
            "score": score,
            "upvote_ratio": upvote_ratio,
            "estimated_upvotes": upvotes,
            "estimated_downvotes": downvotes,
            "num_comments": post.num_comments,
            "post_time": post_time,
            "author": str(post.author),
            "url": post.url
        })

        # ---------------------
        # Comments
        # ---------------------
        post.comments.replace_more(limit=0)

        for comment in post.comments[:10]:

            comment_time = datetime.fromtimestamp(comment.created_utc)

            comments_data.append({
                "post_id": post_id,
                "comment_id": comment.id,
                "subreddit": sub,
                "comment_text": comment.body,
                "comment_score": comment.score,   # ONLY available metric
                "comment_time": comment_time,
                "comment_author": str(comment.author)
            })
    logger.info(f"✅ r/{sub} done")        

# =========================
# Convert to DataFrames
# =========================
posts_df = pd.DataFrame(posts_data)
comments_df = pd.DataFrame(comments_data)

# =========================
# Save CSVs
# =========================
posts_df.to_csv("crypto_posts.csv", index=False)
comments_df.to_csv("crypto_comments.csv", index=False)

logger.info("✅ Data collection completed!")
logger.info("📁 Saved: crypto_posts.csv")
logger.info("📁 Saved: crypto_comments.csv")