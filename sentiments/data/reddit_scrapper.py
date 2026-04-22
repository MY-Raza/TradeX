import praw
import pandas as pd
from datetime import datetime
from TradeX.utils.common.logs import get_logger
from TradeX.utils.db.utils import save_df_to_db, read_df_from_db
import time
import prawcore

logger = get_logger("reddit_scraper")

posts = read_df_from_db(
    table_name="reddit_posts",
    schema="reddit",
    limit=1000
)

comments = read_df_from_db(
    table_name="reddit_comments",
    schema="reddit",
    limit=1000
)

print(posts.head())
print(comments.head())

# =========================
# Reddit API Initialization
# =========================
reddit = praw.Reddit(
    client_id="c4tOcIwaGed2RYnsuEEFUQ",
    client_secret="-yG8JfMUxhn9Th8fctAShn6pi_co0A",
    user_agent="Scraping",
    ratelimit_seconds=300,   # wait up to 5 min if rate limited (PRAW built-in)
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
# Config  (tune these down if 429s persist)
# =========================
POSTS_PER_SUB     = 50    # was 200 — reduce request volume
COMMENTS_PER_POST = 5     # was 10
SLEEP_BETWEEN_SUBS = 5    # seconds between subreddits
SLEEP_BETWEEN_POSTS = 0.3 # seconds between comment fetches

# =========================
# Retry helper
# =========================

def _fetch_with_backoff(fn, max_retries: int = 5):
    """
    Call fn(). On 429 / RateLimitExceeded, wait and retry
    with exponential backoff. Raises on other exceptions.
    """
    wait = 30  # start with 30 s
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except prawcore.exceptions.TooManyRequests:
            logger.warning(f"429 – attempt {attempt}/{max_retries}, waiting {wait}s…")
            time.sleep(wait)
            wait = min(wait * 2, 300)  # cap at 5 min
        except praw.exceptions.RedditAPIException as exc:
            # PRAW wraps rate-limit as RedditAPIException too
            if any(e.error_type == "RATELIMIT" for e in exc.items):
                logger.warning(f"RATELIMIT – attempt {attempt}/{max_retries}, waiting {wait}s…")
                time.sleep(wait)
                wait = min(wait * 2, 300)
            else:
                raise
    raise RuntimeError(f"Exceeded {max_retries} retries due to rate limiting.")


# =========================
# Storage
# =========================
posts_data    = []
comments_data = []

# =========================
# Data Collection
# =========================
for sub in subreddits:
    logger.info(f"⏳ Fetching r/{sub}…")
    try:
        subreddit = reddit.subreddit(sub)
        hot_posts = _fetch_with_backoff(lambda: list(subreddit.hot(limit=POSTS_PER_SUB)))
    except Exception as exc:
        logger.error(f"Skipping r/{sub} — {exc}")
        continue

    for post in hot_posts:
        post_id   = post.id
        post_time = datetime.fromtimestamp(post.created_utc)

        score        = post.score
        upvote_ratio = post.upvote_ratio
        try:
            upvotes   = int(score / upvote_ratio) if upvote_ratio > 0 else score
            downvotes = upvotes - score
        except Exception:
            upvotes, downvotes = score, 0

        posts_data.append({
            "post_id":            post_id,
            "subreddit":          sub,
            "title":              post.title,
            "score":              score,
            "upvote_ratio":       upvote_ratio,
            "estimated_upvotes":  upvotes,
            "estimated_downvotes":downvotes,
            "num_comments":       post.num_comments,
            "post_time":          post_time,
            "author":             str(post.author),
            "url":                post.url,
        })

        # Fetch comments with backoff
        try:
            _fetch_with_backoff(lambda: post.comments.replace_more(limit=0))
        except Exception as exc:
            logger.warning(f"Skipping comments for post {post_id} — {exc}")
            continue

        for comment in post.comments[:COMMENTS_PER_POST]:
            comment_time = datetime.fromtimestamp(comment.created_utc)
            comments_data.append({
                "post_id":        post_id,
                "comment_id":     comment.id,
                "subreddit":      sub,
                "comment_text":   comment.body,
                "comment_score":  comment.score,
                "comment_time":   comment_time,
                "comment_author": str(comment.author),
            })

        time.sleep(SLEEP_BETWEEN_POSTS)

    logger.info(f"✅ r/{sub} done — {len(hot_posts)} posts")
    time.sleep(SLEEP_BETWEEN_SUBS)

# =========================
# Convert to DataFrames
# =========================
posts_df    = pd.DataFrame(posts_data)
comments_df = pd.DataFrame(comments_data)

# =========================
# Save to Database
# =========================
save_df_to_db(
    df=posts_df,
    table_name="reddit_posts",
    schema="reddit",
    time_column="post_time",
    is_timeseries=True,
)

save_df_to_db(
    df=comments_df,
    table_name="reddit_comments",
    schema="reddit",
    time_column="comment_time",
    is_timeseries=True,
)

logger.info("✅ Data saved to database!")