import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="c4tOcIwaGed2RYnsuEEFUQ",
    client_secret="-yG8JfMUxhn9Th8fctAShn6pi_co0A",
    user_agent="Scraping"
)

subreddits = ["cryptocurrency", "bitcoin", "ethtrader", "CryptoMarkets"]

data = []

for sub in subreddits:
    print(f"⏳ Fetching r/{sub}...")
    subreddit = reddit.subreddit(sub)

    for post in subreddit.hot(limit=100):
        upvote_ratio = post.upvote_ratio
        score = post.score

        try:
            upvotes = int(score / upvote_ratio) if upvote_ratio > 0 else score
            downvotes = upvotes - score
        except:
            upvotes, downvotes = score, 0

        # Fetch comments without triggering extra API calls
        try:
            post.comments.replace_more(limit=0)
            comments = [c.body for c in post.comments[:10] if hasattr(c, 'body')]
        except Exception:
            comments = []

        data.append({
            "subreddit": sub,
            "title": post.title,
            "score": score,
            "upvote_ratio": upvote_ratio,
            "estimated_upvotes": upvotes,
            "estimated_downvotes": downvotes,
            "num_comments": post.num_comments,
            "comments": comments,
            "author": str(post.author),
            "created_utc": post.created_utc,
            "url": post.url
        })

    print(f"✅ r/{sub} done")

df = pd.DataFrame(data)
df.to_csv("crypto_reddit_data.csv", index=False)
print(f"\n✅ Saved {len(df)} posts to crypto_reddit_data.csv")