import praw
import pandas as pd

# Initialize Reddit
reddit = praw.Reddit(
    client_id="c4tOcIwaGed2RYnsuEEFUQ",
    client_secret="-yG8JfMUxhn9Th8fctAShn6pi_co0A",
    user_agent="Scraping"
)


# Target subreddits
subreddits = ["cryptocurrency", "bitcoin", "ethtrader", "CryptoMarkets"]

data = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    
    for post in subreddit.hot(limit=100):  # change limit as needed
        
        # Estimate votes
        upvote_ratio = post.upvote_ratio
        score = post.score
        
        try:
            upvotes = int(score / upvote_ratio) if upvote_ratio > 0 else score
            downvotes = upvotes - score
        except:
            upvotes, downvotes = score, 0

        # Fetch comments (top 10)
        post.comments.replace_more(limit=0)
        comments = [comment.body for comment in post.comments[:10]]

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

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("crypto_reddit_data.csv", index=False)

print("✅ Data saved successfully!")