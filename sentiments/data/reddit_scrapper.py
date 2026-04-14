import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT"
)

# Try a simple request
for post in reddit.subreddit("test").hot(limit=5):
    print(post.title)