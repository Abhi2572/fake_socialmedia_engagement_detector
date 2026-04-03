from pyexpat import features

from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

import joblib

model = joblib.load("fake_engagement_model.pkl") 

API_KEY = "AIzaSyCqgSNv010geWmGePkBe2Ck44c_ZGL-Wak"

youtube = build(
    'youtube',
    'v3',
    developerKey=API_KEY
)

from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed_url = urlparse(url)
    video_id = None
    
    if parsed_url.hostname in ("youtu.be"):
        video_id = parsed_url.path[1:]
    elif parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query)["v"][0]
        elif parsed_url.path.startswith("/shorts/"):
            video_id = parsed_url.path.split("/")[2]
    
    if video_id is None:
        print("Invalid YouTube URL")
        exit()

    if parsed_url.hostname in ("youtu.be"):
        return parsed_url.path[1:]

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):

        # Normal video URL
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]

        # Shorts URL
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]

    return None
url = input("Enter YouTube video URL: ")

video_id = extract_video_id(url)

print("Video ID:", video_id)

def get_video_stats(video_id):

    request = youtube.videos().list(
        part="statistics",
        id=video_id
    )

    response = request.execute()

    stats = response["items"][0]["statistics"]

    views = stats.get("viewCount", 0)
    likes = stats.get("likeCount", 0)
    comments = stats.get("commentCount", 0)

    return views, likes, comments
def get_video_comments(video_id, max_comments=50):

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    )

    response = request.execute()

    comments = []

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments


views, likes, comments = get_video_stats(video_id)

comments_list = get_video_comments(video_id)

print("\nSample Comments:")
for c in comments_list[:5]:
    print("-", c)
    
print("Views:", views)
print("Likes:", likes)
print("Comments:", comments)

views = int(views)
likes = int(likes)
comments = int(comments)

like_view_ratio = likes / views if views != 0 else 0
comment_view_ratio = comments / views if views != 0 else 0
like_comment_ratio = likes / (comments + 1)

print("\nCalculated Features")
print("Like/View Ratio:", like_view_ratio)
print("Comment/View Ratio:", comment_view_ratio)
print("Like/Comment Ratio:", like_comment_ratio)

import pandas as pd

features = pd.DataFrame([{
    "like_view_ratio": like_view_ratio,
    "comment_view_ratio": comment_view_ratio,
    "like_comment_ratio": like_comment_ratio
}])

prediction = model.predict(features)

from collections import Counter

def repeated_comment_ratio(comments):

    if len(comments) == 0:
        return 0

    counts = Counter(comments)

    repeated = sum(1 for c in counts.values() if c > 1)

    return repeated / len(comments) 

spam_ratio = repeated_comment_ratio(comments_list)

print("\nComment Spam Ratio:", spam_ratio)

probability = model.predict_proba(features)[0][1]
authenticity_score = (1 - probability) * (1 - spam_ratio)
print("\nAuthenticity Score:", round(authenticity_score, 2))

if authenticity_score < 0.4:
    print("Result: Suspicious Engagement Detected")
elif authenticity_score < 0.7:
    print("Result: Engagement looks slightly unusual")
else:
    print("Result: Engagement appears Genuine")