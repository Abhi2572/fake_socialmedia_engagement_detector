import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

# 🔐 Secure API Key
API_KEY = st.secrets["API_KEY"]

youtube = build("youtube", "v3", developerKey=API_KEY)

import os
import gdown
import joblib
import streamlit as st

# -----------------------------
# MODEL DOWNLOAD
# -----------------------------
MODEL_PATH = "fake_engagement_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model... please wait ⏳")
    
    url = "https://drive.google.com/uc?export=download&id=1xMg8Og9yghzXJI_QJhvzLut8eXOyep0N"
    
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
    except Exception as e:
        st.error("❌ Model download failed. File may be too large or not public.")
        st.stop()

if os.path.exists(MODEL_PATH):
    st.success("✅ Model loaded successfully!")
    model = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model not found after download")
    st.stop()

# -----------------------------
# LOAD MODEL AFTER DOWNLOAD
# -----------------------------
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Fake Engagement Detection", layout="wide")

st.title("🔍 Fake Engagement Detection")
st.write("Analyze YouTube videos for suspicious engagement patterns")

def extract_video_id(url):
    parsed = urlparse(url)

    if parsed.hostname in ("youtu.be",):
        return parsed.path[1:]

    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]

        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]

    return None

def get_video_stats(video_id):

    request = youtube.videos().list(
        part="statistics,snippet",
        id=video_id
    )

    response = request.execute()

    if not response["items"]:
        return None

    data = response["items"][0]

    stats = data["statistics"]
    snippet = data["snippet"]

    views = int(stats.get("viewCount", 0))
    likes = int(stats.get("likeCount", 0))
    comments = int(stats.get("commentCount", 0))

    title = snippet["title"]
    thumbnail = snippet["thumbnails"]["high"]["url"]

    channel_id = snippet["channelId"]

    # Get subscriber count
    channel_request = youtube.channels().list(
        part="statistics",
        id=channel_id
    )

    channel_response = channel_request.execute()

    subscribers = int(
        channel_response["items"][0]["statistics"].get("subscriberCount", 0)
    )

    return views, likes, comments, subscribers, title, thumbnail

# -----------------------------
# UI Input
# -----------------------------
url = st.text_input("Enter YouTube Video URL")

if st.button("Analyze Video"):

    video_id = extract_video_id(url)

    if video_id is None:
        st.error("Invalid YouTube URL")
    else:

        result = get_video_stats(video_id)

        if result is None:
            st.error("Could not fetch video data")
        else:

            views, likes, comments, subscribers, title, thumbnail = result

            # -----------------------------
            # Feature Engineering
            # -----------------------------
            like_view_ratio = likes / (views + 1)
            comment_view_ratio = comments / (views + 1)
            like_comment_ratio = likes / (comments + 1)
            subscriber_view_ratio = views / (subscribers + 1)

            engagement_rate = (likes + comments) / (views + 1)

            # -----------------------------
            # ML Prediction
            # -----------------------------
            features = pd.DataFrame([{
                "like_view_ratio": like_view_ratio,
                "comment_view_ratio": comment_view_ratio,
                "like_comment_ratio": like_comment_ratio
            }])

            probability = model.predict_proba(features)[0][1]
            authenticity_score = 1 - probability

            # -----------------------------
            # Suspicion Score
            # -----------------------------
            suspicion_score = 0

            if subscriber_view_ratio > 5:
                suspicion_score += 1

            if comment_view_ratio < 0.0001:
                suspicion_score += 1

            if engagement_rate < 0.01:
                suspicion_score += 1

            final_score = authenticity_score - (0.1 * suspicion_score)
            final_score = max(0, min(final_score, 1))

            # -----------------------------
            # Display
            # -----------------------------
            st.subheader(title)
            st.image(thumbnail)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Views", f"{views:,}")
            col2.metric("Likes", f"{likes:,}")
            col3.metric("Comments", f"{comments:,}")
            col4.metric("Subscribers", f"{subscribers:,}")

            st.write(" ")

            st.write("### Scores")
            st.write("Authenticity Score:", round(authenticity_score, 2))
            st.write("Suspicion Score:", suspicion_score)
            st.write("Final Score:", round(final_score, 2))

            # -----------------------------
            # Clean Graph (Log Scale)
            # -----------------------------
            ratio_df = pd.DataFrame({
                "Metric": [
                    "Like/View",
                    "Comment/View",
                    "Like/Comment",
                    "Subscriber/View"
                ],
                "Value": [
                    like_view_ratio,
                    comment_view_ratio,
                    like_comment_ratio,
                    subscriber_view_ratio
                ]
            })

            # Log scaling
            ratio_df["Scaled"] = np.log10(ratio_df["Value"] + 1e-6)

            fig = px.bar(
                ratio_df,
                x="Metric",
                y="Scaled",
                color="Metric",
                text=ratio_df["Value"].round(4),
                title="Engagement Metrics (Log Scaled)"
            )

            fig.update_layout(
                template="plotly_dark",
                title_x=0.5,
                showlegend=False
            )

            st.plotly_chart(fig)

            # -----------------------------
            # Result Message
            # -----------------------------
            if final_score < 0.4:
                st.error("⚠ Suspicious Engagement Detected")
            elif final_score < 0.7:
                st.warning("⚠ Engagement looks unusual")
            else:
                st.success("✔ Engagement appears genuine")
