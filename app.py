# your full code starting here
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from textblob import TextBlob
from transformers import pipeline
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import feedparser  # RSS fallback
import urllib.parse   # <-- ADDED FOR URL FIX

# -----------------------------
# SET YOUR API KEY (ENV-SAFE)
# -----------------------------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "9EJ41V9XS6Q5ZN1Y")

# -----------------------------
# Streamlit App Setup
# -----------------------------
st.set_page_config(page_title="Stock Sentiment Predictor", layout="wide")
st.title("📈 Stock Sentiment + Price Prediction Dashboard")
st.write("Predict stock trends using sentiment and a simple deep learning model.")

# -----------------------------
# Caching
# -----------------------------
@st.cache_resource
def get_hf_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline("sentiment-analysis", device=device)
    except Exception:
        return pipeline("sentiment-analysis", device=-1)

hf_pipeline = get_hf_pipeline()

@st.cache_resource
def get_gnews_client():
    try:
        return GNews(language="en", max_results=10)
    except Exception:
        return None

gnews_client = get_gnews_client()


# -----------------------------
# FIXED NEWS FETCH FUNCTION
# -----------------------------
@st.cache_data(ttl=60 * 30)
def fetch_news(symbol: str):
    news = []

    # Try GNews first
    try:
        if gnews_client:
            try:
                if hasattr(gnews_client, "get_news_by_search"):
                    gnews_results = gnews_client.get_news_by_search(symbol)
                else:
                    gnews_results = gnews_client.get_news(symbol)

                if gnews_results:
                    for item in gnews_results:
                        title = item.get("title", "")
                        desc = item.get("description", "")
                        news.append({"title": title, "description": desc})
            except:
                pass
    except:
        pass

    # Fallback to RSS
    if len(news) == 0:
        try:
            # FIX: ENCODE QUERY TO REMOVE SPACES & CONTROL CHARS
            encoded_query = urllib.parse.quote(f"{symbol} stock")
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

            feed = feedparser.parse(rss_url)

            for entry in feed.entries:
                title = getattr(entry, "title", "")
                desc = getattr(entry, "summary", "")
                news.append({"title": title, "description": desc})

        except Exception as e:
            st.warning(f"RSS news fetch error: {e}")
            return []

    return news


# -----------------------------
# Fetch Stock Data
# -----------------------------
@st.cache_data(ttl=600)
def fetch_stock_data(symbol: str):
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize="compact")

        data = data.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        })

        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Failed to fetch stock data: {e}")
        return pd.DataFrame()


# -----------------------------
# Sentiment helpers
# -----------------------------
@st.cache_data
def get_textblob_sentiment(text: str) -> float:
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def get_hf_label_value(label: str) -> int:
    if label.upper().startswith("POS"): return 1
    if label.upper().startswith("NEG"): return -1
    return 0


# -----------------------------
# Simple PyTorch Model
# -----------------------------
def train_model(X, y, epochs=150, lr=0.01):
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        opt.step()

    return model


# -----------------------------
# ANALYSIS FUNCTION
# -----------------------------
def analyze_stock(symbol: str):
    df = fetch_stock_data(symbol)
    if df.empty:
        raise RuntimeError("No stock data available.")

    news = fetch_news(symbol)
    sentiments = []

    for n in news:
        text = (n["title"] + " " + n["description"]).strip()
        if not text:
            continue

        tb = get_textblob_sentiment(text)

        if use_light_mode:
            if tb > 0.1: val = 1
            elif tb < -0.1: val = -1
            else: val = 0
        else:
            try:
                res = hf_pipeline(text[:512])
                label = res[0]["label"]
            except:
                label = "NEUTRAL"

            val = get_hf_label_value(label)

        sentiments.append((text, tb, val))

    if sentiments:
        sent_df = pd.DataFrame(sentiments, columns=["Text", "TextBlob", "HF_Sentiment"])
        avg_sentiment = (sent_df["TextBlob"].mean() + sent_df["HF_Sentiment"].mean()) / 2
    else:
        sent_df = pd.DataFrame(columns=["Text", "TextBlob", "HF_Sentiment"])
        avg_sentiment = 0.0

    df["Volume_Scaled"] = df["Volume"] / df["Volume"].max()
    X_np = df[["Open", "High", "Low", "Volume_Scaled"]].values.astype(np.float32)
    y_np = df["Close"].values.astype(np.float32).reshape(-1, 1)

    X = torch.tensor(X_np)
    y = torch.tensor(y_np)

    model = train_model(X, y)

    last = df.iloc[-1]
    last_feat = np.array([
        last["Open"], last["High"], last["Low"], last["Volume_Scaled"]
    ], dtype=np.float32)

    pred = float(model(torch.tensor(last_feat.reshape(1, -1))).item())
    last_close = float(df["Close"].iloc[-1])

    suggestion = "📈 Hold"
    if pred > last_close and avg_sentiment > 0: suggestion = "📈 Buy"
    if pred < last_close and avg_sentiment < 0: suggestion = "📉 Sell"

    return df, sent_df, pred, suggestion, avg_sentiment


# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Inputs")
symbol = st.sidebar.text_input("Enter Stock Symbol:", "AAPL")
use_light_mode = st.sidebar.checkbox("Use Lightweight Sentiment", value=False)

if st.sidebar.button("Analyze"):
    try:
        with st.spinner("Analyzing..."):
            df, sent_df, pred, suggestion, avg = analyze_stock(symbol)

        st.success("Analysis Complete")
        c1, c2, c3 = st.columns(3)

        c1.metric("Next-Day Prediction", f"${pred:.2f}")
        c2.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
        c3.metric("Avg Sentiment", f"{avg:.3f}")

        st.info(suggestion)

        st.line_chart(df["Close"])

        st.subheader("Sentiment Summary")
        st.dataframe(sent_df)

        if not sent_df.empty:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(sent_df["TextBlob"], bins=10, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write("No news articles found.")

    except Exception as e:
        st.error(str(e))
else:
    st.info("Enter a symbol & click Analyze.")


st.markdown("---")
st.caption("Built with ❤️ using Streamlit + PyTorch + HuggingFace")
