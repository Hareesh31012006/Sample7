import os
import streamlit as st
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
import feedparser   # <-- added for RSS fallback

# -----------------------------
# SET YOUR API KEY HERE
# -----------------------------
ALPHA_VANTAGE_API_KEY = "9EJ41V9XS6Q5ZN1Y"  # Replace with your key

# -----------------------------
# Streamlit App Setup
# -----------------------------
st.set_page_config(page_title="Stock Sentiment Predictor", layout="wide")
st.title("📈 Stock Sentiment + Price Prediction Dashboard")
st.write("Predict stock trends using sentiment and a simple deep learning model.")

# -----------------------------
# Cached heavy resources
# -----------------------------
@st.cache_resource
def get_hf_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline("sentiment-analysis", device=device)
    except:
        return pipeline("sentiment-analysis", device=-1)

hf_pipeline = get_hf_pipeline()

@st.cache_resource
def get_gnews_client():
    return GNews(language="en", max_results=10)

gnews_client = get_gnews_client()

# -----------------------------
# Hybrid News Fetcher (Fix 3)
# -----------------------------
@st.cache_data(ttl=60 * 30)
def fetch_news(symbol: str):
    """
    Hybrid news fetcher:
    1) Try GNews search
    2) If GNews returns empty → fallback to Google News RSS
    """
    news = []

    # Try GNews first
    try:
        gnews_results = gnews_client.get_news_by_search(symbol)
        if gnews_results:
            news.extend(gnews_results)
    except:
        pass

    # Fallback to Google News RSS
    if len(news) == 0:
        try:
            rss_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)

            for entry in feed.entries:
                news.append({
                    "title": entry.title,
                    "description": entry.summary
                })
        except Exception as e:
            st.warning(f"RSS news fetch error: {e}")
            return []

    return news

# -----------------------------
# Fetch Stock Data (unchanged)
# -----------------------------
@st.cache_data(ttl=60 * 10)
def fetch_stock_data(symbol: str):
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize="compact")

        data = data.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }
        )
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        data[["Open", "High", "Low", "Close", "Volume"]] = data[
            ["Open", "High", "Low", "Close", "Volume"]
        ].apply(pd.to_numeric, errors="coerce")

        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Failed to fetch stock data for {symbol}: {e}")
        return pd.DataFrame()

# -----------------------------
# Sentiment Tools
# -----------------------------
@st.cache_data
def get_textblob_sentiment(text: str) -> float:
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def get_hf_label_value(label: str) -> int:
    if label.upper().startswith("POS"):
        return 1
    if label.upper().startswith("NEG"):
        return -1
    return 0

# -----------------------------
# Simple Torch Regression Model
# -----------------------------
def train_model(X: torch.Tensor, y: torch.Tensor, epochs: int = 150, lr: float = 0.01):
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()

    return model

# -----------------------------
# Analyze Stock
# -----------------------------
def analyze_stock(symbol: str):
    df = fetch_stock_data(symbol)
    if df.empty:
        raise RuntimeError("No stock data available.")

    # Fetch news
    news = fetch_news(symbol)
    sentiments = []
    for n in news:
        title = n.get("title", "")
        desc = n.get("description", "")
        text = f"{title} {desc}".strip()

        if not text:
            continue

        tb = get_textblob_sentiment(text)

        try:
            hf_res = hf_pipeline(text[:512])
            hf_label = hf_res[0]["label"] if hf_res else "NEUTRAL"
        except:
            hf_label = "NEUTRAL"

        sentiments.append((text, tb, get_hf_label_value(hf_label)))

    sent_df = (
        pd.DataFrame(sentiments, columns=["Text", "TextBlob", "HF_Sentiment"])
        if sentiments
        else pd.DataFrame(columns=["Text", "TextBlob", "HF_Sentiment"])
    )

    avg_sentiment = (
        float(pd.concat([sent_df["TextBlob"], sent_df["HF_Sentiment"]], axis=1).mean().mean())
        if not sent_df.empty
        else 0.0
    )

    df["Return"] = df["Close"].pct_change()
    df = df.dropna()

    df["Volume_Scaled"] = df["Volume"] / (df["Volume"].max() + 1e-9)

    X_np = df[["Open", "High", "Low", "Volume_Scaled"]].values.astype(np.float32)
    y_np = df["Close"].values.astype(np.float32).reshape(-1, 1)

    X = torch.tensor(X_np)
    y = torch.tensor(y_np)

    model = train_model(X, y)

    last = df.iloc[-1]
    feat = np.array([
        last["Open"], last["High"], last["Low"], last["Volume_Scaled"]
    ], dtype=np.float32).reshape(1, -1)

    with torch.no_grad():
        next_pred = float(model(torch.tensor(feat)).item())

    last_close = float(df["Close"].iloc[-1])

    suggestion = "📈 Hold"
    if next_pred > last_close and avg_sentiment > 0:
        suggestion = "📈 Buy"
    elif next_pred < last_close and avg_sentiment < 0:
        suggestion = "📉 Sell"
    elif abs(next_pred - last_close) / (last_close + 1e-9) < 0.01:
        suggestion = "🔸 Neutral (small move)"

    return df, sent_df, next_pred, suggestion, avg_sentiment

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Inputs")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT):", "AAPL")
analyze_button = st.sidebar.button("Analyze")

if analyze_button:
    try:
        with st.spinner("Fetching data and running analysis..."):
            df, sent_df, next_pred, suggestion, avg_sentiment = analyze_stock(symbol)

        st.success(f"Analysis Complete for {symbol}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Next-Day Close", f"${next_pred:.2f}")
        col2.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
        col3.metric("Avg Sentiment", f"{avg_sentiment:.3f}")

        st.markdown("### Trading Suggestion")
        st.info(suggestion)

        st.subheader("Recent Stock Prices")
        st.line_chart(df["Close"])

        st.subheader("Sentiment Summary")
        st.dataframe(sent_df)

        if not sent_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(sent_df["TextBlob"], bins=10, kde=True, ax=ax)
            ax.set_title("TextBlob Polarity Distribution")
            st.pyplot(fig)
        else:
            st.write("No news articles found.")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
else:
    st.info("Enter a symbol and click Analyze.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, PyTorch, and HuggingFace.")
