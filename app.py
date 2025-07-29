import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from textblob import TextBlob
from bs4 import BeautifulSoup
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from newsapi import NewsApiClient

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock News & Price Correlation",
    page_icon="📈", 
    layout="wide"
)

st.title("📈 Indian Stock News & Price Correlation Analyzer")
st.markdown("Enhanced sentiment analysis with relevance filtering and comprehensive timezone handling")

# Configuration constants
LOCAL_CSV = "equityList\EQUITY_L.csv"
MAX_NEWS = 100
MAX_LAG = 5
ALPHA = 0.05
MIN_RELEVANCE_SCORE = 1.0

# ────────────────────────────────────────────────────────────────────────────────
# STOCK LIST LOADER
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86_400)
def load_stock_list():
    """Load NSE stock list from local CSV file"""
    if not os.path.exists(LOCAL_CSV):
        st.error(f"❌ Required file '{LOCAL_CSV}' not found.")
        return pd.DataFrame()
    
    df = pd.read_csv(LOCAL_CSV)
    if "SYMBOL" not in df.columns:
        st.error("❌ 'SYMBOL' column missing in local CSV.")
        return pd.DataFrame()
    
    # Find company name column
    name_col = next((c for c in df.columns if "NAME" in c.upper()), None)
    if name_col is None:
        df["NAME OF COMPANY"] = df["SYMBOL"]
        name_col = "NAME OF COMPANY"
    
    df["display_name"] = df["SYMBOL"] + " - " + df[name_col].astype(str)
    return df[["SYMBOL", name_col, "display_name"]].rename(
        columns={name_col: "NAME OF COMPANY"}
    ).sort_values("SYMBOL")

# ────────────────────────────────────────────────────────────────────────────────
# NEWS QUERY BUILDING & RELEVANCE FILTERING
# ────────────────────────────────────────────────────────────────────────────────
def build_news_query(symbol: str, company_name: str) -> str:
    """Build a more precise news query for the selected stock"""
    queries = []
    
    # Always include exact symbol
    queries.append(f'"{symbol}"')
    
    # Add company name if different from symbol
    if company_name.upper() != symbol.upper():
        queries.append(f'"{company_name}"')
    
    # Add key parts of company name for better coverage
    company_parts = company_name.split()
    if len(company_parts) >= 2:
        two_word_combo = f"{company_parts[0]} {company_parts[1]}"
        queries.append(f'"{two_word_combo}"')
    
    # Add Indian stock market context to each query
    enhanced_queries = []
    for q in queries:
        enhanced_queries.append(f"({q} AND (NSE OR BSE OR India OR stock OR share))")
    
    return " OR ".join(enhanced_queries)

def calculate_relevance_score(article, symbol: str, company_name: str) -> float:
    """Calculate relevance score for an article"""
    title = article.get('title', '').lower()
    description = article.get('description', '').lower()
    combined_text = f"{title} {description}"
    
    score = 0
    
    # High relevance indicators
    if symbol.lower() in combined_text:
        score += 3
    
    # Company name variations
    company_words = company_name.lower().split()
    for word in company_words:
        if len(word) > 3 and word in combined_text:  # Avoid common short words
            score += 1
    
    # Financial context keywords
    financial_keywords = ['stock', 'share', 'equity', 'nse', 'bse', 'market', 'trading']
    for keyword in financial_keywords:
        if keyword in combined_text:
            score += 0.5
    
    return max(0, score)

def filter_relevant_articles(articles, symbol: str, company_name: str, min_score=MIN_RELEVANCE_SCORE):
    """Filter articles by relevance score"""
    scored_articles = []
    for article in articles:
        score = calculate_relevance_score(article, symbol, company_name)
        if score >= min_score:
            article['relevance_score'] = score
            scored_articles.append(article)
    
    # Sort by relevance score (highest first)
    return sorted(scored_articles, key=lambda x: x.get('relevance_score', 0), reverse=True)

# ────────────────────────────────────────────────────────────────────────────────
# NEWS FETCHING WITH RELEVANCE FILTERING
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def fetch_relevant_news(api_key, symbol, company_name, start_date):
    """Fetch relevant news articles with enhanced query and filtering"""
    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        # Build enhanced query
        query = build_news_query(symbol, company_name)
        
        # Fetch articles
        response = newsapi.get_everything(
            q=query,
            from_param=start_date,
            language='en',
            sort_by='relevancy',
            page_size=MAX_NEWS
        )
        
        raw_articles = response.get('articles', [])
        
        # Filter for relevance
        relevant_articles = filter_relevant_articles(raw_articles, symbol, company_name)
        
        return relevant_articles, query, len(raw_articles)
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return [], "", 0

# ────────────────────────────────────────────────────────────────────────────────
# STOCK PRICE FETCHING WITH TIMEZONE HANDLING
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3_600)
def fetch_prices(symbol, days):
    """Fetch Indian stock data with comprehensive timezone handling"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period=f"{days}d")
        
        # COMPREHENSIVE TIMEZONE FIX
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            # Convert to UTC first, then remove timezone info
            hist.index = hist.index.tz_convert('UTC').tz_localize(None)
        
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────────
# ENHANCED SENTIMENT ANALYSIS
# ────────────────────────────────────────────────────────────────────────────────
# Financial keyword sentiment boosters
FINANCIAL_KEYWORDS = {
    # Positive financial terms
    "beat": 0.15, "beats": 0.15, "outperform": 0.12, "outperformed": 0.12,
    "growth": 0.10, "grew": 0.10, "upgrade": 0.12, "upgraded": 0.12,
    "profit": 0.12, "profits": 0.12, "revenue": 0.08, "revenues": 0.08,
    "strong": 0.10, "robust": 0.10, "solid": 0.08, "impressive": 0.12,
    "positive": 0.08, "bullish": 0.12, "rally": 0.10, "surge": 0.12,
    "gain": 0.08, "gains": 0.08, "rise": 0.08, "rises": 0.08,
    "buy": 0.10, "recommend": 0.08, "target": 0.06,
    
    # Negative financial terms
    "miss": -0.15, "missed": -0.15, "underperform": -0.12, "underperformed": -0.12,
    "decline": -0.10, "declined": -0.10, "downgrade": -0.12, "downgraded": -0.12,
    "loss": -0.12, "losses": -0.12, "weak": -0.10, "poor": -0.10,
    "negative": -0.08, "bearish": -0.12, "fall": -0.08, "falls": -0.08,
    "drop": -0.10, "drops": -0.10, "crash": -0.15, "plunge": -0.12,
    "sell": -0.10, "avoid": -0.08, "concern": -0.08, "concerns": -0.08,
    "risk": -0.06, "risks": -0.06, "volatility": -0.04, "volatile": -0.04
}

# Source credibility weights
SOURCE_CREDIBILITY = {
    "Reuters": 1.0, "Bloomberg": 1.0, "The Economic Times": 0.9,
    "Moneycontrol": 0.9, "Yahoo Finance": 0.8, "Business Standard": 0.85,
    "Financial Express": 0.85, "Mint": 0.85, "CNBC": 0.8,
    "MarketWatch": 0.75, "Investing.com": 0.7
}

def clean_text(text: str) -> str:
    """Clean and normalize text for sentiment analysis"""
    if not text:
        return ""
    try:
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(' ')
        
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)
        
        # Keep only alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^A-Za-z0-9\s\.\,\-\'\%\$]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception:
        return re.sub(r'[^A-Za-z0-9\s]', ' ', str(text))

def keyword_boost(text: str) -> float:
    """Calculate sentiment boost from financial keywords"""
    if not text:
        return 0.0
    
    words = text.lower().split()
    boost = sum(FINANCIAL_KEYWORDS.get(word, 0) for word in words)
    
    # Normalize by text length to prevent very long articles from getting excessive boost
    return boost / max(1, len(words) / 20)

def source_weight(source_name: str) -> float:
    """Get credibility weight for news source"""
    if not source_name:
        return 0.5
    
    source_lower = source_name.lower()
    for source, weight in SOURCE_CREDIBILITY.items():
        if source.lower() in source_lower:
            return weight
    
    return 0.6  # Default weight for unknown sources

def sentiment_score(text: str) -> float:
    """Calculate enhanced sentiment score using TextBlob + financial keywords"""
    if not text:
        return 0.0
    
    try:
        # Use TextBlob for base sentiment analysis
        blob = TextBlob(text)
        base_sentiment = blob.sentiment.polarity
        
        # Add financial keyword boost
        keyword_adjustment = keyword_boost(text)
        
        # Combine base sentiment with keyword boost
        final_score = base_sentiment + keyword_adjustment
        
        # Apply confidence weighting based on text length
        confidence_weight = min(1.0, len(text.split()) / 10)
        final_score *= confidence_weight
        
        # Clamp between -1 and 1
        return max(-1.0, min(1.0, final_score))
    except Exception:
        return 0.0

def analyze_sentiment(articles):
    """Analyze sentiment of news articles with enhanced processing"""
    rows = []
    for article in articles:
        try:
            title = article.get("title", "")
            desc = article.get("description", "")
            
            if not title and not desc:
                continue
                
            # Combine and clean text
            combined_text = f"{title} {desc or ''}".strip()
            cleaned_text = clean_text(combined_text)
            
            if not cleaned_text:
                continue
            
            # Calculate sentiment score
            score = sentiment_score(cleaned_text)
            
            # Calculate source weight
            source_name = article.get("source", {}).get("name", "Unknown")
            src_weight = source_weight(source_name)
            
            # Final weight combines source credibility with sentiment confidence
            final_weight = src_weight * (abs(score) + 0.1)
            
            rows.append({
                "date": article["publishedAt"][:10],
                "datetime": article["publishedAt"],
                "sentiment": score,
                "weight": final_weight,
                "title": title,
                "source": source_name,
                "relevance": article.get('relevance_score', 0)
            })
        except Exception:
            continue  # Skip problematic articles
    
    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS WITH LAG TESTING
# ────────────────────────────────────────────────────────────────────────────────
def merge_and_corr(price_df, sent_df, lag):
    """Calculate correlation with specified lag and comprehensive timezone handling"""
    try:
        if price_df.empty or sent_df.empty:
            return np.nan, np.nan, 0
        
        # Prepare sentiment data with lag
        s_df = sent_df.copy()
        s_df["date"] = pd.to_datetime(s_df["date"]) + pd.Timedelta(days=lag)
        
        # Weighted average sentiment by date
        daily_sentiment = s_df.groupby("date").apply(
            lambda g: np.average(g["sentiment"], weights=g["weight"]) if len(g) > 0 else 0
        ).rename("avg_sentiment").reset_index()

        # Prepare price data
        price = price_df.reset_index()
        
        # COMPREHENSIVE TIMEZONE FIX
        if hasattr(price["Date"].dtype, 'tz') and price["Date"].dt.tz is not None:
            price["Date"] = price["Date"].dt.tz_convert('UTC').tz_localize(None)
        
        # Ensure both date columns are timezone-naive
        price["Date"] = pd.to_datetime(price["Date"].dt.date)
        daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"].dt.date)
        
        # Calculate price changes
        price["price_change"] = price["Close"].pct_change() * 100
        
        # Merge datasets
        merged = price.merge(daily_sentiment, left_on="Date", right_on="date", how="inner")
        merged = merged.dropna(subset=["avg_sentiment", "price_change"])
        
        if len(merged) < 3:  # Need minimum data points
            return np.nan, np.nan, 0
        
        # Calculate Pearson correlation
        r, p = pearsonr(merged["avg_sentiment"], merged["price_change"])
        return r, p, len(merged)
    
    except Exception as e:
        st.error(f"Error in correlation calculation: {str(e)}")
        return np.nan, np.nan, 0

def best_correlation(price_df, sent_df):
    """Find optimal correlation across different time lags"""
    results = []
    
    for lag in range(MAX_LAG):
        r, p, n = merge_and_corr(price_df, sent_df, lag)
        results.append({
            "lag": lag, 
            "r": r, 
            "p": p, 
            "n": n,
            "abs_r": abs(r) if not np.isnan(r) else 0
        })
    
    df = pd.DataFrame(results)
    valid_results = df.dropna(subset=["r"])
    
    if valid_results.empty:
        return None, df
    
    # Find best correlation by absolute value
    best_idx = valid_results["abs_r"].idxmax()
    best = valid_results.loc[best_idx]
    
    return best, df

# ────────────────────────────────────────────────────────────────────────────────
# USER INTERFACE
# ────────────────────────────────────────────────────────────────────────────────
stocks = load_stock_list()
sidebar = st.sidebar
sidebar.header("Configuration")

api_key = sidebar.text_input(
    "NewsAPI.org API Key", 
    type="password",
    help="Get your free API key from newsapi.org"
)

if stocks.empty:
    st.error("Cannot load stock list. Please ensure 'nse_stocks_fallback.csv' exists.")
    st.stop()

# Stock selection
selection = sidebar.selectbox("Select Stock", options=stocks["display_name"])
chosen = stocks[stocks["display_name"] == selection].iloc[0]
symbol = chosen["SYMBOL"]
company = chosen["NAME OF COMPANY"]

sidebar.info(f"**{symbol}** | {company}")

days = sidebar.slider("Days to analyze", 7, 60, 21)
sidebar.caption(f"Will analyze {days} days with up to {MAX_LAG-1} day lag testing")

# ────────────────────────────────────────────────────────────────────────────────
# QUERY TESTING FEATURE
# ────────────────────────────────────────────────────────────────────────────────
if api_key and sidebar.button("🧪 Test News Query"):
    with st.spinner("Testing news query..."):
        test_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        test_articles, test_query, raw_count = fetch_relevant_news(api_key, symbol, company, test_start)
    
    st.subheader("🔍 Query Test Results")
    st.write(f"**Built Query:** `{test_query}`")
    st.write(f"**Raw Articles Found:** {raw_count}")
    st.write(f"**Relevant Articles (score ≥ {MIN_RELEVANCE_SCORE}):** {len(test_articles)}")
    
    if test_articles:
        st.write("**Top 10 Relevant Articles:**")
        for i, article in enumerate(test_articles[:10], 1):
            relevance = article.get('relevance_score', 0)
            st.write(f"{i}. **{article['title']}** *(relevance: {relevance:.1f})*")
    else:
        st.warning("No relevant articles found. Try a different stock or check your API key.")
    
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ────────────────────────────────────────────────────────────────────────────────
if api_key and sidebar.button("🔍 Analyze Correlation", type="primary"):
    with st.spinner("Running enhanced analysis..."):
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Fetch relevant news articles
            articles, query_used, raw_total = fetch_relevant_news(api_key, symbol, company, start_date)
            
            if not articles:
                st.warning("No relevant news articles found. Try increasing the time period or selecting a different stock.")
                st.stop()
            
            # Analyze sentiment
            sent_df = analyze_sentiment(articles)
            
            if sent_df.empty:
                st.warning("No valid sentiment data extracted from articles.")
                st.stop()

            # Fetch stock price data
            price_df = fetch_prices(symbol, days + MAX_LAG)
            
            if price_df.empty:
                st.warning(f"No price data found for {symbol}. Check if the symbol is correct.")
                st.stop()

            # Find optimal correlation
            best, lag_table = best_correlation(price_df, sent_df)
            
            if best is None:
                st.warning("Insufficient overlapping data for correlation analysis.")
                st.stop()

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.stop()

    st.success("✅ Enhanced analysis completed!")

    # ── KEY METRICS DASHBOARD ──────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📰 Articles", f"{len(articles)}/{raw_total}")
        st.caption("Relevant/Total")
    
    with col2:
        avg_sent = sent_df['sentiment'].mean()
        sentiment_emoji = "😊" if avg_sent > 0.05 else "😞" if avg_sent < -0.05 else "😐"
        st.metric("🎭 Avg Sentiment", f"{avg_sent:+.3f}")
        st.caption(sentiment_emoji)
    
    with col3:
        st.metric("⏱️ Optimal Lag", f"{int(best['lag'])} days")
        st.caption("News impact delay")
    
    with col4:
        significance = "✅ Significant" if best["p"] < ALPHA else "❌ Not Significant"
        st.metric("🔗 Correlation", f"{best['r']:+.3f}")
        st.caption(f"p={best['p']:.4f} | {significance}")
    
    with col5:
        price_change = ((price_df["Close"].iloc[-1] - price_df["Close"].iloc[0]) / 
                       price_df["Close"].iloc[0] * 100)
        st.metric("📈 Price Change", f"{price_change:+.2f}%")
        st.caption(f"{days}-day period")

    # ── CORRELATION VISUALIZATION ───────────────────────────────────────────────
    st.subheader(f"📊 {symbol} Price vs Sentiment Correlation")
    
    # Get data for visualization
    optimal_lag = int(best['lag'])
    r_display, p_display, n_display = merge_and_corr(price_df, sent_df, optimal_lag)
    
    # Prepare visualization data
    vis_sent = sent_df.copy()
    vis_sent["date"] = pd.to_datetime(vis_sent["date"]) + pd.Timedelta(days=optimal_lag)
    
    daily_sentiment = vis_sent.groupby("date").apply(
        lambda g: np.average(g["sentiment"], weights=g["weight"]) if len(g) > 0 else np.nan
    ).rename("avg_sentiment").reset_index()

    # Merge with price data
    plot_price = price_df.reset_index()
    if hasattr(plot_price["Date"].dtype, 'tz') and plot_price["Date"].dt.tz is not None:
        plot_price["Date"] = plot_price["Date"].dt.tz_localize(None)
    
    plot_df = plot_price.merge(daily_sentiment, left_on="Date", right_on="date", how="left")

    # Create dual-axis chart
    fig = go.Figure()
    
    # Stock price line
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], 
        y=plot_df["Close"],
        name="Stock Price (₹)", 
        yaxis="y1", 
        line=dict(color="blue", width=2.5),
        hovertemplate="Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>"
    ))
    
    # Sentiment line (scaled)
    sentiment_data = plot_df.dropna(subset=["avg_sentiment"])
    if not sentiment_data.empty:
        sentiment_scaled = sentiment_data["avg_sentiment"] * plot_df["Close"].mean()
        fig.add_trace(go.Scatter(
            x=sentiment_data["Date"],
            y=sentiment_scaled,
            name=f"Sentiment (lag {optimal_lag}d)", 
            yaxis="y2",
            line=dict(color="red", dash="dash", width=2),
            hovertemplate="Date: %{x}<br>Sentiment: %{customdata:.3f}<extra></extra>",
            customdata=sentiment_data["avg_sentiment"]
        ))
    
    # Update layout
    corr_strength = "Strong" if abs(r_display) > 0.5 else "Moderate" if abs(r_display) > 0.3 else "Weak"
    corr_direction = "Positive" if r_display > 0 else "Negative"
    
    fig.update_layout(
        title=f"{symbol} - {corr_strength} {corr_direction} Correlation: {r_display:+.3f} (p={p_display:.4f})",
        xaxis_title="Date",
        yaxis=dict(title="Stock Price (₹)", side="left"),
        yaxis2=dict(title="Sentiment Score", overlaying="y", side="right"),
        height=550,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Correlation interpretation
    if abs(r_display) > 0.5:
        st.success(f"🎯 **Strong correlation detected!** News sentiment shows significant impact on {symbol} stock price.")
    elif abs(r_display) > 0.3:
        st.info(f"📊 **Moderate correlation found.** News sentiment has measurable influence on {symbol} price movements.")
    else:
        st.warning(f"📈 **Weak correlation observed.** News sentiment shows limited direct impact on {symbol} stock price.")

    # ── DEBUG AND DETAILED ANALYSIS ─────────────────────────────────────────────
    with st.expander("🔍 Query & News Debug Information"):
        st.write(f"**Enhanced Query Used:** `{query_used}`")
        st.write(f"**Articles Retrieved:** {raw_total} total, {len(articles)} relevant")
        st.write(f"**Relevance Threshold:** {MIN_RELEVANCE_SCORE}")
        st.write(f"**Sample Article Titles:**")
        for i, article in enumerate(articles[:5], 1):
            st.write(f"{i}. {article['title']} *(relevance: {article.get('relevance_score', 0):.1f})*")

    # ── ANALYSIS TABS ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Analysis", "⏱️ Lag Comparison", "📰 News Articles", "📋 Data Export"])
    
    with tab1:
        col_hist, col_stats = st.columns([2, 1])
        
        with col_hist:
            st.subheader("Sentiment Distribution")
            fig_hist = px.histogram(
                sent_df, 
                x="sentiment", 
                nbins=25,
                title="Distribution of News Sentiment Scores"
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_stats:
            st.subheader("Statistics")
            st.metric("Mean", f"{sent_df['sentiment'].mean():+.3f}")
            st.metric("Std Dev", f"{sent_df['sentiment'].std():.3f}")
            st.metric("Positive", f"{(sent_df['sentiment'] > 0.1).sum()}")
            st.metric("Negative", f"{(sent_df['sentiment'] < -0.1).sum()}")
            st.metric("Neutral", f"{((sent_df['sentiment'] >= -0.1) & (sent_df['sentiment'] <= 0.1)).sum()}")
    
    with tab2:
        st.subheader("Correlation by Time Lag")
        display_lag = lag_table.copy().round(4)
        display_lag["significance"] = display_lag["p"].apply(
            lambda x: "✅ Significant" if pd.notna(x) and x < ALPHA else "❌ Not Significant" if pd.notna(x) else "N/A"
        )
        
        st.dataframe(
            display_lag[["lag", "r", "p", "n", "significance"]].rename(columns={
                "lag": "Lag (days)", "r": "Correlation", "p": "P-value", 
                "n": "Sample Size", "significance": "Statistical Significance"
            }),
            use_container_width=True
        )
        
        # Lag visualization
        valid_lag = display_lag.dropna(subset=["r"])
        if not valid_lag.empty:
            fig_lag = px.bar(valid_lag, x="lag", y="r", title="Correlation by Lag")
            st.plotly_chart(fig_lag, use_container_width=True)
    
    with tab3:
        st.subheader("📰 News Articles Analysis")
        sorted_news = sent_df.reindex(sent_df['sentiment'].abs().sort_values(ascending=False).index)
        
        for _, row in sorted_news.head(15).iterrows():
            sentiment_score = row["sentiment"]
            emoji = "😊" if sentiment_score > 0.1 else "😞" if sentiment_score < -0.1 else "😐"
            color = "🟢" if sentiment_score > 0.1 else "🔴" if sentiment_score < -0.1 else "🟡"
            
            with st.expander(f"{color} {emoji} **{row['title'][:80]}...** [{sentiment_score:+.3f}]"):
                col_art, col_met = st.columns([3, 1])
                with col_art:
                    st.write(f"**Title:** {row['title']}")
                    st.write(f"**Source:** {row['source']}")
                    st.write(f"**Date:** {row['date']}")
                with col_met:
                    st.metric("Sentiment", f"{sentiment_score:+.3f}")
                    st.metric("Weight", f"{row['weight']:.3f}")
                    st.metric("Relevance", f"{row.get('relevance', 0):.1f}")
    
    with tab4:
        st.subheader("📋 Export Analysis Data")
        
        # Prepare export data
        export_data = plot_df[["Date", "Open", "High", "Low", "Close", "Volume", "avg_sentiment"]].copy()
        export_data["price_change_pct"] = export_data["Close"].pct_change() * 100
        export_data = export_data.rename(columns={
            "avg_sentiment": f"Sentiment_Lag_{optimal_lag}d",
            "price_change_pct": "Price_Change_Percent"
        })
        
        st.dataframe(export_data, use_container_width=True)
        
        # Export buttons
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            csv_data = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Price & Sentiment Data",
                data=csv_data,
                file_name=f"{symbol}_enhanced_analysis.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            # Summary statistics
            summary = {
                "Stock_Symbol": symbol,
                "Company_Name": company,
                "Analysis_Period": days,
                "Total_Articles": raw_total,
                "Relevant_Articles": len(articles),
                "Optimal_Lag": optimal_lag,
                "Correlation": r_display,
                "P_Value": p_display,
                "Significance": "Yes" if p_display < ALPHA else "No",
                "Average_Sentiment": sent_df['sentiment'].mean(),
                "Price_Change_Percent": price_change
            }
            
            summary_csv = pd.DataFrame([summary]).to_csv(index=False)
            st.download_button(
                label="📊 Download Analysis Summary",
                data=summary_csv,
                file_name=f"{symbol}_analysis_summary.csv",
                mime="text/csv"
            )

else:
    if not api_key:
        st.info("👈 Please enter your NewsAPI.org API key in the sidebar to begin analysis.")
        
        with st.expander("ℹ️ How to get a NewsAPI.org API key"):
            st.markdown("""
            1. Visit [newsapi.org](https://newsapi.org)
            2. Click "Get API Key" and sign up for a free account
            3. Copy your API key and paste it in the sidebar
            4. Free tier includes 100 requests per day
            """)

# Display application status
if not stocks.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Stocks Available:** {len(stocks)}")
    st.sidebar.markdown("**🔧 Analysis Engine:** Enhanced TextBlob + Financial Keywords")
    st.sidebar.markdown(f"**⏱️ Lag Testing:** 0-{MAX_LAG-1} days")
    st.sidebar.markdown(f"**📈 Significance Level:** {ALPHA}")
    st.sidebar.markdown(f"**🎯 Relevance Threshold:** {MIN_RELEVANCE_SCORE}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Enhanced Financial Sentiment Analysis with News Relevance Filtering")
