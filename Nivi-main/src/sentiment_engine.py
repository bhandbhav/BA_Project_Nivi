"""
Nivi Oracle Engine — Sentiment Engine
======================================
Analyzes recent news headlines using NLTK VADER.
Returns a sentiment score adjustment for the Oracle Engine.
"""

import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils import get_logger, fetch_news

log = get_logger("sentiment_engine")

# Ensure VADER lexicon is downloaded gracefully
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

analyzer = SentimentIntensityAnalyzer()

def analyse_sentiment(ticker: str, max_articles: int = 10) -> dict:
    """
    Fetches news for a ticker, runs VADER sentiment analysis,
    and translates it into an Oracle Score adjustment (-5 to +5).
    """
    result = {
        "ticker": ticker,
        "article_count": 0,
        "average_compound": 0.0,
        "sentiment_adj": 0,
        "interpretation": "Neutral"
    }

    try:
        news = fetch_news(ticker, max_articles=max_articles)
        if not news:
            return result

        total_compound = 0.0
        
        for article in news:
            title = article.get("title", "")
            if title:
                scores = analyzer.polarity_scores(title)
                total_compound += scores['compound']
        
        count = len(news)
        avg_compound = total_compound / count if count > 0 else 0.0
        
        # ── Translate to Oracle Adjustment ────────────────────────────────────
        adj = 0
        interpretation = "Neutral"
        
        if avg_compound >= 0.25:
            adj = 5
            interpretation = "Strongly Bullish"
        elif avg_compound >= 0.05:
            adj = 2
            interpretation = "Slightly Bullish"
        elif avg_compound <= -0.25:
            adj = -5
            interpretation = "Strongly Bearish"
        elif avg_compound <= -0.05:
            adj = -2
            interpretation = "Slightly Bearish"

        result["article_count"] = count
        result["average_compound"] = round(avg_compound, 3)
        result["sentiment_adj"] = adj
        result["interpretation"] = interpretation

    except Exception as e:
        log.error(f"Sentiment analysis failed for {ticker}: {e}")

    return result

# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n📰 Testing sentiment_engine.py...\n")
    
    test_tickers = ["RELIANCE.NS", "HDFCBANK.NS"]
    
    for t in test_tickers:
        res = analyse_sentiment(t)
        print(f"  {t} Sentiment: {res['interpretation']} (Adj: {res['sentiment_adj']:+d})")
        print(f"     ↳ Articles Scanned: {res['article_count']}")
        print(f"     ↳ Avg Compound Score: {res['average_compound']}\n")
    
    print("✅ sentiment_engine.py test complete.\n")