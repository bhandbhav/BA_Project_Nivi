"""
Nivi Oracle Engine — Technical Engine
======================================
Computes:
  1. Trend: SMA-200, SMA-50
  2. Momentum: RSI (14), MACD (12, 26, 9), 1M Return
  3. Volatility: Bollinger Bands Width
  4. Relative Strength: 6M Alpha vs Nifty 50
  5. Conviction: Volume Ratio, Delivery % (Proxy)

Returns a dictionary of raw technical indicators and a synthesized Technical Score.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
import json
from pathlib import Path
from utils import get_logger, fetch_prices_single

log = get_logger("technical_engine")

# ── PASTE THIS BLOCK after your imports, before _calculate_rsi ────────────────

def _scalar(val):
    """
    Safely extracts a Python float from a pandas scalar, 0-d array,
    or a 1-element Series. Eliminates FutureWarning from float(Series).
    """
    import pandas as pd, numpy as np
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    if isinstance(val, np.generic):
        return float(val)
    return float(val)

# ── Config ────────────────────────────────────────────────────────────────────
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
VOL_AVG_PERIOD = 20

def _calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    return _scalar(rsi.iloc[-1]) if not rsi.empty else 50.0


def _calculate_macd(prices):
    ema_fast   = prices.ewm(span=12, adjust=False).mean()
    ema_slow   = prices.ewm(span=26, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram  = macd_line - signal_line
    hist_val   = _scalar(histogram.iloc[-1])
    return {
        "macd":      _scalar(macd_line.iloc[-1]),
        "signal":    _scalar(signal_line.iloc[-1]),
        "hist":      hist_val,
        "is_bullish": bool(hist_val > 0),
    }


def _calculate_bollinger_width(prices):
    sma   = prices.rolling(20).mean()
    std   = prices.rolling(20).std()
    upper = sma + (2.0 * std)
    lower = sma - (2.0 * std)
    width = (upper - lower) / sma
    return _scalar(width.iloc[-1]) if not width.empty else 0.0

def _get_delivery_proxy(ticker: str) -> float:
    """
    Placeholder for NSE Delivery %. 
    Yahoo doesn't provide this natively. In a live production environment, 
    this would hit the NSE API. For the demo, we map a static proxy or 
    default to 55% (healthy delivery).
    """
    # High conviction proxies for demo purposes
    high_conviction = ["RELIANCE.NS", "HDFCBANK.NS", "ITC.NS", "LT.NS"]
    if ticker in high_conviction:
        return 62.5
    return 55.0

def analyse_technical(ticker: str, benchmark_ticker: str = "^NSEI") -> dict:
    """
    Master function — runs all technical and volume checks.
    """
    result = {
        "ticker": ticker,
        "indicators": {},
        "tech_score": 0,
        "error": None
    }

    try:
        # 1. Fetch OHLCV Data for the last 1 year
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty or len(df) < 50:
            raise ValueError("Insufficient price data for technicals")

        close = df['Close']
        volume = df['Volume']

        # Fetch Benchmark for Alpha
        bench_close = fetch_prices_single(benchmark_ticker, period="1y")

        # ── 2. Calculate Indicators ───────────────────────────────────────────
        current_price = _scalar(close.iloc[-1])
        sma_200 = _scalar(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else current_price
        sma_50 = _scalar(close.rolling(50).mean().iloc[-1])
        
        rsi = _calculate_rsi(close)
        macd_data = _calculate_macd(close)
        bb_width = _calculate_bollinger_width(close)
        
        # Volume Ratio (Current Vol / 20-Day Avg)
        avg_vol   = _scalar(volume.rolling(VOL_AVG_PERIOD).mean().iloc[-1])
        vol_ratio = (_scalar(volume.iloc[-1]) / avg_vol) if avg_vol > 0 else 1.0
        
        delivery_pct = _get_delivery_proxy(ticker)

        # 1M Momentum (21 trading days)
        mom_1m = ((current_price / close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
        
        # 6M Alpha vs Benchmark (126 trading days)
        if len(close) >= 126 and len(bench_close) >= 126:
            stock_6m = (current_price / close.iloc[-126]) - 1
            bench_6m = (bench_close.iloc[-1] / bench_close.iloc[-126]) - 1
            alpha_6m = (stock_6m - bench_6m) * 100
        else:
            alpha_6m = 0.0

        # ── 3. SHAP-Weighted Scoring Logic (0 to 100) ─────────────────────────────
        # Load dynamic weights from backtester
        WEIGHTS_FILE = Path("data/static/dynamic_weights.json")
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                shap_weights = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log.warning("SHAP weights not found. Defaulting to baseline importance.")
            shap_weights = {"bb_width": 28.0, "trend_200": 21.7, "macd_hist": 20.1, "rsi": 13.1, "trend_50": 8.6, "mom_1m": 8.6}

        score = 0.0
        
        # 1. BB Width (Weight: ~28%) - Reward volatility squeezes
        if bb_width < 0.20: 
            score += shap_weights.get("bb_width", 28.0)
        elif bb_width < 0.35: 
            score += shap_weights.get("bb_width", 28.0) * 0.5
        
        # 2. Trend 200 (Weight: ~21.7%) - Long-term macro trend
        if current_price > sma_200: 
            score += shap_weights.get("trend_200", 21.7)
        
        # 3. MACD Hist (Weight: ~20.1%) - Bullish momentum
        if macd_data["is_bullish"]: 
            score += shap_weights.get("macd_hist", 20.1)
        
        # 4. RSI (Weight: ~13.1%) - Healthy momentum vs Overbought
        if 40 <= rsi <= 70: 
            score += shap_weights.get("rsi", 13.1)
        elif rsi > 70: 
            score += shap_weights.get("rsi", 13.1) * 0.3
        
        # 5. Trend 50 (Weight: ~8.6%) - Medium-term trend
        if current_price > sma_50: 
            score += shap_weights.get("trend_50", 8.6)
        
        # 6. Momentum 1M (Weight: ~8.6%) - Recent strength
        if mom_1m > 1.5: 
            score += shap_weights.get("mom_1m", 8.6)
        
        # ── Group B Adjustments (Rule-based Conviction Bonuses) ───────────────
        if alpha_6m > 0: score += 5       # Outperforming Nifty
        if vol_ratio > 1.5: score += 5    # Institutional volume spike
        if delivery_pct > 60.0: score += 5 # High delivery conviction

        # Cap final score between 0 and 100
        final_score = max(0.0, min(100.0, score))

        result["indicators"] = {
            "current_price": round(current_price, 2),
            "sma_200": round(sma_200, 2),
            "sma_50": round(sma_50, 2),
            "rsi": round(rsi, 1),
            "macd_hist": round(macd_data["hist"], 2),
            "bb_width": round(bb_width, 3),
            "mom_1m_pct": round(mom_1m, 1),
            "alpha_6m_pct": round(alpha_6m, 1),
            "vol_ratio": round(vol_ratio, 2),
            "delivery_pct": round(delivery_pct, 1)
        }
        result["tech_score"] = final_score
        return result

    except Exception as e:
        log.error(f"Technical analysis failed for {ticker}: {e}")
        result["error"] = str(e)
        return result

# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n📈 Testing technical_engine.py...\n")
    
    test_tickers = ["RELIANCE.NS", "TCS.NS"]
    
    for ticker in test_tickers:
        print(f"\n{'─'*55}")
        print(f"  {ticker} - Technical Report")
        print(f"{'─'*55}")
        
        res = analyse_technical(ticker)
        
        if res["error"]:
            print(f"  Error: {res['error']}")
        else:
            ind = res["indicators"]
            print(f"  Tech Score : {res['tech_score']}/100")
            print(f"  Price      : ₹{ind['current_price']} (SMA200: ₹{ind['sma_200']})")
            print(f"  RSI (14)   : {ind['rsi']}")
            print(f"  MACD Hist  : {ind['macd_hist']} (Bullish: {ind['macd_hist'] > 0})")
            print(f"  6M Alpha   : {ind['alpha_6m_pct']:+.1f}% vs Nifty")
            print(f"  Vol Ratio  : {ind['vol_ratio']}x average")
            print(f"  Delivery   : {ind['delivery_pct']}%")

    print(f"\n{'─'*55}")
    print("✅ technical_engine.py test complete.\n")