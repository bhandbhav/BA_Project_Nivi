"""
Nivi Oracle Engine — Derivative Engine
========================================
Black-Scholes option pricer with:
  - Delta calculation (directional sensitivity)
  - Leverage ratio (capital efficiency)
  - ROI Impact = |Delta × Leverage| (expected ROI speed)

Triggered when sentiment score is strong enough to suggest
a directional options trade (call or put).

Ported and improved from predict_daily.py + auto_derivative_engine.py.
"""

import numpy as np
from scipy.stats import norm
from utils import get_logger

log = get_logger("derivative_engine")

RISK_FREE_RATE  = 0.072   # 7.2% India risk-free
EXPIRY_DAYS     = 7       # 1-week default expiry
OTM_FACTOR      = 0.02    # 2% OTM strike
MIN_SENTIMENT   = 0.05    # Minimum |sentiment| to trigger analysis


def black_scholes(S: float, K: float, T: float, sigma: float,
                  option_type: str = "call") -> tuple[float, float]:
    """
    Calculates Black-Scholes theoretical option price and Delta.

    Args:
        S:           Spot price
        K:           Strike price
        T:           Time to expiry in years
        sigma:       Annualised volatility
        option_type: 'call' or 'put'

    Returns:
        (price, delta)
    """
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0

    d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-RISK_FREE_RATE * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-RISK_FREE_RATE * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    return round(float(price), 2), round(float(delta), 4)


def analyse_derivative(
    ticker:    str,
    sentiment: float,
    price:     float,
    volatility: float,
) -> dict:
    """
    Given a stock's sentiment, current price, and volatility,
    calculates the recommended option trade and its impact metrics.

    Args:
        ticker:     NSE ticker string (for logging)
        sentiment:  VADER compound score (-1 to +1)
        price:      Current spot price
        volatility: Annualised historical volatility (e.g. 0.25 = 25%)

    Returns dict with:
        action:       e.g. 'BUY CALL', 'BUY PUT', or 'None'
        strike:       Recommended strike price
        option_price: Black-Scholes fair value of the option
        delta:        Option delta
        leverage:     Spot / Option price (capital efficiency)
        roi_impact:   |delta × leverage| — ROI speed multiplier
        implied_vol:  Volatility used in calculation
        sentiment:    Input sentiment score
        error:        Error string if calculation failed
    """
    empty = {
        "action":       "None",
        "strike":       None,
        "option_price": None,
        "delta":        None,
        "leverage":     0.0,
        "roi_impact":   0.0,
        "implied_vol":  round(volatility * 100, 1) if volatility else None,
        "sentiment":    round(sentiment, 4),
        "error":        None,
    }

    try:
        if abs(sentiment) < MIN_SENTIMENT:
            return {**empty, "action": "None (Neutral Sentiment)"}

        if price is None or price <= 0:
            return {**empty, "error": "Invalid price"}

        if volatility is None or volatility <= 0:
            return {**empty, "error": "Invalid volatility"}

        T = EXPIRY_DAYS / 365

        if sentiment > 0:
            option_type = "call"
            strike      = round(price * (1 + OTM_FACTOR), 0)
            action      = "BUY CALL"
        else:
            option_type = "put"
            strike      = round(price * (1 - OTM_FACTOR), 0)
            action      = "BUY PUT"

        opt_price, delta = black_scholes(price, strike, T, volatility, option_type)

        if opt_price <= 0:
            return {**empty, "error": "Option price calculated as zero (deep OTM)"}

        leverage   = round(price / opt_price, 1)
        roi_impact = round(abs(delta * leverage), 2)

        return {
            "action":       action,
            "strike":       strike,
            "option_price": opt_price,
            "delta":        delta,
            "leverage":     leverage,
            "roi_impact":   roi_impact,
            "implied_vol":  round(volatility * 100, 1),
            "sentiment":    round(sentiment, 4),
            "error":        None,
        }

    except Exception as e:
        log.warning(f"Derivative analysis failed for {ticker}: {e}")
        return {**empty, "error": str(e)}


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    print("\n📐 Testing derivative_engine.py...\n")

    test_cases = [
        ("RELIANCE.NS", 0.65),    # Bullish
        ("TATASTEEL.NS", -0.45),  # Bearish
        ("INFY.NS", 0.02),        # Neutral
    ]

    for ticker, sentiment in test_cases:
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period="3mo")
            price = float(hist["Close"].iloc[-1])
            vol   = float(hist["Close"].pct_change().std() * np.sqrt(252))

            res = analyse_derivative(ticker, sentiment, price, vol)

            print(f"  {ticker} (Sentiment: {sentiment:+.2f})")
            print(f"    Action      : {res['action']}")
            if res["action"] not in ("None", "None (Neutral Sentiment)"):
                print(f"    Strike      : ₹{res['strike']}")
                print(f"    Option Price: ₹{res['option_price']}")
                print(f"    Delta       : {res['delta']}")
                print(f"    Leverage    : {res['leverage']}x")
                print(f"    ROI Impact  : {res['roi_impact']}x")
                print(f"    Implied Vol : {res['implied_vol']}%")
            print()
        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}\n")

    print("✅ derivative_engine.py test complete.\n")
