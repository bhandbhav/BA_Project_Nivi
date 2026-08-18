"""
Nivi Oracle Engine — Arbitrage Engine
======================================
Checks NSE vs BSE price spreads for the same stock.
A spread > 0.3% signals a temporary mispricing opportunity.

Ported from the original predict_daily.py arbitrage logic.
"""

import yfinance as yf
from utils import get_logger

log = get_logger("arbitrage_engine")

MIN_SPREAD_PCT = 0.30  # Minimum spread % to flag as an opportunity


def check_arbitrage(ticker_ns: str) -> dict:
    """
    Compares NSE (.NS) vs BSE (.BO) price for the same stock.

    Returns:
        spread_pct:   Absolute price difference as a % of average price
        arb_action:   'Buy BSE' / 'Buy NSE' / 'None'
        nse_price:    Last NSE price
        bse_price:    Last BSE price
        is_opportunity: True if spread > MIN_SPREAD_PCT
    """
    empty = {
        "spread_pct":       0.0,
        "arb_action":       "None",
        "nse_price":        None,
        "bse_price":        None,
        "is_opportunity":   False,
        "error":            None,
    }

    try:
        ticker_bo = ticker_ns.replace(".NS", ".BO")

        data = yf.download(
            [ticker_ns, ticker_bo],
            period="1d",
            progress=False,
            threads=False,
            auto_adjust=True,
        )

        if data.empty:
            return {**empty, "error": "No data returned"}

        import pandas as pd
        if isinstance(data.columns, pd.MultiIndex):
            try:
                p_nse = float(data["Close"][ticker_ns].iloc[-1])
                p_bse = float(data["Close"][ticker_bo].iloc[-1])
            except (KeyError, IndexError):
                return {**empty, "error": "Could not parse multi-index columns"}
        else:
            return {**empty, "error": "Unexpected data format"}

        if p_nse <= 0 or p_bse <= 0:
            return {**empty, "error": "Invalid prices (zero or negative)"}

        diff        = p_nse - p_bse
        avg         = (p_nse + p_bse) / 2
        spread_pct  = round((abs(diff) / avg) * 100, 3)

        if spread_pct < MIN_SPREAD_PCT:
            return {
                **empty,
                "nse_price":  round(p_nse, 2),
                "bse_price":  round(p_bse, 2),
                "spread_pct": spread_pct,
            }

        arb_action = "Buy BSE" if p_nse > p_bse else "Buy NSE"

        return {
            "spread_pct":     spread_pct,
            "arb_action":     arb_action,
            "nse_price":      round(p_nse, 2),
            "bse_price":      round(p_bse, 2),
            "is_opportunity": True,
            "error":          None,
        }

    except Exception as e:
        log.warning(f"Arbitrage check failed for {ticker_ns}: {e}")
        return {**empty, "error": str(e)}


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n⚡ Testing arbitrage_engine.py...\n")

    test_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    for ticker in test_tickers:
        res = check_arbitrage(ticker)
        if res["is_opportunity"]:
            print(f"  🔥 {ticker}: SPREAD {res['spread_pct']}% → {res['arb_action']}")
            print(f"       NSE: ₹{res['nse_price']} | BSE: ₹{res['bse_price']}")
        elif res["error"]:
            print(f"  ⚠️  {ticker}: Error — {res['error']}")
        else:
            print(f"  ✅ {ticker}: No arbitrage (Spread: {res['spread_pct']}%)")

    print("\n✅ arbitrage_engine.py test complete.\n")