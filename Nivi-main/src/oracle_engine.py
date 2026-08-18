"""
Nivi Oracle Engine — Master Orchestrator  (v2 — Unified)
==========================================================
Integrates the best of both versions:

SCORING FORMULA (Named Buckets, sums to 100):
  Value      35%  — DCF upside + Graham cross-check
  Quality    30%  — Piotroski F-Score + Promoter signals
  Momentum   15%  — SHAP-weighted technical score (SMA, RSI, MACD, BB, mom)
  Impact     15%  — Arbitrage spread OR derivative leverage
  Sentiment   5%  — VADER news compound score

GATES (applied before scoring):
  1. Promoter Pledge > 15%   → hard reject
  2. Piotroski F-Score < 3   → hard reject
  3. Macro multiplier        → scales final score (VIX / FII / PCR)

EXTRAS:
  - DOL / DFL (Degree of Operating & Financial Leverage)
  - Black-Litterman portfolio allocation (pure NumPy)
"""

import logging
import numpy as np
import pandas as pd

from fundamental_engine import analyse_fundamental, load_promoter_data
from technical_engine   import analyse_technical
from sentiment_engine   import analyse_sentiment
from macro_engine       import analyse_macro
from arbitrage_engine   import check_arbitrage
from derivative_engine  import analyse_derivative
from utils              import get_logger, fetch_prices

log = get_logger("oracle_engine")


# ══════════════════════════════════════════════════════════
# SCORING WEIGHTS  (must sum to 100)
# ══════════════════════════════════════════════════════════
W_VALUE     = 35.0
W_QUALITY   = 30.0
W_MOMENTUM  = 15.0
W_IMPACT    = 15.0
W_SENTIMENT =  5.0


# ══════════════════════════════════════════════════════════
# DOL / DFL CALCULATION
# ══════════════════════════════════════════════════════════

def calculate_dol_dfl(ticker: str) -> dict:
    """
    Degree of Operating Leverage (DOL) and Financial Leverage (DFL).

    DOL = % change in EBIT / % change in Revenue
    DFL = % change in EPS  / % change in EBIT

    A high DOL means fixed-cost heavy. A high DFL means debt-heavy.
    Both amplify earnings volatility.
    """
    empty = {"dol": 1.0, "dfl": 1.0, "error": None}

    try:
        from yahooquery import Ticker as YQTicker
        yq  = YQTicker(ticker)
        inc = yq.income_statement(frequency="annual")

        if isinstance(inc, str) or inc.empty:
            raise ValueError("Empty income statement")

        inc = inc.sort_values("asOfDate", ascending=False).reset_index(drop=True)
        if len(inc) < 2:
            raise ValueError("Need at least 2 years of data")

        def get(col, row=0):
            if col in inc.columns and len(inc) > row:
                val = inc[col].iloc[row]
                return float(val) if pd.notna(val) else None
            return None

        # DOL
        rev_0  = get("TotalRevenue", 0) or get("OperatingRevenue", 0)
        rev_1  = get("TotalRevenue", 1) or get("OperatingRevenue", 1)
        ebit_0 = get("EBIT", 0) or get("OperatingIncome", 0)
        ebit_1 = get("EBIT", 1) or get("OperatingIncome", 1)

        dol = 1.0
        if rev_1 and rev_1 != 0 and ebit_1 and ebit_1 != 0:
            pct_rev  = (rev_0  - rev_1)  / abs(rev_1)
            pct_ebit = (ebit_0 - ebit_1) / abs(ebit_1)
            if pct_rev != 0:
                dol = round(pct_ebit / pct_rev, 2)

        # DFL
        ni_0   = get("NetIncome", 0)
        ni_1   = get("NetIncome", 1)

        dfl = 1.0
        if ni_1 and ni_1 != 0 and ebit_1 and ebit_1 != 0:
            pct_ni   = (ni_0   - ni_1)   / abs(ni_1)
            pct_ebit = (ebit_0 - ebit_1) / abs(ebit_1)
            if pct_ebit != 0:
                dfl = round(pct_ni / pct_ebit, 2)

        return {"dol": dol, "dfl": dfl, "error": None}

    except Exception as e:
        log.warning(f"DOL/DFL failed for {ticker}: {e}")
        return empty


# ══════════════════════════════════════════════════════════
# MAIN SIGNAL GENERATOR
# ══════════════════════════════════════════════════════════

def generate_signal(ticker: str, promoter_data: dict, macro_state: dict) -> dict:
    """
    Runs the full Nivi pipeline for a single stock.

    Returns a result dict with:
        final_score:   0–100
        decision:      STRONG BUY / BUY / HOLD / SELL / STRONG SELL / REJECT
        bucket_scores: Named breakdown of the 5 buckets
        indicators:    Raw technical indicators
        dol_dfl:       Operating and financial leverage metrics
        arbitrage:     NSE/BSE spread info
        derivative:    Options recommendation
        reasons:       List of human-readable explanations
    """
    result = {
        "ticker":       ticker,
        "final_score":  0.0,
        "decision":     "REJECT",
        "bucket_scores": {},
        "breakdown":    {},
        "indicators":   {},
        "dol_dfl":      {},
        "arbitrage":    {},
        "derivative":   {},
        "sentiment":    {},
        "fundamental":  {},   # f_score, dcf, graham, promoter — for CSV + dashboard
        "reasons":      [],
    }

    # ── GATE 1 & 2: Fundamental Gates ────────────────────────────────────────
    fund_res = analyse_fundamental(ticker, promoter_data)
    if not fund_res["pass_gate"]:
        result["reasons"].append(f"Fundamental Reject: {fund_res['reject_reason']}")
        return result

    # Store full fundamental data for CSV / dashboard
    f_data_stored  = fund_res.get("f_score_data") or {}
    dcf_stored     = fund_res.get("dcf") or {}
    graham_stored  = fund_res.get("graham") or {}
    promoter_stored = fund_res.get("promoter") or {}

    result["fundamental"] = {
        "f_score":       f_data_stored.get("f_score"),
        "f_breakdown":   f_data_stored.get("breakdown", {}),
        "dcf_value":     dcf_stored.get("intrinsic_value"),
        "dcf_upside":    dcf_stored.get("upside_pct"),
        "dcf_price":     dcf_stored.get("current_price"),
        "graham_value":  graham_stored.get("graham_value"),
        "graham_upside": graham_stored.get("upside_pct"),
        "pledge_pct":    promoter_stored.get("pledge_pct", 0),
        "holding_change":promoter_stored.get("holding_change", 0),
        "promoter_adj":  promoter_stored.get("score_adj", 0),
    }

    # ── BUCKET 1: VALUE (35%) ────────────────────────────────────────────────
    val_score = 0.0
    dcf       = dcf_stored
    graham    = graham_stored

    if dcf.get("upside_pct") is not None:
        upside = dcf["upside_pct"]
        # Scale: 0% upside = 0pts, 50%+ upside = full 35pts. Cap at 35.
        val_score += min(max(upside, 0), 50) * (W_VALUE / 50)

    # Graham cross-check: add up to 5 bonus points if Graham also agrees
    if graham.get("upside_pct") is not None and graham["upside_pct"] > 20:
        val_score = min(val_score + 5, W_VALUE)

    result["bucket_scores"]["Value_35"] = round(val_score, 1)

    # ── BUCKET 2: QUALITY (30%) ──────────────────────────────────────────────
    qual_score = 0.0

    # F-Score: 0–9 mapped to 0–27
    f_data = f_data_stored
    if f_data.get("f_score") is not None:
        qual_score += (f_data["f_score"] / 9) * (W_QUALITY * 0.90)

    # Promoter signal: ±3 bonus points mapped to ±3
    promoter     = promoter_stored
    promoter_adj = promoter.get("score_adj", 0)
    qual_score   = min(max(qual_score + promoter_adj, 0), W_QUALITY)

    result["bucket_scores"]["Quality_30"] = round(qual_score, 1)

    # ── BUCKET 3: MOMENTUM (15%) — SHAP-weighted technical score ─────────────
    tech_res = analyse_technical(ticker)
    if tech_res.get("error"):
        result["reasons"].append(f"Technical Error: {tech_res['error']}")
        # Don't reject — continue with 0 momentum score
        mom_score = 0.0
    else:
        # tech_score is 0–100; remap to 0–15
        mom_score = (tech_res["tech_score"] / 100) * W_MOMENTUM

    result["bucket_scores"]["Momentum_15"] = round(mom_score, 1)
    result["indicators"] = tech_res.get("indicators", {})

    # ── BUCKET 4: IMPACT (15%) — Arbitrage OR Derivative ────────────────────
    impact_score = 0.0

    # 4a. Arbitrage check (NSE vs BSE spread)
    arb_res = check_arbitrage(ticker)
    result["arbitrage"] = arb_res

    if arb_res.get("is_opportunity"):
        # Spread 0.3%→2%+ mapped to 5→15 pts
        impact_score += min((arb_res["spread_pct"] / 2.0) * W_IMPACT, W_IMPACT)
        result["reasons"].append(
            f"Arbitrage opportunity: {arb_res['spread_pct']}% spread → {arb_res['arb_action']}"
        )

    # 4b. Derivative leverage (if sentiment is strong and arb didn't fill the bucket)
    sent_res      = analyse_sentiment(ticker)
    result["sentiment"] = sent_res          # ← store for CSV
    sent_compound = sent_res.get("average_compound", 0.0)

    curr_price = result["indicators"].get("current_price")
    if curr_price and not arb_res.get("is_opportunity"):
        # Calculate volatility from indicators or fall back
        import yfinance as yf
        try:
            hist = yf.Ticker(ticker).history(period="3mo")
            vol  = float(hist["Close"].pct_change().std() * (252 ** 0.5))
        except Exception:
            vol  = 0.25  # fallback 25% vol

        deriv_res = analyse_derivative(ticker, sent_compound, curr_price, vol)
        result["derivative"] = deriv_res

        if deriv_res.get("leverage", 0) > 10:
            # Leverage 10x→30x+ mapped to 5→15 pts
            impact_score = max(
                impact_score,
                min((deriv_res["leverage"] / 30.0) * W_IMPACT, W_IMPACT)
            )
    else:
        result["derivative"] = {"action": "Skipped (arbitrage took priority)"}

    result["bucket_scores"]["Impact_15"] = round(impact_score, 1)

    # ── BUCKET 5: SENTIMENT (5%) ─────────────────────────────────────────────
    # Hard floor: strongly negative news zeroes out the entire score
    if sent_compound <= -0.50:
        result["decision"] = "REJECT"
        result["reasons"].append(
            f"Sentiment Disaster: compound={sent_compound:.2f} — automatic reject"
        )
        return result

    # Positive sentiment: 0→5 pts scaled by |compound|/0.5
    sent_score = min((abs(sent_compound) / 0.5) * W_SENTIMENT, W_SENTIMENT)
    if sent_compound < 0:
        sent_score = 0.0   # Negative but not catastrophic → no bonus

    result["bucket_scores"]["Sentiment_5"] = round(sent_score, 1)

    # ── DOL / DFL (informational, not in score) ──────────────────────────────
    result["dol_dfl"] = calculate_dol_dfl(ticker)

    # ── PRE-MACRO SCORE ───────────────────────────────────────────────────────
    pre_macro = val_score + qual_score + mom_score + impact_score + sent_score

    # ── GATE 3: MACRO MULTIPLIER ──────────────────────────────────────────────
    macro_mult  = macro_state.get("macro_multiplier", 1.0)
    final_score = max(0.0, min(100.0, pre_macro * macro_mult))

    # ── DECISION LOGIC ────────────────────────────────────────────────────────
    if   final_score >= 80: decision = "STRONG BUY"
    elif final_score >= 65: decision = "BUY"
    elif final_score <= 20: decision = "STRONG SELL"
    elif final_score <= 35: decision = "SELL"
    else:                   decision = "HOLD / NEUTRAL"

    result.update({
        "final_score":     round(final_score, 1),
        "pre_macro_score": round(pre_macro, 1),
        "macro_multiplier": macro_mult,
        "decision":        decision,
    })

    # Legacy breakdown key (keeps compatibility with main.py report)
    result["breakdown"] = {
        "Value_35":      result["bucket_scores"]["Value_35"],
        "Quality_30":    result["bucket_scores"]["Quality_30"],
        "Momentum_15":   result["bucket_scores"]["Momentum_15"],
        "Impact_15":     result["bucket_scores"]["Impact_15"],
        "Sentiment_5":   result["bucket_scores"]["Sentiment_5"],
        "Macro_Mult":    macro_mult,
    }

    result["reasons"].append("Passed all fundamental gates.")
    if macro_mult != 1.0:
        result["reasons"].append(
            f"Macro regime adjusted score by {macro_mult}x "
            f"(VIX={macro_state.get('vix', 'N/A')}, "
            f"PCR={macro_state.get('pcr', 'N/A')})"
        )

    log.info(
        f"{ticker} | {decision} | Score: {final_score:.1f} "
        f"[V:{val_score:.0f} Q:{qual_score:.0f} M:{mom_score:.0f} "
        f"I:{impact_score:.0f} S:{sent_score:.0f}]"
    )

    return result


# ══════════════════════════════════════════════════════════
# BLACK-LITTERMAN PORTFOLIO ALLOCATION (Pure NumPy)
# ══════════════════════════════════════════════════════════

def allocate_black_litterman(scores_dict: dict, period: str = "1y") -> dict:
    """
    Converts Nivi Oracle Scores into optimal portfolio weights
    using the Black-Litterman model (pure NumPy implementation).

    Args:
        scores_dict: {ticker: final_score (0-100)}
        period:      Historical period for covariance estimation

    Returns:
        {ticker: weight_pct}
    """
    tickers = list(scores_dict.keys())
    prices  = fetch_prices(tickers, period=period)

    if prices.empty:
        log.error("Price fetch failed for Black-Litterman allocation.")
        return {}

    valid_tickers = [t for t in tickers if t in prices.columns]
    prices        = prices[valid_tickers]
    N             = len(valid_tickers)

    if N == 0:
        return {}

    returns = prices.pct_change().dropna()
    Sigma   = returns.cov().values * 252

    risk_aversion = 2.5
    tau           = 0.05
    w_mkt         = np.ones(N) / N
    Pi            = risk_aversion * Sigma.dot(w_mkt)

    P = np.eye(N)
    Q = np.array([
        ((scores_dict[t] - 50) / 100.0) + 0.05
        for t in valid_tickers
    ])

    Omega    = np.diag(np.diag(P.dot(tau * Sigma).dot(P.T)))
    tau_S_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)

    term1 = np.linalg.inv(tau_S_inv + P.T.dot(Omega_inv).dot(P))
    term2 = tau_S_inv.dot(Pi) + P.T.dot(Omega_inv).dot(Q)

    posterior_returns = term1.dot(term2)
    w_opt = np.linalg.inv(risk_aversion * Sigma).dot(posterior_returns)

    w_opt = np.maximum(w_opt, 0)
    if np.sum(w_opt) > 0:
        w_opt = w_opt / np.sum(w_opt)

    return {valid_tickers[i]: round(w_opt[i] * 100, 2) for i in range(N)}


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧠 Testing oracle_engine.py (v2 — Unified Scoring)...\n")

    promoter_data = load_promoter_data()
    macro_state   = analyse_macro()
    test_tickers  = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS"]

    nivi_scores = {}
    for t in test_tickers:
        signal = generate_signal(t, promoter_data, macro_state)

        print(f"\n{'─'*60}")
        print(f"  {t}  →  {signal['decision']}  ({signal['final_score']}/100)")
        print(f"{'─'*60}")
        print(f"  Buckets : {signal['bucket_scores']}")
        print(f"  DOL/DFL : {signal['dol_dfl']}")
        print(f"  Arb     : spread={signal['arbitrage'].get('spread_pct', 0)}%")
        print(f"  Deriv   : {signal['derivative'].get('action', 'N/A')}")
        print(f"  Reasons : {signal['reasons']}")

        if signal["decision"] not in ("REJECT",):
            nivi_scores[t] = signal["final_score"]

    if nivi_scores:
        print(f"\n\n💼 Running Black-Litterman Allocation on {len(nivi_scores)} stocks...")
        weights = allocate_black_litterman(nivi_scores)
        for t, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"   ↳ {t}: {w:.2f}%")

    print("\n✅ oracle_engine.py v2 test complete.\n")
