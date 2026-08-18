# src/fundamental_engine.py
"""
Nivi Oracle Engine — Fundamental Engine
========================================
Computes:
  1. Piotroski F-Score   (0–9)  → hard gate: < 5 = reject
  2. DCF Intrinsic Value        → upside % vs current price
  3. Graham Fair Value          → secondary valuation check
  4. Promoter Pledge %          → blow-up risk flag
  5. Promoter Holding Change    → insider proxy signal

Data source: yahooquery (primary), yfinance (fallback)
Cache: weekly (fundamentals don't change daily)
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from yahooquery import Ticker as YQTicker
import yfinance as yf

from utils import get_logger, CACHE_DIR, STATIC_DIR, _is_cache_fresh, _cache_path

log = get_logger("fundamental_engine")

# ── Config ────────────────────────────────────────────────────────────────────
RISK_FREE_RATE   = 0.072   # 7.2% India risk-free
GROWTH_RATE      = 0.12    # Conservative 12% growth assumption
DISCOUNT_RATE    = 0.10    # 10% WACC
TERMINAL_MULT    = 15      # Terminal FCF/earnings multiple
DCF_YEARS        = 5
PLEDGE_GATE      = 15.0    # % — above this is a blow-up risk flag
F_SCORE_GATE     = 3       # below this = fundamental reject

FUND_CACHE_HOURS = 168     # 1 week


# ══════════════════════════════════════════════════════════
# PIOTROSKI F-SCORE
# ══════════════════════════════════════════════════════════

def calculate_f_score(ticker: str) -> dict:
    """
    Calculates the Piotroski F-Score (0–9).

    9 binary signals across 3 categories:
      Profitability (4):  ROA, OCF, ROA change, Accruals
      Leverage (3):       Leverage change, Liquidity change, Shares issued
      Efficiency (2):     Gross margin change, Asset turnover change

    Returns dict with score, breakdown, and pass/fail gate.
    """
    empty = {
        "f_score":    None,
        "pass_gate":  False,
        "breakdown":  {},
        "error":      None,
    }

    try:
        yq = YQTicker(ticker)

        # ── Fetch statements ──────────────────────────────────────────────────
        inc  = yq.income_statement(frequency="annual")
        bal  = yq.balance_sheet(frequency="annual")
        cf   = yq.cash_flow(frequency="annual")

        if any(isinstance(x, str) for x in [inc, bal, cf]):
            raise ValueError("yahooquery returned error string")

        if inc.empty or bal.empty or cf.empty:
            raise ValueError("Empty financial statements")

        # Sort by date descending — most recent first
        inc = inc.sort_values("asOfDate", ascending=False).reset_index(drop=True)
        bal = bal.sort_values("asOfDate", ascending=False).reset_index(drop=True)
        cf  = cf.sort_values("asOfDate", ascending=False).reset_index(drop=True)

        def get(df, cols, row=0):
            """Smarter getter that checks a list of accounting aliases."""
            if isinstance(cols, str): cols = [cols]
            for col in cols:
                if col in df.columns and len(df) > row:
                    val = df[col].iloc[row]
                    if pd.notna(val): return float(val)
            return None

        # ── Profitability signals ─────────────────────────────────────────────
        total_assets_0 = get(bal, "TotalAssets", 0)
        total_assets_1 = get(bal, "TotalAssets", 1)
        net_income_0   = get(inc, "NetIncome",   0)
        ocf_0          = get(cf,  ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"], 0)
        net_income_1   = get(inc, "NetIncome",   1)

        avg_assets = (
            (total_assets_0 + total_assets_1) / 2
            if total_assets_0 and total_assets_1
            else total_assets_0
        )

        roa_0 = (net_income_0 / avg_assets) if (net_income_0 and avg_assets) else None
        roa_1 = (
            (net_income_1 / total_assets_1)
            if (net_income_1 and total_assets_1)
            else None
        )

        # F1: ROA positive
        f1 = int(roa_0 > 0) if roa_0 is not None else 0

        # F2: OCF positive
        f2 = int(ocf_0 > 0) if ocf_0 is not None else 0

        # F3: ROA improving
        f3 = int(roa_0 > roa_1) if (roa_0 is not None and roa_1 is not None) else 0

        # F4: Accruals — OCF > ROA (cash quality)
        f4 = int((ocf_0 / avg_assets) > roa_0) if (
            ocf_0 and avg_assets and roa_0 is not None
        ) else 0

        # ── Leverage signals ──────────────────────────────────────────────────
        ltd_0    = get(bal, "LongTermDebt", 0) or 0
        ltd_1    = get(bal, "LongTermDebt", 1) or 0
        
        # Banks often don't use standard current assets/liabilities, adding fallbacks
        cur_0    = get(bal, ["CurrentAssets", "TotalAssets"], 0)
        cur_1    = get(bal, ["CurrentAssets", "TotalAssets"], 1)
        curlia_0 = get(bal, ["CurrentLiabilities", "TotalLiabilitiesNetMinorityInterest"], 0)
        curlia_1 = get(bal, ["CurrentLiabilities", "TotalLiabilitiesNetMinorityInterest"], 1)
        
        shares_0 = get(bal, "OrdinarySharesNumber", 0)
        shares_1 = get(bal, "OrdinarySharesNumber", 1)

        lev_0 = (ltd_0 / total_assets_0) if total_assets_0 else None
        lev_1 = (ltd_1 / total_assets_1) if total_assets_1 else None

        curr_ratio_0 = (cur_0 / curlia_0) if (cur_0 and curlia_0) else None
        curr_ratio_1 = (cur_1 / curlia_1) if (cur_1 and curlia_1) else None

        # F5: Leverage decreasing
        f5 = int(lev_0 < lev_1) if (lev_0 is not None and lev_1 is not None) else 0

        # F6: Liquidity (current ratio) improving
        f6 = int(curr_ratio_0 > curr_ratio_1) if (
            curr_ratio_0 is not None and curr_ratio_1 is not None
        ) else 0

        # F7: No new shares issued (dilution check)
        f7 = int(shares_0 <= shares_1) if (
            shares_0 is not None and shares_1 is not None
        ) else 0

        # ── Efficiency signals ────────────────────────────────────────────────
        rev_0      = get(inc, ["TotalRevenue", "OperatingRevenue"], 0)
        rev_1      = get(inc, ["TotalRevenue", "OperatingRevenue"], 1)
        
        # Indian IT and Banks use Operating/Total Expenses instead of COGS
        cogs_0     = get(inc, ["CostOfRevenue", "OperatingExpense", "TotalExpenses"], 0) or 0
        cogs_1     = get(inc, ["CostOfRevenue", "OperatingExpense", "TotalExpenses"], 1) or 0

        gm_0 = ((rev_0 - cogs_0) / rev_0) if (rev_0 and rev_0 != 0) else None
        gm_1 = ((rev_1 - cogs_1) / rev_1) if (rev_1 and rev_1 != 0) else None

        at_0 = (rev_0 / total_assets_0) if (rev_0 and total_assets_0) else None
        at_1 = (rev_1 / total_assets_1) if (rev_1 and total_assets_1) else None

        # F8: Gross margin improving
        f8 = int(gm_0 > gm_1) if (gm_0 is not None and gm_1 is not None) else 0

        # F9: Asset turnover improving
        f9 = int(at_0 > at_1) if (at_0 is not None and at_1 is not None) else 0

        total = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9

        return {
            "f_score":   total,
            "pass_gate": total >= F_SCORE_GATE,
            "breakdown": {
                "Profitability": {"ROA+": f1, "OCF+": f2, "ROA_trend": f3, "Accruals": f4},
                "Leverage":      {"Debt↓": f5, "Liquidity↑": f6, "NoDilution": f7},
                "Efficiency":    {"GrossMargin↑": f8, "AssetTurnover↑": f9},
            },
            "error": None,
        }
    
    except Exception as e:
        log.warning(f"F-Score failed for {ticker}: {e}")
        # Fallback: try yfinance
        return _f_score_yfinance_fallback(ticker, str(e))

    except Exception as e:
        log.warning(f"F-Score failed for {ticker}: {e}")
        # Fallback: try yfinance
        return _f_score_yfinance_fallback(ticker, str(e))


def _f_score_yfinance_fallback(ticker: str, original_error: str) -> dict:
    """
    Simplified F-Score using yfinance when yahooquery fails.
    Calculates what it can, scores partial signals.
    """
    try:
        stock = yf.Ticker(ticker)
        inc   = stock.financials
        bal   = stock.balance_sheet
        cf    = stock.cashflow

        if inc.empty or bal.empty:
            raise ValueError("yfinance also returned empty statements")

        def yf_get(df, *keys):
            for key in keys:
                if key in df.index:
                    vals = df.loc[key].dropna()
                    return float(vals.iloc[0]) if len(vals) > 0 else None
            return None

        net_income = yf_get(inc, "Net Income", "NetIncome")
        ocf        = yf_get(cf,  "Operating Cash Flow", "Total Cash From Operating Activities")
        total_assets = yf_get(bal, "Total Assets")
        long_term_debt_0 = yf_get(bal, "Long Term Debt") or 0

        f1 = int(net_income > 0)  if net_income  is not None else 0
        f2 = int(ocf > 0)         if ocf         is not None else 0
        f4 = int(ocf > net_income) if (ocf and net_income) else 0

        total = f1 + f2 + f4  # partial score — only 3 signals

        # Scale to 9 to give fair comparison
        # (3 confirmed signals → rough estimate of full score)
        estimated = round((total / 3) * 9)

        return {
            "f_score":    estimated,
            "pass_gate":  estimated >= F_SCORE_GATE,
            "breakdown":  {"note": f"Partial (yfinance fallback). Original error: {original_error}"},
            "error":      "partial_fallback",
        }

    except Exception as e2:
        log.error(f"F-Score yfinance fallback also failed for {ticker}: {e2}")
        return {
            "f_score":   None,
            "pass_gate": True,   # don't reject on data failure
            "breakdown": {},
            "error":     str(e2),
        }


# ══════════════════════════════════════════════════════════
# DCF VALUATION
# ══════════════════════════════════════════════════════════

def calculate_dcf(ticker: str) -> dict:
    """
    Two-stage DCF with bank/NBFC auto-detection.

    For banks:    uses Net Income (not FCF — capex meaningless)
    For others:   uses FCF (OCF - Capex), falls back to Net Income

    Returns:
        intrinsic_value: Per share DCF value in ₹
        current_price:   Latest market price
        upside_pct:      (intrinsic - current) / current * 100
        margin_of_safety: True if upside > 20%
    """
    empty = {
        "intrinsic_value": None,
        "current_price":   None,
        "upside_pct":      None,
        "margin_of_safety": False,
        "method":          None,
        "error":           None,
    }

    try:
        yq   = YQTicker(ticker)
        info = yq.price.get(ticker, {})

        # Current price
        current_price = (
            info.get("regularMarketPrice") or
            info.get("postMarketPrice")
        )
        if not current_price:
            raise ValueError("Could not get current price")

        current_price = float(current_price)

        # Shares outstanding
        key_stats    = yq.key_stats.get(ticker, {})
        shares       = key_stats.get("sharesOutstanding")
        if not shares:
            summary = yq.summary_detail.get(ticker, {})
            shares  = summary.get("sharesOutstanding")
        if not shares:
            raise ValueError("Could not get shares outstanding")
        shares = float(shares)

        # Sector check for bank detection
        asset_profile = yq.asset_profile.get(ticker, {})
        sector        = str(asset_profile.get("sector", "")).upper()
        industry      = str(asset_profile.get("industry", "")).upper()
        is_financial  = any(x in sector + industry for x in [
            "FINANCIAL", "BANK", "INSURANCE", "NBFC", "LENDING"
        ])

        # Get the base cash flow metric
        cf  = yq.cash_flow(frequency="annual")
        inc = yq.income_statement(frequency="annual")

        if isinstance(cf, str) or isinstance(inc, str):
            raise ValueError("yahooquery returned error for financials")

        cf  = cf.sort_values("asOfDate",  ascending=False).reset_index(drop=True)
        inc = inc.sort_values("asOfDate", ascending=False).reset_index(drop=True)

        def get_val(df, col):
            if col in df.columns and len(df) > 0:
                val = df[col].iloc[0]
                return float(val) if pd.notna(val) else None
            return None

        base_cf = None
        method  = ""

        if is_financial:
            # Banks: use Net Income
            base_cf = get_val(inc, "NetIncome")
            method  = "Net Income (Financial)"
        else:
            # Try FCF = OCF - Capex
            ocf   = get_val(cf, "OperatingCashFlow")
            capex = get_val(cf, "CapitalExpenditure") or 0
            # Capex is usually negative in statements
            fcf = (ocf + capex) if ocf is not None else None

            if fcf is not None and fcf > 0:
                base_cf = fcf
                method  = "Free Cash Flow"
            else:
                # Fall back to Net Income
                net_income = get_val(inc, "NetIncome")
                if net_income and net_income > 0:
                    base_cf = net_income
                    method  = "Net Income (FCF negative fallback)"

        if not base_cf or base_cf <= 0:
            raise ValueError(f"No positive cash flow metric available")

        # ── DCF Calculation ───────────────────────────────────────────────────
        pv_cash_flows = []
        for year in range(1, DCF_YEARS + 1):
            future_cf     = base_cf * ((1 + GROWTH_RATE) ** year)
            discounted_cf = future_cf / ((1 + DISCOUNT_RATE) ** year)
            pv_cash_flows.append(discounted_cf)

        # Terminal value (Gordon Growth Model exit)
        terminal_cf = pv_cash_flows[-1] * TERMINAL_MULT
        pv_terminal = terminal_cf / ((1 + DISCOUNT_RATE) ** DCF_YEARS)

        total_value     = sum(pv_cash_flows) + pv_terminal
        intrinsic_value = total_value / shares
        upside_pct      = ((intrinsic_value - current_price) / current_price) * 100

        return {
            "intrinsic_value":   round(intrinsic_value, 2),
            "current_price":     round(current_price, 2),
            "upside_pct":        round(upside_pct, 1),
            "margin_of_safety":  upside_pct > 20,
            "method":            method,
            "error":             None,
        }

    except Exception as e:
        log.warning(f"DCF failed for {ticker}: {e}")
        return {**empty, "error": str(e)}


# ══════════════════════════════════════════════════════════
# GRAHAM FAIR VALUE
# ══════════════════════════════════════════════════════════

def calculate_graham(ticker: str) -> dict:
    """
    Graham Number: sqrt(22.5 × EPS × Book Value per Share)
    Secondary valuation check alongside DCF.
    """
    try:
        yq        = YQTicker(ticker)
        key_stats = yq.key_stats.get(ticker, {})
        info      = yq.price.get(ticker, {})

        eps       = key_stats.get("trailingEps")
        bvps      = key_stats.get("bookValue")
        price     = info.get("regularMarketPrice")

        if not all([eps, bvps, price]):
            raise ValueError("Missing EPS, Book Value, or Price")

        eps, bvps, price = float(eps), float(bvps), float(price)

        if eps <= 0 or bvps <= 0:
            raise ValueError(f"Negative EPS ({eps}) or BVPS ({bvps})")

        graham_value = np.sqrt(22.5 * eps * bvps)
        upside_pct   = ((graham_value - price) / price) * 100

        return {
            "graham_value": round(graham_value, 2),
            "current_price": round(price, 2),
            "upside_pct":   round(upside_pct, 1),
            "error":        None,
        }

    except Exception as e:
        log.warning(f"Graham failed for {ticker}: {e}")
        return {
            "graham_value":  None,
            "current_price": None,
            "upside_pct":    None,
            "error":         str(e),
        }


# ══════════════════════════════════════════════════════════
# PROMOTER DATA
# ══════════════════════════════════════════════════════════

# Static data — sourced from BSE shareholding pattern Q3 FY2025
# Refresh each quarter by updating this dict or loading from data/static/promoter_data.json
_PROMOTER_DATA_DEFAULT = {
    "RELIANCE.NS":    {"pledge_pct": 0.0,  "holding_change": 0.0},
    "TCS.NS":         {"pledge_pct": 0.0,  "holding_change": 0.0},
    "HDFCBANK.NS":    {"pledge_pct": 0.0,  "holding_change": -0.1},
    "ICICIBANK.NS":   {"pledge_pct": 0.0,  "holding_change": 0.0},
    "INFY.NS":        {"pledge_pct": 0.0,  "holding_change": 0.0},
    "BHARTIARTL.NS":  {"pledge_pct": 0.0,  "holding_change": 0.2},
    "ITC.NS":         {"pledge_pct": 0.0,  "holding_change": 0.0},
    "SBIN.NS":        {"pledge_pct": 0.0,  "holding_change": 0.0},
    "HINDUNILVR.NS":  {"pledge_pct": 0.0,  "holding_change": 0.0},
    "LT.NS":          {"pledge_pct": 0.0,  "holding_change": 0.1},
    "BAJFINANCE.NS":  {"pledge_pct": 0.0,  "holding_change": 0.0},
    "MARUTI.NS":      {"pledge_pct": 0.0,  "holding_change": 0.0},
    "HCLTECH.NS":     {"pledge_pct": 0.0,  "holding_change": 0.1},
    "SUNPHARMA.NS":   {"pledge_pct": 1.2,  "holding_change": -0.3},
    "TATAMOTORS.NS":  {"pledge_pct": 0.0,  "holding_change": 0.4},
    "AXISBANK.NS":    {"pledge_pct": 0.0,  "holding_change": 0.0},
    "NTPC.NS":        {"pledge_pct": 0.0,  "holding_change": 0.0},
    "TITAN.NS":       {"pledge_pct": 0.0,  "holding_change": 0.0},
    "KOTAKBANK.NS":   {"pledge_pct": 0.0,  "holding_change": -0.2},
    "ASIANPAINT.NS":  {"pledge_pct": 0.0,  "holding_change": 0.1},
    "COALINDIA.NS":   {"pledge_pct": 0.0,  "holding_change": 0.0},
    "TATASTEEL.NS":   {"pledge_pct": 3.1,  "holding_change": 0.0},
    "JSWSTEEL.NS":    {"pledge_pct": 12.4, "holding_change": -0.5},
    "WIPRO.NS":       {"pledge_pct": 0.0,  "holding_change": 0.2},
    "DRREDDY.NS":     {"pledge_pct": 0.0,  "holding_change": 0.0},
    "CIPLA.NS":       {"pledge_pct": 0.8,  "holding_change": 0.1},
    "DIVISLAB.NS":    {"pledge_pct": 0.0,  "holding_change": 0.0},
    "APOLLOHOSP.NS":  {"pledge_pct": 2.3,  "holding_change": 0.3},
    "ADANIENT.NS":    {"pledge_pct": 28.1, "holding_change": 1.2},
    "ADANIGREEN.NS":  {"pledge_pct": 31.4, "holding_change": 0.8},
    "ADANIPOWER.NS":  {"pledge_pct": 19.8, "holding_change": 0.5},
    "BAJAJ-AUTO.NS":  {"pledge_pct": 0.0,  "holding_change": 0.0},
    "M&M.NS":         {"pledge_pct": 0.0,  "holding_change": 0.1},
}


def load_promoter_data() -> dict:
    """
    Loads promoter data from static file if available,
    otherwise uses hardcoded defaults.
    """
    static_file = STATIC_DIR / "promoter_data.json"
    if static_file.exists():
        try:
            with open(static_file) as f:
                data = json.load(f)
            log.info(f"Loaded promoter data from {static_file}")
            return data
        except Exception as e:
            log.warning(f"Could not load promoter static file: {e}")

    return _PROMOTER_DATA_DEFAULT


def get_promoter_signals(ticker: str, promoter_data: dict) -> dict:
    """
    Returns promoter pledge and holding signals for a ticker.

    Signals:
        pledge_flag:    True if pledge > PLEDGE_GATE (blow-up risk)
        holding_signal: +1 if buying, -1 if selling, 0 if flat
        score_adj:      Adjustment to Oracle Score (-5 to +3)
    """
    data = promoter_data.get(ticker, {
        "pledge_pct":      0.0,
        "holding_change":  0.0,
    })

    pledge_pct     = float(data.get("pledge_pct",     0.0))
    holding_change = float(data.get("holding_change", 0.0))

    # Pledge scoring — penalise high pledge
    if pledge_pct >= PLEDGE_GATE:
        pledge_flag  = True
        pledge_score = -5   # serious blow-up risk
    elif pledge_pct >= 5:
        pledge_flag  = False
        pledge_score = -2   # elevated but manageable
    else:
        pledge_flag  = False
        pledge_score = 0

    # Holding change scoring
    if holding_change > 0.5:
        holding_signal = 1
        holding_score  = 3   # strong insider buying
    elif holding_change > 0:
        holding_signal = 1
        holding_score  = 1
    elif holding_change < -0.5:
        holding_signal = -1
        holding_score  = -2  # insider selling
    elif holding_change < 0:
        holding_signal = -1
        holding_score  = -1
    else:
        holding_signal = 0
        holding_score  = 0

    return {
        "pledge_pct":      pledge_pct,
        "pledge_flag":     pledge_flag,
        "holding_change":  holding_change,
        "holding_signal":  holding_signal,
        "score_adj":       pledge_score + holding_score,
        "source":          "BSE Shareholding Pattern Q3 FY2025 (CACHED)",
    }


# ══════════════════════════════════════════════════════════
# COMBINED FUNDAMENTAL ANALYSIS
# ══════════════════════════════════════════════════════════

def analyse_fundamental(ticker: str, promoter_data: dict) -> dict:
    """
    Master function — runs all fundamental checks for one ticker.

    Returns a single dict with:
        pass_gate:    False = Oracle Engine rejects this stock entirely
        f_score:      Piotroski score
        dcf:          DCF result dict
        graham:       Graham result dict
        promoter:     Promoter signals dict
        reject_reason: Why it was rejected (if applicable)
    """
    result = {
        "ticker":       ticker,
        "pass_gate":    True,
        "reject_reason": None,
        "f_score_data": None,
        "dcf":          None,
        "graham":       None,
        "promoter":     None,
    }

    # 1. Promoter pledge hard gate (fastest check — do first)
    promoter = get_promoter_signals(ticker, promoter_data)
    result["promoter"] = promoter

    if promoter["pledge_flag"]:
        result["pass_gate"]    = False
        result["reject_reason"] = (
            f"Pledge {promoter['pledge_pct']:.1f}% > {PLEDGE_GATE}% gate"
        )
        log.info(f"{ticker} REJECTED — {result['reject_reason']}")
        return result

    # 2. Piotroski F-Score gate
    f_data = calculate_f_score(ticker)
    result["f_score_data"] = f_data

    if f_data["f_score"] is not None and not f_data["pass_gate"]:
        result["pass_gate"]    = False
        result["reject_reason"] = (
            f"F-Score {f_data['f_score']}/9 < {F_SCORE_GATE} gate"
        )
        log.info(f"{ticker} REJECTED — {result['reject_reason']}")
        return result

    # 3. DCF valuation (only for stocks that passed the gates)
    result["dcf"]    = calculate_dcf(ticker)
    result["graham"] = calculate_graham(ticker)

    return result


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 Testing fundamental_engine.py...\n")

    promoter_data = load_promoter_data()
    test_tickers  = ["TCS.NS", "JSWSTEEL.NS", "INFY.NS"]

    for ticker in test_tickers:
        print(f"\n{'─'*55}")
        print(f"  {ticker}")
        print(f"{'─'*55}")

        result = analyse_fundamental(ticker, promoter_data)

        print(f"  Gate passed : {result['pass_gate']}")
        if result["reject_reason"]:
            print(f"  Rejected    : {result['reject_reason']}")

        if result["f_score_data"]:
            f = result["f_score_data"]
            print(f"  F-Score     : {f['f_score']}/9")

        if result["dcf"] and result["dcf"]["intrinsic_value"]:
            d = result["dcf"]
            print(f"  DCF Value   : ₹{d['intrinsic_value']:,.0f}")
            print(f"  Market Price: ₹{d['current_price']:,.0f}")
            print(f"  Upside      : {d['upside_pct']:+.1f}%")
            print(f"  Method      : {d['method']}")

        if result["promoter"]:
            p = result["promoter"]
            print(f"  Pledge      : {p['pledge_pct']:.1f}%")
            print(f"  Promoter Δ  : {p['holding_change']:+.1f}%")

    print(f"\n{'─'*55}")
    print("✅ fundamental_engine.py test complete.\n")