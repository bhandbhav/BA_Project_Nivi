"""
Nivi Oracle Engine — Macro Engine
==================================
Computes market-wide systemic risk:
  1. India VIX (Fear Gauge)
  2. FII/DII Net Flow (Institutional Conviction)
  3. Put-Call Ratio (PCR) (Contrarian Sentiment)

Returns a `macro_multiplier` applied to final stock scores.
< 1.0 = Risk Off (Haircut scores)
> 1.0 = Risk On  (Boost scores)
"""

import logging
import pandas as pd
from utils import get_logger, fetch_prices_single

log = get_logger("macro_engine")

def get_india_vix() -> float:
    """Fetches the latest India VIX close."""
    try:
        # Use utils to grab VIX using Yahoo's ticker
        vix = fetch_prices_single("^INDIAVIX", period="1mo")
        return float(vix.iloc[-1]) if not vix.empty else 15.0
    except Exception as e:
        log.warning(f"Failed to fetch India VIX: {e}. Defaulting to 15.0")
        return 15.0

def get_fii_dii_flow() -> dict:
    """
    Returns net institutional flows in Crores.
    In production, this scrapes NSE. For the demo, we use a static placeholder 
    representing a mild sell-off scenario.
    """
    return {
        "fii_net_cr": -1250.50,  # FIIs selling
        "dii_net_cr": 850.00,    # DIIs absorbing
        "net_institutional": -400.50
    }

def get_pcr() -> float:
    """
    Returns Nifty Put-Call Ratio.
    > 1.5 = Overbought (Bearish)
    < 0.7 = Oversold (Bullish)
    """
    return 0.85 # Neutral/Slightly bullish placeholder

def analyse_macro() -> dict:
    """
    Synthesizes VIX, Flows, and PCR into a master risk multiplier.
    """
    vix = get_india_vix()
    flow = get_fii_dii_flow()
    pcr = get_pcr()
    
    multiplier = 1.0 # Baseline Neutral
    reasons = []

    # ── 1. India VIX Gate (Volatility) ──────────────────────────────────────
    if vix > 22.0:
        multiplier -= 0.20
        reasons.append(f"Panic Level VIX ({vix:.1f}) -> -20% Risk Haircut")
    elif vix < 15.0:
        multiplier += 0.10
        reasons.append(f"Stable VIX ({vix:.1f}) -> +10% Confidence Boost")

    # ── 2. Flow Signal (Liquidity) ──────────────────────────────────────────
    if flow["net_institutional"] < -2000:
        multiplier -= 0.10
        reasons.append("Heavy Institutional Selling (FII+DII) -> -10% Haircut")
    elif flow["net_institutional"] > 2000:
        multiplier += 0.10
        reasons.append("Heavy Institutional Buying (FII+DII) -> +10% Boost")

    # ── 3. PCR Contrarian Gate (Options Sentiment) ──────────────────────────
    if pcr > 1.5:
        multiplier -= 0.10
        reasons.append(f"PCR Overbought ({pcr}) -> -10% Contrarian Haircut")
    elif pcr < 0.7:
        multiplier += 0.10
        reasons.append(f"PCR Oversold ({pcr}) -> +10% Contrarian Boost")

    return {
        "vix": round(vix, 2),
        "fii_net_cr": flow["fii_net_cr"],
        "dii_net_cr": flow["dii_net_cr"],
        "pcr": pcr,
        "macro_multiplier": round(multiplier, 2),
        "reasons": reasons
    }

# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌍 Testing macro_engine.py...\n")
    
    state = analyse_macro()
    
    print(f"  India VIX       : {state['vix']}")
    print(f"  FII Net (Cr)    : ₹{state['fii_net_cr']}")
    print(f"  DII Net (Cr)    : ₹{state['dii_net_cr']}")
    print(f"  Put-Call Ratio  : {state['pcr']}\n")
    
    print(f"  🎯 FINAL MULTIPLIER : {state['macro_multiplier']}x")
    for r in state["reasons"]:
        print(f"     ↳ {r}")
        
    print("\n✅ macro_engine.py test complete.\n")