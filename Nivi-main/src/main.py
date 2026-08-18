"""
Nivi Oracle Engine — Main Entry Point  (v2 — Unified)
=======================================================
Runs the full pipeline and writes a rich CSV that the dashboard consumes.

Usage:
    python src/main.py                              # Full Nifty 500 scan
    python src/main.py --tickers RELIANCE.NS TCS.NS
    python src/main.py --sector Technology
    python src/main.py --top 20
    python src/main.py --no-allocation
    python src/main.py --capital 500000             # Set portfolio capital
    streamlit run src/dashboard.py                  # Launch UI after scan
"""

import argparse
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils              import get_logger, ensure_dirs
from oracle_engine      import generate_signal, allocate_black_litterman
from fundamental_engine import load_promoter_data
from macro_engine       import analyse_macro
from reality_simulator  import IndiaTradingCostModel
from sector_map         import NIFTY500_TICKERS, SECTOR_MAP, get_tickers_by_sector

log = get_logger("main")

OUTPUTS_DIR  = Path("outputs")
REPORTS_DIR  = OUTPUTS_DIR / "reports"
SCAN_CSV     = OUTPUTS_DIR / "nivi_oracle_complete.csv"

MAX_WORKERS  = 1           # Conservative — avoids Yahoo throttle burst
RETRY_DELAYS = [2, 5, 10]  # Exponential backoff on rate limit errors


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 65)
    print("  ███╗   ██╗██╗██╗   ██╗██╗")
    print("  ████╗  ██║██║██║   ██║██║")
    print("  ██╔██╗ ██║██║██║   ██║██║")
    print("  ██║╚██╗██║██║╚██╗ ██╔╝██║")
    print("  ██║ ╚████║██║ ╚████╔╝ ██║")
    print("  ╚═╝  ╚═══╝╚═╝  ╚═══╝  ╚═╝  Oracle Engine v2")
    print("  Value 35% · Quality 30% · Momentum 15% · Impact 15% · Sentiment 5%")
    print("=" * 65 + "\n")


def print_macro_state(macro_state: dict):
    regime = macro_state.get("regime", "UNKNOWN")
    mult   = macro_state.get("macro_multiplier", 1.0)
    vix    = macro_state.get("vix", "N/A")
    pcr    = macro_state.get("pcr", "N/A")
    fii    = macro_state.get("fii_net_cr", 0)
    dii    = macro_state.get("dii_net_cr", 0)
    emoji  = "🟢" if regime == "BULL" else "🔴" if regime == "BEAR" else "⚪"

    print(f"  {emoji} Market Regime    : {regime}  (Multiplier: {mult}x)")
    print(f"  📊 India VIX        : {vix}")
    print(f"  📐 Put-Call Ratio   : {pcr}")
    print(f"  🏦 FII Net (Cr)     : ₹{fii:+,.0f}")
    print(f"  🏦 DII Net (Cr)     : ₹{dii:+,.0f}")
    for r in macro_state.get("reasons", []):
        print(f"     ↳ {r}")
    print()


def print_result(res: dict):
    decision = res["decision"]
    score    = res["final_score"]
    ticker   = res["ticker"]
    buckets  = res.get("bucket_scores", {})

    icons = {
        "STRONG BUY":     "🟢🟢",
        "BUY":            "🟢",
        "HOLD / NEUTRAL": "🟡",
        "SELL":           "🔴",
        "STRONG SELL":    "🔴🔴",
        "REJECT":         "❌",
    }
    icon = icons.get(decision, "⚪")
    bar  = "█" * int(score // 5)

    bucket_str = ""
    if buckets:
        v = buckets.get("Value_35",    0)
        q = buckets.get("Quality_30",  0)
        m = buckets.get("Momentum_15", 0)
        i = buckets.get("Impact_15",   0)
        s = buckets.get("Sentiment_5", 0)
        bucket_str = f" [V:{v:.0f} Q:{q:.0f} M:{m:.0f} I:{i:.0f} S:{s:.0f}]"

    print(f"  {icon}  {ticker:<20} | {score:>5.1f}/100 | {decision:<15} |{bucket_str} {bar}")


def result_to_csv_row(res: dict) -> dict:
    """Flattens a signal result dict into a flat CSV row for the dashboard."""
    buckets  = res.get("bucket_scores", {})
    dol_dfl  = res.get("dol_dfl", {})
    arb      = res.get("arbitrage", {})
    deriv    = res.get("derivative", {})
    ind      = res.get("indicators", {})
    fund     = res.get("fundamental", {})
    sent     = res.get("sentiment", {})

    return {
        "Date":           datetime.now().strftime("%Y-%m-%d"),
        "Ticker":         res["ticker"],
        "Oracle_Score":   res["final_score"],
        "Decision":       res["decision"],
        "Sector":         SECTOR_MAP.get(res["ticker"], "Others"),

        # ── Bucket scores ──────────────────────────────────────────────────
        "Value_35":       buckets.get("Value_35",    0),
        "Quality_30":     buckets.get("Quality_30",  0),
        "Momentum_15":    buckets.get("Momentum_15", 0),
        "Impact_15":      buckets.get("Impact_15",   0),
        "Sentiment_5":    buckets.get("Sentiment_5", 0),

        # ── Fundamental (for Deep Dive tab) ───────────────────────────────
        "F_Score":        fund.get("f_score"),
        "DCF_Value":      fund.get("dcf_value"),
        "DCF_Upside":     fund.get("dcf_upside"),
        "Graham_Value":   fund.get("graham_value"),
        "Graham_Upside":  fund.get("graham_upside"),
        "Pledge_Pct":     fund.get("pledge_pct", 0),
        "Holding_Change": fund.get("holding_change", 0),
        "Promoter_Adj":   fund.get("promoter_adj", 0),

        # ── Technical indicators ───────────────────────────────────────────
        "Price":          ind.get("current_price"),
        "SMA_50":         ind.get("sma_50"),
        "SMA_200":        ind.get("sma_200"),
        "RSI":            ind.get("rsi"),
        "MACD_Hist":      ind.get("macd_hist"),
        "BB_Width":       ind.get("bb_width"),
        "Mom_1M":         ind.get("mom_1m_pct"),
        "Alpha_6M":       ind.get("alpha_6m_pct"),
        "Vol_Ratio":      ind.get("vol_ratio"),

        # ── DOL / DFL ──────────────────────────────────────────────────────
        "DOL":            dol_dfl.get("dol", 1.0),
        "DFL":            dol_dfl.get("dfl", 1.0),

        # ── Arbitrage ──────────────────────────────────────────────────────
        "Arb_Spread":     arb.get("spread_pct", 0),
        "Arb_Action":     arb.get("arb_action", "None"),
        "NSE_Price":      arb.get("nse_price"),
        "BSE_Price":      arb.get("bse_price"),

        # ── Derivative ─────────────────────────────────────────────────────
        "Deriv_Action":   deriv.get("action", "None"),
        "Strike":         deriv.get("strike"),
        "Option_Price":   deriv.get("option_price"),
        "Delta":          deriv.get("delta"),
        "Leverage":       deriv.get("leverage", 0),
        "ROI_Speed":      deriv.get("roi_impact", 0),
        "Implied_Vol":    deriv.get("implied_vol"),

        # ── Sentiment ──────────────────────────────────────────────────────
        "Sentiment":      sent.get("average_compound", 0),
        "Sent_Label":     sent.get("interpretation", "Neutral"),
        "News_Count":     sent.get("article_count", 0),

        # ── Macro ──────────────────────────────────────────────────────────
        "Macro_Mult":     res.get("macro_multiplier", 1.0),
        "Pre_Macro":      res.get("pre_macro_score", 0),
    }


def save_results(results: list, allocation: dict, macro_state: dict):
    """Saves results to CSV (for dashboard) and JSON (full detail)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV for dashboard
    import pandas as pd
    rows = [result_to_csv_row(r) for r in results]
    df   = pd.DataFrame(rows).sort_values("Oracle_Score", ascending=False)
    df.to_csv(SCAN_CSV, index=False)
    print(f"\n  📊 Scan CSV saved → {SCAN_CSV}")

    # Full JSON report
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"nivi_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "macro_state":  macro_state,
            "signals":      results,
            "allocation":   allocation,
        }, f, indent=4, default=str)
    print(f"  💾 Full report  saved → {report_path}")


# ── Worker ────────────────────────────────────────────────────────────────────

def _scan_ticker(ticker: str, promoter_data: dict, macro_state: dict) -> dict:
    _empty = {
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
        "fundamental":  {},
        "reasons":      [],
    }

    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            log.warning(f"{ticker} throttled — retrying in {delay}s (attempt {attempt + 1})")
            time.sleep(delay)
        try:
            return generate_signal(ticker, promoter_data, macro_state)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(x in err for x in
                ["rate limit", "too many requests", "429", "connection reset",
                 "yfratelimit", "remotedisconnected"])
            if is_rate_limit and attempt < len(RETRY_DELAYS):
                continue
            log.error(f"Pipeline failed for {ticker} after {attempt + 1} attempt(s): {e}")
            return {**_empty, "reasons": [f"Error after {attempt + 1} attempt(s): {e}"]}


# ── Core Scan ─────────────────────────────────────────────────────────────────

def run_scan(
    tickers:        list,
    top_n:          int   = None,
    run_allocation: bool  = True,
    capital:        float = 100_000.0,
):
    print_banner()

    # 1. Shared state
    print("⚙️  Loading shared state...")
    macro_state   = analyse_macro()
    print_macro_state(macro_state)

    promoter_data = load_promoter_data()
    print(f"  ✅ Promoter data loaded ({len(promoter_data)} entries)\n")

    # 2. Parallel scan
    total = len(tickers)
    print(f"🔍 Scanning {total} ticker(s) with {MAX_WORKERS} parallel workers...\n")
    print("  " + "─" * 75)

    all_results   = []
    passed_scores = {}
    completed     = 0
    start_time    = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_scan_ticker, t, promoter_data, macro_state): t
            for t in tickers
        }

        for future in as_completed(futures):
            result     = future.result()
            completed += 1
            all_results.append(result)
            print_result(result)

            if result["decision"] in ("BUY", "STRONG BUY"):
                passed_scores[result["ticker"]] = result["final_score"]

            if completed % 25 == 0:
                elapsed = time.time() - start_time
                rate    = completed / elapsed
                eta     = (total - completed) / rate if rate > 0 else 0
                print(f"\n  ⏱  {completed}/{total} | {rate:.1f}/sec | ETA: {eta:.0f}s\n")

    elapsed_total = time.time() - start_time
    print("  " + "─" * 75)

    # 3. Summary
    strong_buy = sum(1 for r in all_results if r["decision"] == "STRONG BUY")
    buys       = sum(1 for r in all_results if r["decision"] in ("BUY", "STRONG BUY"))
    holds      = sum(1 for r in all_results if r["decision"] == "HOLD / NEUTRAL")
    sells      = sum(1 for r in all_results if r["decision"] in ("SELL", "STRONG SELL"))
    rejects    = sum(1 for r in all_results if r["decision"] == "REJECT")

    print(f"\n📊 Scan Summary  (completed in {elapsed_total:.1f}s):")
    print(f"   Scanned     : {len(all_results)}")
    print(f"   Strong Buy  : {strong_buy}")
    print(f"   Buy         : {buys - strong_buy}")
    print(f"   Hold        : {holds}")
    print(f"   Sell/SS     : {sells}")
    print(f"   Rejected    : {rejects}")

    if top_n and len(passed_scores) > top_n:
        passed_scores = dict(
            sorted(passed_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        print(f"\n   ↳ Trimmed to top {top_n} for allocation.")

    # 4. Black-Litterman allocation
    allocation = {}
    if run_allocation and passed_scores:
        print(f"\n💼 Running Black-Litterman Allocation ({len(passed_scores)} stocks)...")
        try:
            allocation = allocate_black_litterman(passed_scores)
            print("\n  Optimal Portfolio Weights:")
            print("  " + "─" * 50)
            for ticker, weight in sorted(allocation.items(), key=lambda x: -x[1]):
                bar = "█" * int(weight // 2)
                print(f"  {ticker:<22} | {weight:>5.2f}%  {bar}")
            print("  " + "─" * 50)
        except Exception as e:
            log.error(f"Black-Litterman failed: {e}")
            print(f"\n  ⚠️  Allocation failed: {e}")
    elif not passed_scores:
        print("\n  ℹ️  No BUY signals — skipping allocation.")

    # 5. Transaction cost reality check
    if allocation:
        import yfinance as yf
        prices = {}
        for ticker in allocation:
            try:
                prices[ticker] = float(
                    yf.download(ticker, period="1d", progress=False,
                                auto_adjust=True)["Close"].iloc[-1]
                )
            except Exception:
                pass

        if prices:
            cost_model = IndiaTradingCostModel(capital=capital)
            cost_model.friction_report(allocation, prices)

    # 6. Save
    save_results(all_results, allocation, macro_state)
    print("\n✅ Nivi Oracle scan complete.")
    print("   👉 Launch dashboard: streamlit run src/dashboard.py\n")

    return all_results, allocation


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nivi Oracle Engine v2 — Quantitative Stock Scanner",
    )
    parser.add_argument("--tickers",  "-t", nargs="+", metavar="TICKER",
                        help="Space-separated NSE tickers.")
    parser.add_argument("--sector",   "-s", type=str,  metavar="SECTOR",
                        help="Scan one sector (e.g. Technology).")
    parser.add_argument("--top",      "-n", type=int,  metavar="N", default=None,
                        help="Keep only top N BUY signals for allocation.")
    parser.add_argument("--workers",  "-w", type=int,  default=MAX_WORKERS,
                        help=f"Parallel workers (default {MAX_WORKERS}).")
    parser.add_argument("--capital",  "-c", type=float, default=100_000.0,
                        help="Portfolio capital in ₹ for cost analysis (default 100000).")
    parser.add_argument("--no-allocation", action="store_true",
                        help="Skip Black-Litterman allocation.")
    parser.add_argument("--list-sectors",  action="store_true",
                        help="Print available sectors and exit.")
    return parser.parse_args()


def main():
    ensure_dirs()
    args = parse_args()

    global MAX_WORKERS
    MAX_WORKERS = args.workers

    if args.list_sectors:
        from collections import Counter
        dist = Counter(SECTOR_MAP.values())
        print("\nAvailable sectors:\n")
        for sector, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {sector:<22} ({count} stocks)")
        print()
        sys.exit(0)

    if args.tickers:
        tickers = args.tickers
    elif args.sector:
        tickers = get_tickers_by_sector(args.sector)
        if not tickers:
            print(f"\n❌ No tickers found for sector '{args.sector}'.")
            sys.exit(1)
    else:
        tickers = NIFTY500_TICKERS

    run_scan(
        tickers        = tickers,
        top_n          = args.top,
        run_allocation = not args.no_allocation,
        capital        = args.capital,
    )


if __name__ == "__main__":
    main()
