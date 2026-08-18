# src/sector_map.py
"""
Nivi Oracle Engine — Sector Map
================================
Builds and maintains the full Nifty 500 ticker → sector mapping.

Usage:
    from sector_map import SECTOR_MAP, NIFTY500_TICKERS

Refresh manually:
    python src/sector_map.py
"""

import requests
import pandas as pd
import json
import os
import io
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
CACHE_DIR  = Path(os.getenv("CACHE_DIR", "data/cache"))
CACHE_FILE = CACHE_DIR / "sector_map.json"
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | sector_map | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

NSE_URLS = [
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": "https://www.nseindia.com",
}


# ── Sector normalisation ─────────────────────────────────────────────────────
SECTOR_RULES = [
    (["BANK", "FINANCIAL", "NBFC", "INSURANCE", "MICROFINANCE"],  "Financials"),
    (["INFORMATION TECH", "SOFTWARE", "COMPUTER", " IT "],         "Technology"),
    (["PHARMA", "HEALTH", "HOSPITAL", "MEDICAL", "BIOTECH"],       "Healthcare"),
    (["AUTO", "VEHICLE", "TYRE", "ANCILLAR"],                      "Auto"),
    (["OIL", "GAS", "PETROLEUM", "ENERGY", "POWER", "COAL"],       "Energy"),
    (["FMCG", "CONSUMER", "RETAIL", "FOOD", "BEVERAGE", "TOBACCO"],"Consumer"),
    (["CEMENT", "CONSTRUCT", "INFRA", "REALTY", "HOUSING"],        "Construction"),
    (["METAL", "STEEL", "MINING", "ALUMIN", "COPPER", "ZINC"],     "Materials"),
    (["TELECOM", "COMMUNICATION"],                                  "Telecom"),
    (["CHEMICAL", "FERTILISER", "PESTICIDE", "AGROCHEMI"],         "Chemicals"),
    (["MEDIA", "ENTERTAINMENT", "BROADCAST"],                      "Media"),
    (["TEXTILE", "APPAREL", "GARMENT", "FIBRE"],                   "Textiles"),
    (["TRANSPORT", "LOGISTICS", "SHIPPING", "AVIATION"],           "Logistics"),
]

def normalise_sector(raw: str) -> str:
    r = str(raw).upper().strip()
    for keywords, sector in SECTOR_RULES:
        if any(k in r for k in keywords):
            return sector
    return "Others"


# ── NSE fetch ────────────────────────────────────────────────────────────────
def _fetch_from_nse() -> pd.DataFrame | None:
    session = requests.Session()
    # Warm up cookie
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=8)
    except Exception:
        pass

    for url in NSE_URLS:
        try:
            log.info(f"Trying NSE URL: {url}")
            r = session.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Series" in df.columns:
                    df = df[df["Series"].str.strip() == "EQ"]
                log.info(f"Fetched {len(df)} constituents from NSE.")
                return df
            else:
                log.warning(f"NSE returned {r.status_code} for {url}")
        except Exception as e:
            log.warning(f"Failed {url}: {e}")

    return None


# ── Build map ────────────────────────────────────────────────────────────────
def build_sector_map(force_refresh: bool = False) -> dict:
    """
    Builds {ticker: sector} for full Nifty 500.

    Priority:
        1. NSE live download
        2. Local JSON cache
        3. Hardcoded top-150 fallback

    Args:
        force_refresh: If True, skips cache and hits NSE directly.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load from cache if fresh enough (and not forcing refresh)
    if not force_refresh and CACHE_FILE.exists():
        age_days = (
            pd.Timestamp.now() - pd.Timestamp(CACHE_FILE.stat().st_mtime, unit="s")
        ).days
        if age_days < 7:
            log.info(f"Loading sector map from cache (age: {age_days}d).")
            with open(CACHE_FILE) as f:
                return json.load(f)
        else:
            log.info(f"Cache is {age_days} days old — refreshing from NSE.")

    # Try NSE
    df = _fetch_from_nse()

    if df is not None:
        sector_map = {}
        symbol_col  = "Symbol"   if "Symbol"   in df.columns else df.columns[0]
        industry_col = "Industry" if "Industry" in df.columns else (
                        "Sector"  if "Sector"   in df.columns else None)

        for _, row in df.iterrows():
            sym = str(row.get(symbol_col, "")).strip()
            ind = str(row.get(industry_col, "Others")).strip() if industry_col else "Others"
            if sym:
                sector_map[f"{sym}.NS"] = normalise_sector(ind)

        # Persist
        with open(CACHE_FILE, "w") as f:
            json.dump(sector_map, f, indent=2)
        log.info(f"Sector map saved → {CACHE_FILE} ({len(sector_map)} stocks)")
        return sector_map

    # Fallback to cache even if stale
    if CACHE_FILE.exists():
        log.warning("NSE unreachable — loading stale cache.")
        with open(CACHE_FILE) as f:
            return json.load(f)

    # Last resort — hardcoded
    log.warning("Using hardcoded fallback (top 150 stocks).")
    return _hardcoded_fallback()


# ── Hardcoded fallback ───────────────────────────────────────────────────────
def _hardcoded_fallback() -> dict:
    """Top 150 Nifty stocks hardcoded as final fallback."""
    return {
        # Financials
        "HDFCBANK.NS":"Financials",    "ICICIBANK.NS":"Financials",
        "SBIN.NS":"Financials",        "KOTAKBANK.NS":"Financials",
        "AXISBANK.NS":"Financials",    "BAJFINANCE.NS":"Financials",
        "BAJAJFINSV.NS":"Financials",  "HDFCLIFE.NS":"Financials",
        "SBILIFE.NS":"Financials",     "ICICIPRULI.NS":"Financials",
        "BANDHANBNK.NS":"Financials",  "FEDERALBNK.NS":"Financials",
        "IDFCFIRSTB.NS":"Financials",  "PNB.NS":"Financials",
        "BANKBARODA.NS":"Financials",  "INDUSINDBK.NS":"Financials",
        "LICI.NS":"Financials",        "MUTHOOTFIN.NS":"Financials",
        "CHOLAFIN.NS":"Financials",    "M&MFIN.NS":"Financials",
        "SHRIRAMFIN.NS":"Financials",  "CANBK.NS":"Financials",
        "UNIONBANK.NS":"Financials",   "MANAPPURAM.NS":"Financials",
        # Technology
        "TCS.NS":"Technology",         "INFY.NS":"Technology",
        "HCLTECH.NS":"Technology",     "WIPRO.NS":"Technology",
        "TECHM.NS":"Technology",       "LTIM.NS":"Technology",
        "PERSISTENT.NS":"Technology",  "COFORGE.NS":"Technology",
        "MPHASIS.NS":"Technology",     "OFSS.NS":"Technology",
        "KPITTECH.NS":"Technology",    "TATAELXSI.NS":"Technology",
        # Energy
        "RELIANCE.NS":"Energy",        "NTPC.NS":"Energy",
        "POWERGRID.NS":"Energy",       "ONGC.NS":"Energy",
        "COALINDIA.NS":"Energy",       "BPCL.NS":"Energy",
        "IOC.NS":"Energy",             "GAIL.NS":"Energy",
        "TATAPOWER.NS":"Energy",       "ADANIPOWER.NS":"Energy",
        "ADANIGREEN.NS":"Energy",      "ADANIENT.NS":"Energy",
        "CESC.NS":"Energy",            "TORNTPOWER.NS":"Energy",
        # Consumer
        "HINDUNILVR.NS":"Consumer",    "ITC.NS":"Consumer",
        "NESTLEIND.NS":"Consumer",     "BRITANNIA.NS":"Consumer",
        "DABUR.NS":"Consumer",         "MARICO.NS":"Consumer",
        "GODREJCP.NS":"Consumer",      "COLPAL.NS":"Consumer",
        "TATACONSUM.NS":"Consumer",    "VBL.NS":"Consumer",
        "TITAN.NS":"Consumer",         "ASIANPAINT.NS":"Consumer",
        "BERGEPAINT.NS":"Consumer",    "PIDILITIND.NS":"Consumer",
        "EMAMILTD.NS":"Consumer",      "RADICO.NS":"Consumer",
        # Auto
        "MARUTI.NS":"Auto",            "TATAMOTORS.NS":"Auto",
        "M&M.NS":"Auto",               "BAJAJ-AUTO.NS":"Auto",
        "HEROMOTOCO.NS":"Auto",        "EICHERMOT.NS":"Auto",
        "TVSMOTOR.NS":"Auto",          "BOSCHLTD.NS":"Auto",
        "MOTHERSON.NS":"Auto",         "BALKRISIND.NS":"Auto",
        "APOLLOTYRE.NS":"Auto",        "MRF.NS":"Auto",
        "BHARATFORG.NS":"Auto",        "ASHOKLEY.NS":"Auto",
        # Healthcare
        "SUNPHARMA.NS":"Healthcare",   "DRREDDY.NS":"Healthcare",
        "CIPLA.NS":"Healthcare",       "DIVISLAB.NS":"Healthcare",
        "APOLLOHOSP.NS":"Healthcare",  "TORNTPHARM.NS":"Healthcare",
        "BIOCON.NS":"Healthcare",      "AUROPHARMA.NS":"Healthcare",
        "LUPIN.NS":"Healthcare",       "ALKEM.NS":"Healthcare",
        "IPCALAB.NS":"Healthcare",     "GLENMARK.NS":"Healthcare",
        "ABBOTINDIA.NS":"Healthcare",  "PFIZER.NS":"Healthcare",
        # Construction & Realty
        "LT.NS":"Construction",        "ULTRACEMCO.NS":"Construction",
        "GRASIM.NS":"Construction",    "SHREECEM.NS":"Construction",
        "AMBUJACEM.NS":"Construction", "ACC.NS":"Construction",
        "DLF.NS":"Construction",       "GODREJPROP.NS":"Construction",
        "LODHA.NS":"Construction",     "OBEROIRLTY.NS":"Construction",
        "PRESTIGE.NS":"Construction",  "PHOENIXLTD.NS":"Construction",
        # Materials
        "TATASTEEL.NS":"Materials",    "JSWSTEEL.NS":"Materials",
        "HINDALCO.NS":"Materials",     "VEDL.NS":"Materials",
        "SAIL.NS":"Materials",         "NMDC.NS":"Materials",
        "NATIONALUM.NS":"Materials",   "JINDALSTEL.NS":"Materials",
        "APL.NS":"Materials",          "RATNAMANI.NS":"Materials",
        # Telecom
        "BHARTIARTL.NS":"Telecom",     "IDEA.NS":"Telecom",
        # Chemicals
        "SRF.NS":"Chemicals",          "ATUL.NS":"Chemicals",
        "DEEPAKNTR.NS":"Chemicals",    "NAVINFLUOR.NS":"Chemicals",
        "FLUOROCHEM.NS":"Chemicals",   "ALKYLAMINE.NS":"Chemicals",
        "CLEAN.NS":"Chemicals",        "AAVAS.NS":"Chemicals",
        # Logistics
        "DELHIVERY.NS":"Logistics",    "BLUEDART.NS":"Logistics",
        "CONCOR.NS":"Logistics",       "MAHLOG.NS":"Logistics",
        # Media
        "ZEEL.NS":"Media",             "SUNTV.NS":"Media",
        "PVRINOX.NS":"Media",
        # Textiles
        "PAGEIND.NS":"Textiles",       "RAYMOND.NS":"Textiles",
        "ARVIND.NS":"Textiles",
    }


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_tickers_by_sector(sector: str) -> list[str]:
    """Returns all tickers in a given sector."""
    return [t for t, s in SECTOR_MAP.items() if s == sector]


def get_sector(ticker: str) -> str:
    """Returns sector for a ticker. Defaults to 'Others'."""
    return SECTOR_MAP.get(ticker, "Others")


# ── Module-level init ────────────────────────────────────────────────────────
SECTOR_MAP       = build_sector_map()
NIFTY500_TICKERS = list(SECTOR_MAP.keys())


# ── Run directly to refresh ──────────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter

    print("\n🌍 Refreshing Nifty 500 sector map...\n")
    SECTOR_MAP = build_sector_map(force_refresh=True)
    NIFTY500_TICKERS = list(SECTOR_MAP.keys())

    print(f"\n✅  Total stocks mapped : {len(NIFTY500_TICKERS)}")
    print(f"    Cache location      : {CACHE_FILE}\n")

    dist = Counter(SECTOR_MAP.values())
    print("📊  Sector Distribution:")
    print(f"    {'Sector':<18} {'Count':>6}")
    print(f"    {'─'*18} {'─'*6}")
    for sector, count in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 3)
        print(f"    {sector:<18} {count:>5}  {bar}")

    print(f"\n    Sample tickers: {NIFTY500_TICKERS[:8]}")