# src/utils.py
"""
Nivi Oracle Engine — Utilities
================================
Central data fetching, caching, and helper functions.
All other modules import from here.

Handles:
  - Bulk price downloads (yfinance) with local parquet cache
  - Fundamental data (yahooquery) with error handling
  - Rate limiting to avoid Yahoo blocks
  - Logging setup
"""

import os
import time
import logging
import hashlib
from pathlib import Path
# Ensure directories exist immediately
Path("logs").mkdir(exist_ok=True)
Path("data/cache").mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
from yahooquery import Ticker as YQTicker
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
CACHE_DIR   = Path(os.getenv("CACHE_DIR",  "data/cache"))
STATIC_DIR  = Path(os.getenv("STATIC_DIR", "data/static"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR","outputs"))
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")

# Cache expiry settings (in hours)
PRICE_CACHE_HOURS     = 4      # refresh intraday prices every 4h
FUNDAMENTAL_CACHE_HOURS = 168  # fundamentals once a week

# Rate limiting
REQUEST_DELAY = 0.3   # seconds between individual ticker requests
CHUNK_SIZE    = 50    # tickers per batch download

# ── Directory setup ───────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [CACHE_DIR, STATIC_DIR,
              OUTPUTS_DIR / "charts",
              OUTPUTS_DIR / "reports",
              Path("logs")]:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/nivi.log", mode="a"),
    ]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

log = get_logger("utils")


# ── Directory setup ───────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [CACHE_DIR, STATIC_DIR,
              OUTPUTS_DIR / "charts",
              OUTPUTS_DIR / "reports",
              Path("logs")]:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_path(key: str, ext: str = "parquet") -> Path:
    """Returns a deterministic cache file path for a given key."""
    safe = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{safe}_{key[:30].replace('/', '_')}.{ext}"


def _is_cache_fresh(path: Path, max_hours: float) -> bool:
    """Returns True if cache file exists and is newer than max_hours."""
    if not path.exists():
        return False
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime))
    return age < timedelta(hours=max_hours)


# ── Price fetching ────────────────────────────────────────────────────────────
def fetch_prices(
    tickers:       list[str],
    period:        str  = "2y",
    interval:      str  = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Downloads adjusted close prices for a list of tickers.
    Caches result as parquet. Returns a wide DataFrame (date × ticker).

    Args:
        tickers:       List of Yahoo Finance ticker strings e.g. ['TCS.NS']
        period:        yfinance period string: '1y', '2y', '5y', '10y', 'max'
        interval:      '1d' for daily (default), '1wk', '1mo'
        force_refresh: Bypass cache and re-download

    Returns:
        pd.DataFrame with DatetimeIndex and tickers as columns.
        Missing tickers are silently dropped.
    """
    cache_key  = f"prices_{period}_{interval}_{len(tickers)}"
    cache_file = _cache_path(cache_key)

    if not force_refresh and _is_cache_fresh(cache_file, PRICE_CACHE_HOURS):
        log.info(f"Loading prices from cache ({cache_file.name})")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            log.warning(f"Cache read failed: {e} — re-downloading")

    log.info(f"Downloading prices for {len(tickers)} tickers "
             f"(period={period}, interval={interval})...")

    all_chunks = []
    failed     = []

    # Download in chunks to avoid rate limits
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        try:
            raw = yf.download(
                chunk,
                period      = period,
                interval    = interval,
                auto_adjust = True,
                progress    = False,
                threads     = True,
            )

            if raw.empty:
                log.warning(f"Chunk {i//CHUNK_SIZE + 1}: empty response")
                failed.extend(chunk)
                continue

            # Extract Close prices cleanly
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                # Single ticker — reshape
                prices = raw[["Close"]] if "Close" in raw.columns else raw
                if len(chunk) == 1:
                    prices.columns = chunk

            all_chunks.append(prices)
            log.info(f"  Chunk {i//CHUNK_SIZE + 1}/{(len(tickers)-1)//CHUNK_SIZE + 1} "
                     f"— {len(prices.columns)} tickers loaded")

        except Exception as e:
            log.error(f"Chunk {i//CHUNK_SIZE + 1} failed: {e}")
            failed.extend(chunk)

        time.sleep(REQUEST_DELAY)

    if not all_chunks:
        log.error("No price data fetched at all.")
        return pd.DataFrame()

    # Combine all chunks
    combined = pd.concat(all_chunks, axis=1)

    # Remove duplicate columns (can happen with multi-index edge cases)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Strip timezone
    if combined.index.tz is not None:
        combined.index = combined.index.tz_convert(None)

    # Forward fill gaps (weekends, holidays) then drop fully empty rows
    combined = combined.ffill().dropna(how="all")

    if failed:
        log.warning(f"Failed tickers ({len(failed)}): {failed[:10]}...")

    log.info(f"Price data ready: {combined.shape[0]} days × "
             f"{combined.shape[1]} tickers")

    # Cache
    try:
        combined.to_parquet(cache_file)
        log.info(f"Prices cached → {cache_file.name}")
    except Exception as e:
        log.warning(f"Could not cache prices: {e}")

    return combined


def fetch_prices_single(
    ticker: str,
    period: str = "2y",
) -> pd.Series:
    """
    Convenience wrapper for a single ticker.
    Returns a price Series with DatetimeIndex.
    """
    df = fetch_prices([ticker], period=period)
    if ticker in df.columns:
        return df[ticker].dropna()
    return pd.Series(dtype=float)


# ── Fundamental data (yahooquery) ─────────────────────────────────────────────
def fetch_fundamentals(
    tickers:       list[str],
    force_refresh: bool = False,
) -> dict:
    """
    Fetches fundamental data using yahooquery.
    Returns a dict keyed by ticker with sub-keys:
      'income_statement', 'balance_sheet', 'cash_flow', 'key_stats', 'info'

    Caches as parquet files per ticker.
    Uses yahooquery for better fundamental coverage than yfinance.
    Falls back to empty dict on failure.
    """
    results = {}

    for ticker in tickers:
        cache_file = _cache_path(f"fundamentals_{ticker}", "json")

        if not force_refresh and _is_cache_fresh(cache_file, FUNDAMENTAL_CACHE_HOURS):
            try:
                import json
                with open(cache_file) as f:
                    results[ticker] = json.load(f)
                continue
            except Exception:
                pass

        try:
            yq = YQTicker(ticker)

            data = {
                "income_statement": _safe_df_to_dict(
                    yq.income_statement(frequency="annual")
                ),
                "balance_sheet": _safe_df_to_dict(
                    yq.balance_sheet(frequency="annual")
                ),
                "cash_flow": _safe_df_to_dict(
                    yq.cash_flow(frequency="annual")
                ),
                "key_stats":    yq.key_stats.get(ticker, {}),
                "info":         yq.price.get(ticker, {}),
                "fetched_at":   datetime.now().isoformat(),
            }

            results[ticker] = data

            # Cache
            import json
            with open(cache_file, "w") as f:
                json.dump(data, f, default=str)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            log.warning(f"Fundamental fetch failed for {ticker}: {e}")
            results[ticker] = {}

    return results


def fetch_fundamentals_single(ticker: str) -> dict:
    """Convenience wrapper for single ticker fundamentals."""
    result = fetch_fundamentals([ticker])
    return result.get(ticker, {})


# ── News fetching (for VADER sentiment) ───────────────────────────────────────
def fetch_news(ticker: str, max_articles: int = 15) -> list[dict]:
    """
    Fetches recent news headlines via yfinance.
    Returns list of {title, publisher, link}.
    """
    try:
        stock     = yf.Ticker(ticker)
        news_list = stock.news or []
        results   = []

        for article in news_list[:max_articles]:
            content = article.get("content", {})
            title   = (
                content.get("title") or
                article.get("title") or
                ""
            )
            if title:
                results.append({
                    "title":     title,
                    "publisher": content.get("provider", {}).get("displayName", ""),
                    "link":      content.get("canonicalUrl", {}).get("url", ""),
                })

        return results

    except Exception as e:
        log.warning(f"News fetch failed for {ticker}: {e}")
        return []


# ── Market data helpers ───────────────────────────────────────────────────────
def get_latest_price(ticker: str) -> Optional[float]:
    """Returns the most recent closing price for a ticker."""
    try:
        data = yf.download(
            ticker, period="2d",
            auto_adjust=True, progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"Could not get latest price for {ticker}: {e}")
    return None


def get_market_regime(benchmark: str = "^NSEI") -> dict:
    """
    Returns current market regime based on SMA-200.
    Used as a global gate by the Oracle Engine.
    """
    try:
        prices = fetch_prices_single(benchmark, period="2y")
        if len(prices) < 50:
            return {"regime": "UNKNOWN", "is_bull": True}

        window  = min(200, len(prices) - 1)
        sma200  = prices.rolling(window).mean().iloc[-1]
        current = prices.iloc[-1]
        is_bull = bool(current > sma200)

        pct_above = ((current - sma200) / sma200) * 100

        return {
            "regime":     "BULL" if is_bull else "BEAR",
            "is_bull":    is_bull,
            "nifty":      round(float(current), 2),
            "sma200":     round(float(sma200), 2),
            "pct_above":  round(float(pct_above), 2),
        }
    except Exception as e:
        log.error(f"Market regime check failed: {e}")
        return {"regime": "UNKNOWN", "is_bull": True}


# ── Generic helpers ───────────────────────────────────────────────────────────
def _safe_df_to_dict(df) -> list[dict]:
    """Converts a DataFrame to a JSON-serialisable list of dicts."""
    try:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return []
        return df.reset_index().to_dict(orient="records")
    except Exception:
        return []


def pct_change_safe(series: pd.Series, periods: int = 1) -> pd.Series:
    """pct_change with division-by-zero protection."""
    shifted = series.shift(periods)
    return series.where(shifted == 0, (series - shifted) / shifted.abs())


def annualised_return(series: pd.Series) -> float:
    """Annualised return from a price series."""
    if len(series) < 2:
        return 0.0
    days = (series.index[-1] - series.index[0]).days
    if days <= 0:
        return 0.0
    return float(((series.iloc[-1] / series.iloc[0]) ** (365 / days)) - 1)


def max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown as a negative fraction."""
    if len(series) < 2:
        return 0.0
    peak = series.cummax()
    dd   = (series - peak) / peak
    return float(dd.min())


def sortino_ratio(
    returns:   pd.Series,
    risk_free: float = 0.072,
) -> float:
    """Annualised Sortino ratio from daily return series."""
    if len(returns) < 30:
        return 0.0
    ann_ret   = returns.mean() * 252
    down_std  = returns[returns < 0].std() * np.sqrt(252)
    if down_std == 0:
        return 0.0
    return float((ann_ret - risk_free) / down_std)


def rolling_beta(
    stock_returns: pd.Series,
    bench_returns: pd.Series,
    window:        int = 60,
) -> pd.Series:
    """Rolling beta of stock vs benchmark."""
    cov  = stock_returns.rolling(window).cov(bench_returns)
    var  = bench_returns.rolling(window).var()
    return cov / var.replace(0, np.nan)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔧 Testing utils.py...\n")

    # 1. Test price fetch (small sample)
    print("1. Fetching prices for 3 stocks...")
    prices = fetch_prices(["RELIANCE.NS", "TCS.NS", "INFY.NS"], period="1mo")
    print(f"   ✅ Shape: {prices.shape}")
    print(f"   Columns: {list(prices.columns)}")
    print(f"   Latest:\n{prices.tail(2)}\n")

    # 2. Test market regime
    print("2. Checking market regime...")
    regime = get_market_regime()
    print(f"   ✅ Nifty 50: {regime}\n")

    # 3. Test news fetch
    print("3. Fetching news for TCS...")
    news = fetch_news("TCS.NS", max_articles=3)
    for n in news:
        print(f"   📰 {n['title'][:70]}")

    print("\n✅ utils.py working correctly.\n")
