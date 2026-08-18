"""
Nivi Oracle Engine — Portfolio Manager
========================================
Shadow Ledger: Tracks virtual holdings, cash, and trade history.

Features:
  - Persistent JSON storage of holdings and cash
  - BUY / SELL execution with cost basis tracking (avg price)
  - Trade history logged to CSV
  - Live P&L via price injection
  - Jensen's Alpha calculation vs Nifty 50

Ported and improved from the original portfolio_manager.py.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LinearRegression

from utils import get_logger
from reality_simulator import IndiaTradingCostModel

log = get_logger("portfolio_manager")

PORTFOLIO_FILE = Path("data/static/portfolio.json")
TRADE_LOG      = Path("outputs/trade_history.csv")


class PortfolioManager:
    """
    Tracks virtual portfolio state: cash, holdings, and trade history.
    Uses IndiaTradingCostModel for realistic transaction costs.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.cost_model = IndiaTradingCostModel(initial_capital)
        self._load(initial_capital)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self, initial_capital: float):
        PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
        if PORTFOLIO_FILE.exists():
            try:
                with open(PORTFOLIO_FILE) as f:
                    self.data = json.load(f)
                log.info("Portfolio loaded from disk.")
                return
            except Exception as e:
                log.warning(f"Could not load portfolio file: {e}. Starting fresh.")

        self.data = {
            "cash":         initial_capital,
            "holdings":     {},      # {ticker: {qty, avg_price}}
            "equity_value": 0.0,
            "total_value":  initial_capital,
            "initial_capital": initial_capital,
        }
        self._save()

    def _save(self):
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.data, f, indent=4)

    # ── Valuation ─────────────────────────────────────────────────────────────

    def update_valuations(self, current_prices: dict):
        """Refreshes equity value using latest prices."""
        equity_val = 0.0
        for ticker, info in self.data["holdings"].items():
            price      = current_prices.get(ticker, info["avg_price"])
            equity_val += info["qty"] * price

        self.data["equity_value"] = round(equity_val, 2)
        self.data["total_value"]  = round(self.data["cash"] + equity_val, 2)
        self._save()

    # ── Trade Execution ───────────────────────────────────────────────────────

    def execute_trade(
        self,
        action:   str,     # 'BUY' or 'SELL'
        ticker:   str,
        price:    float,
        quantity: int,
    ) -> bool:
        """
        Executes a BUY or SELL and updates the shadow ledger.
        Transaction costs are calculated via IndiaTradingCostModel.

        Returns True if executed, False if rejected (e.g. insufficient funds).
        """
        trade_info  = self.cost_model.calculate_trade_cost(price, quantity, action)
        exec_price  = trade_info["execution_price"]
        total_cost  = price * quantity
        taxes       = trade_info["total_tax"]

        if action == "BUY":
            required = total_cost + taxes
            if self.data["cash"] < required:
                log.warning(
                    f"BUY {ticker} rejected: need ₹{required:,.0f}, "
                    f"have ₹{self.data['cash']:,.0f}"
                )
                return False

            self.data["cash"] -= required

            if ticker in self.data["holdings"]:
                old_qty   = self.data["holdings"][ticker]["qty"]
                old_cost  = old_qty * self.data["holdings"][ticker]["avg_price"]
                new_qty   = old_qty + quantity
                new_cost  = old_cost + total_cost
                self.data["holdings"][ticker] = {
                    "qty":       new_qty,
                    "avg_price": round(new_cost / new_qty, 2),
                }
            else:
                self.data["holdings"][ticker] = {
                    "qty":       quantity,
                    "avg_price": round(price, 2),
                }

        elif action == "SELL":
            if ticker not in self.data["holdings"]:
                log.warning(f"SELL {ticker} rejected: not in holdings")
                return False

            held_qty = self.data["holdings"][ticker]["qty"]
            if quantity > held_qty:
                log.warning(
                    f"SELL {ticker} rejected: want {quantity}, "
                    f"have {held_qty}"
                )
                return False

            proceeds = total_cost - taxes
            self.data["cash"] += proceeds

            self.data["holdings"][ticker]["qty"] -= quantity
            if self.data["holdings"][ticker]["qty"] <= 0:
                del self.data["holdings"][ticker]

        else:
            log.error(f"Unknown action: {action}")
            return False

        self._log_trade(ticker, action, exec_price, quantity, taxes)
        self._save()
        log.info(f"{action} {quantity}x {ticker} @ ₹{exec_price:.2f} | Tax: ₹{taxes:.2f}")
        return True

    def _log_trade(
        self,
        ticker:   str,
        action:   str,
        price:    float,
        qty:      int,
        costs:    float,
    ):
        TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = pd.DataFrame([{
            "Date":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker":   ticker,
            "Action":   action,
            "Price":    round(price, 2),
            "Qty":      qty,
            "Costs_Tax": round(costs, 2),
            "Value":    round(price * qty, 2),
        }])

        if not TRADE_LOG.exists():
            record.to_csv(TRADE_LOG, index=False)
        else:
            record.to_csv(TRADE_LOG, mode="a", header=False, index=False)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return self.data

    def print_portfolio(self, current_prices: dict = None):
        """Prints a formatted holdings table to the console."""
        self.update_valuations(current_prices or {})

        total_invested = sum(
            info["qty"] * info["avg_price"]
            for info in self.data["holdings"].values()
        )

        print("\n" + "=" * 95)
        print("  PORTFOLIO HOLDINGS")
        print("─" * 95)
        print(f"  {'TICKER':<18} | {'QTY':>6} | {'AVG COST (₹)':>12} | "
              f"{'CURRENT (₹)':>12} | {'P&L (₹)':>12} | {'P&L %':>8}")
        print("─" * 95)

        for ticker, info in self.data["holdings"].items():
            qty        = info["qty"]
            avg        = info["avg_price"]
            curr       = current_prices.get(ticker, avg) if current_prices else avg
            pl         = (curr - avg) * qty
            pl_pct     = ((curr - avg) / avg) * 100

            print(f"  {ticker:<18} | {qty:>6} | ₹{avg:>11,.2f} | "
                  f"₹{curr:>11,.2f} | ₹{pl:>+11,.2f} | {pl_pct:>+7.2f}%")

        print("─" * 95)
        total_pl     = self.data["equity_value"] - total_invested
        total_pl_pct = (total_pl / total_invested * 100) if total_invested else 0

        print(f"\n  💰 Cash:         ₹{self.data['cash']:>12,.2f}")
        print(f"  📈 Equity Value: ₹{self.data['equity_value']:>12,.2f}")
        print(f"  🏦 Total Value:  ₹{self.data['total_value']:>12,.2f}")
        print(f"  📊 Total P&L:    ₹{total_pl:>+12,.2f}  ({total_pl_pct:+.2f}%)")
        print("=" * 95 + "\n")

    def calculate_alpha(self, benchmark_ticker: str = "^NSEI") -> dict:
        """
        Calculates Jensen's Alpha and Beta vs Nifty 50 using trade history.
        Requires at least 30 trade records.
        """
        empty = {"alpha": None, "beta": None, "error": None}

        if not TRADE_LOG.exists():
            return {**empty, "error": "No trade log found"}

        try:
            import yfinance as yf

            trades = pd.read_csv(TRADE_LOG)
            if len(trades) < 30:
                return {**empty, "error": f"Need 30+ trades, have {len(trades)}"}

            trades["Date"] = pd.to_datetime(trades["Date"])

            # Approximate daily portfolio returns from trade value changes
            daily_value = trades.groupby(
                trades["Date"].dt.date
            )["Value"].sum().pct_change().dropna()

            benchmark = yf.Ticker(benchmark_ticker)
            bench_hist = benchmark.history(period="1y")["Close"]
            bench_ret  = bench_hist.pct_change().dropna() * 100

            min_len = min(len(daily_value), len(bench_ret))
            if min_len < 30:
                return {**empty, "error": "Insufficient overlapping data"}

            y = np.array(daily_value.values[-min_len:]).reshape(-1, 1)
            X = np.array(bench_ret.values[-min_len:]).reshape(-1, 1)

            model = LinearRegression().fit(X, y)
            return {
                "alpha": round(float(model.intercept_[0]), 4),
                "beta":  round(float(model.coef_[0][0]), 4),
                "error": None,
            }

        except Exception as e:
            log.error(f"Alpha calculation failed: {e}")
            return {**empty, "error": str(e)}


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n💼 Testing portfolio_manager.py...\n")

    pm = PortfolioManager(initial_capital=500_000)

    # Simulate some trades
    prices = {"RELIANCE.NS": 2900.0, "TCS.NS": 3800.0}

    print("1. Executing trades...")
    pm.execute_trade("BUY",  "RELIANCE.NS", 2900.0, 10)
    pm.execute_trade("BUY",  "TCS.NS",      3800.0, 5)
    pm.execute_trade("SELL", "RELIANCE.NS", 2950.0, 3)

    print("2. Current portfolio state:")
    pm.print_portfolio(current_prices={"RELIANCE.NS": 2970.0, "TCS.NS": 3850.0})

    print("\n✅ portfolio_manager.py test complete.\n")
