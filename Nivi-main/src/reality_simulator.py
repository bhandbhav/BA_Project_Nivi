"""
Nivi Oracle Engine — Reality Simulator
========================================
Layer 4: The Reality Check.

Calculates the exact "friction" cost of Indian equity delivery trades:
  - STT (Securities Transaction Tax)
  - NSE Transaction Charge
  - Stamp Duty (Buy only)
  - SEBI Fees
  - GST (on brokerage + txn charges)
  - Slippage (0.1% assumed for market orders)

Also computes the breakeven % a stock must move just to cover costs.

Ported and improved from the original reality_simulator.py.
"""

from utils import get_logger

log = get_logger("reality_simulator")

# ── Indian Govt & Exchange Fee Structure (Equity Delivery) ────────────────────
STT_RATE         = 0.001        # 0.1% on Buy & Sell
TXN_CHARGE_NSE   = 0.0000325   # 0.00325% NSE transaction fee
STAMP_DUTY       = 0.00015     # 0.015% on Buy only
SEBI_FEES        = 0.000001    # ₹10 per crore
GST_RATE         = 0.18        # 18% on (brokerage + txn charges)
BROKERAGE        = 0.0         # Assume discount broker (₹0 delivery)
SLIPPAGE         = 0.001       # 0.1% assumed market order slippage


class IndiaTradingCostModel:
    """
    Calculates exact round-trip trading friction for Indian equity delivery.
    """

    def __init__(self, capital: float = 100_000.0):
        self.capital = capital

    def calculate_trade_cost(
        self,
        price:    float,
        quantity: int,
        action:   str,    # 'BUY' or 'SELL'
    ) -> dict:
        """
        Calculates all charges for a single trade leg.

        Returns:
            execution_price: Price after slippage
            total_tax:       Sum of all charges
            breakdown:       Itemised charges dict
        """
        value = price * quantity

        stt        = value * STT_RATE
        txn_charge = value * TXN_CHARGE_NSE
        sebi_fees  = value * SEBI_FEES
        gst        = (txn_charge + BROKERAGE) * GST_RATE

        total_tax = stt + txn_charge + sebi_fees + gst

        if action == "BUY":
            stamp_duty       = value * STAMP_DUTY
            total_tax       += stamp_duty
            slippage_amt     = price * SLIPPAGE
            execution_price  = price + slippage_amt
        else:  # SELL
            stamp_duty       = 0.0
            slippage_amt     = price * SLIPPAGE
            execution_price  = price - slippage_amt

        return {
            "execution_price": round(execution_price, 2),
            "total_tax":       round(total_tax, 2),
            "breakdown": {
                "stt":          round(stt, 2),
                "txn_charge":   round(txn_charge, 2),
                "stamp_duty":   round(stamp_duty, 2),
                "sebi_fees":    round(sebi_fees, 4),
                "gst":          round(gst, 4),
                "slippage":     round(slippage_amt * quantity, 2),
            },
        }

    def calculate_round_trip(
        self,
        ticker:  str,
        weight:  float,   # Portfolio weight 0.0–1.0
        price:   float,
    ) -> dict:
        """
        Calculates full round-trip (buy + sell) friction for a position.

        Returns the cost in ₹ and as a breakeven % the stock must
        overcome just to be profitable.
        """
        position_value = self.capital * weight
        quantity       = max(1, int(position_value / price))

        buy_costs  = self.calculate_trade_cost(price, quantity, "BUY")
        sell_costs = self.calculate_trade_cost(price, quantity, "SELL")

        total_friction  = buy_costs["total_tax"] + sell_costs["total_tax"]
        breakeven_pct   = (total_friction / position_value) * 100

        return {
            "ticker":          ticker,
            "position_value":  round(position_value, 2),
            "quantity":        quantity,
            "buy_costs":       buy_costs["total_tax"],
            "sell_costs":      sell_costs["total_tax"],
            "total_friction":  round(total_friction, 2),
            "breakeven_pct":   round(breakeven_pct, 3),
        }

    def friction_report(self, allocations: dict, prices: dict) -> dict:
        """
        Generates a full friction report for a portfolio allocation.

        Args:
            allocations: {ticker: weight_pct} e.g. {"RELIANCE.NS": 15.2}
            prices:      {ticker: current_price}

        Returns:
            report dict with per-stock breakdown and total friction
        """
        results        = []
        total_friction = 0.0

        print(f"\n💸 REALITY CHECK — Transaction Cost Analysis")
        print(f"   Capital: ₹{self.capital:,.0f}")
        print("─" * 80)
        print(f"  {'TICKER':<20} {'WEIGHT':>7} {'VALUE (₹)':>12} {'FRICTION (₹)':>13} {'BREAKEVEN':>10}")
        print("─" * 80)

        for ticker, weight_pct in sorted(allocations.items(),
                                          key=lambda x: -x[1]):
            if weight_pct <= 0.01:
                continue

            weight = weight_pct / 100.0
            price  = prices.get(ticker)

            if not price:
                continue

            rt = self.calculate_round_trip(ticker, weight, price)
            total_friction += rt["total_friction"]
            results.append(rt)

            print(f"  {ticker:<20} {weight_pct:>6.1f}%  "
                  f"₹{rt['position_value']:>10,.0f}  "
                  f"₹{rt['total_friction']:>11,.2f}  "
                  f"{rt['breakeven_pct']:>8.3f}%")

        friction_pct = (total_friction / self.capital) * 100

        print("─" * 80)
        print(f"  {'TOTAL FRICTION':<20}           "
              f"₹{total_friction:>11,.2f}  ({friction_pct:.2f}% of Capital)")
        print(f"  This is the alpha you lose to the market automatically.")
        print("─" * 80 + "\n")

        return {
            "positions":       results,
            "total_friction":  round(total_friction, 2),
            "friction_pct":    round(friction_pct, 3),
        }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n⚙️  Testing reality_simulator.py...\n")

    model = IndiaTradingCostModel(capital=500_000)

    # Simulate a small portfolio
    test_allocations = {
        "RELIANCE.NS":  20.0,
        "TCS.NS":       18.0,
        "HDFCBANK.NS":  15.0,
        "INFY.NS":      12.0,
        "ICICIBANK.NS": 10.0,
    }

    test_prices = {
        "RELIANCE.NS":  2900.0,
        "TCS.NS":       3800.0,
        "HDFCBANK.NS":  1650.0,
        "INFY.NS":      1700.0,
        "ICICIBANK.NS": 1250.0,
    }

    report = model.friction_report(test_allocations, test_prices)
    print(f"  Total friction: ₹{report['total_friction']:,.2f} "
          f"({report['friction_pct']}% of capital)")

    print("\n✅ reality_simulator.py test complete.\n")
