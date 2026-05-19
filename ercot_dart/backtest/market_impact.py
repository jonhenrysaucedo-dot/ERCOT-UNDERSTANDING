"""
Market Impact Simulator for ERCOT DAM Virtual Trading.

The fundamental problem with naive backtesting of DART strategies:
  A historical backtest that simply applies DAM_SPP - RTM_SPP per MWh
  assumes our virtual trades had zero market impact — i.e., the clearing
  price was identical whether or not we participated.

This is incorrect and leads to systematically overstated P&L because:
  1. Our 10-tier offer injects additional MW into the supply stack, which
     may push the marginal unit price DOWN, reducing the DAM_SPP we receive.
  2. Our 10-tier demand bid injects additional demand, which may push the
     clearing price UP, increasing our cost of the DAM purchase.
  3. At illiquid nodes (small number of MW in the historical stack), even
     1-2 MW of virtual bids can move the clearing price by $/MWh.

The Market Impact Simulator injects our generated 10-tier curves into the
reconstructed historical supply/demand stacks and re-runs the merit-order
clearing algorithm to compute:
  - true_clearing_price : price at which the augmented market clears
  - slippage : true_clearing_price - historical_clearing_price
  - awarded_mw_per_tier : which of our 10 tiers actually cleared
  - total_awarded_mw : total MW we received
  - fill_rate : awarded_mw / target_mw

Merit-Order Clearing Algorithm
-------------------------------
Given:
  - Supply stack S = sorted ascending list of (price, MW) pairs
  - Demand stack D = sorted descending list of (price, MW) pairs
  - Our offer/bid: O = sorted list of (price, MW) pairs

The clearing algorithm:
  1. Merge S with O (for VS) or merge D with O (for VD)
  2. Compute cumulative supply MW and cumulative demand MW
  3. Find the intersection: smallest price P* where cum_supply(P*) ≥ cum_demand(P*)
  4. P* = clearing price; all supply tiers with price ≤ P* are filled

This is the same algorithm ERCOT uses in the actual DAM clearing engine
(SCED/DACF), modulo network constraints and ORDC adder computation which
are beyond the scope of the price-taking virtual trading approximation.

Slippage Model
--------------
For Virtual Supply, our MW adds to the supply stack. If our tiers are
priced below the pre-injection clearing price, we become infra-marginal
and do NOT change the clearing price. If we are at the margin, we may:
  a. Push the clearing price down (if we crowd out higher-cost offers)
  b. Clear at the same price (if we're below the existing marginal unit)
  c. Not clear (if our price > new clearing price)

Case (a) is the primary market impact risk for large virtual positions
at illiquid settlement points. The slippage can be negative (we hurt our
own clearing price) or positive (we inadvertently tighten the spread).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.config import PRICE_CAP, PRICE_FLOOR, MIN_MW
from ercot_dart.trading.order import ERCOTOrder
from ercot_dart.trading.kelly import TradeDirection
from ercot_dart.trading.tier_curve import Tier, TierCurve
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Clearing result for a single (timestamp, node) simulation
# ---------------------------------------------------------------------------

@dataclass
class ClearingResult:
    """
    Output of the market impact simulation for one order.

    Attributes
    ----------
    order_id : str
    node : str
    delivery_timestamp : pd.Timestamp
    direction : str
    historical_clearing_price : float
        The DAM clearing price from historical data (without our participation).
    true_clearing_price : float
        The clearing price AFTER injecting our 10-tier curve.
    slippage : float
        true_clearing_price - historical_clearing_price.
        Negative slippage (VS) = we lowered our own clearing price.
    awarded_mw : float
        Total MW cleared from our 10-tier curve.
    fill_rate : float
        awarded_mw / target_mw ∈ [0, 1].
    tier_awards : list[dict]
        Per-tier award: {tier_id, price, offered_mw, awarded_mw}.
    target_mw : float
    """
    order_id: str
    node: str
    delivery_timestamp: pd.Timestamp
    direction: str
    historical_clearing_price: float
    true_clearing_price: float
    slippage: float
    awarded_mw: float
    fill_rate: float
    tier_awards: list[dict]
    target_mw: float

    @property
    def effective_price(self) -> float:
        """The price at which we effectively cleared (= true_clearing_price for all filled tiers)."""
        return self.true_clearing_price

    @property
    def is_fully_filled(self) -> bool:
        return self.fill_rate >= 0.999

    @property
    def is_partially_filled(self) -> bool:
        return 0 < self.fill_rate < 0.999

    def to_series(self) -> pd.Series:
        return pd.Series({
            "order_id": self.order_id,
            "node": self.node,
            "timestamp": self.delivery_timestamp,
            "direction": self.direction,
            "historical_price": round(self.historical_clearing_price, 4),
            "true_clearing_price": round(self.true_clearing_price, 4),
            "slippage": round(self.slippage, 4),
            "awarded_mw": round(self.awarded_mw, 2),
            "target_mw": round(self.target_mw, 2),
            "fill_rate": round(self.fill_rate, 4),
        })


# ---------------------------------------------------------------------------
# Core merit-order clearing engine
# ---------------------------------------------------------------------------

class MeritOrderEngine:
    """
    Vectorised merit-order clearing algorithm.

    Clears a supply stack against a demand stack and returns the clearing
    price and cumulative MW at clearing.

    All inputs are numpy arrays for performance — this function is called
    tens of thousands of times during a full walk-forward backtest.
    """

    @staticmethod
    def clear(
        supply_prices: np.ndarray,
        supply_mws: np.ndarray,
        total_demand_mw: float,
    ) -> tuple[float, float]:
        """
        Find the clearing price for an augmented supply stack.

        Parameters
        ----------
        supply_prices : Sorted ascending array of offer prices ($/MWh).
        supply_mws : MW offered at each price step (same length as supply_prices).
        total_demand_mw : Total inelastic demand MW to clear.

        Returns
        -------
        (clearing_price, clearing_mw)
        """
        if len(supply_prices) == 0 or total_demand_mw <= 0:
            return PRICE_CAP, 0.0

        # Sort by price ascending (should already be sorted, but guard)
        order = np.argsort(supply_prices, kind="stable")
        prices = supply_prices[order]
        mws = supply_mws[order]

        cum_mw = np.cumsum(mws)
        idx = np.searchsorted(cum_mw, total_demand_mw, side="left")

        if idx >= len(prices):
            # Demand exceeds all available supply → scarcity clearing
            return float(PRICE_CAP), float(cum_mw[-1])

        return float(prices[idx]), float(cum_mw[idx])

    @staticmethod
    def inject_supply_offer(
        historical_prices: np.ndarray,
        historical_mws: np.ndarray,
        offer_prices: np.ndarray,
        offer_mws: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Merge our Virtual Supply tiers into the historical supply stack.

        Returns the combined (prices, mws) array pair, sorted ascending.
        """
        all_prices = np.concatenate([historical_prices, offer_prices])
        all_mws = np.concatenate([historical_mws, offer_mws])
        order = np.argsort(all_prices, kind="stable")
        return all_prices[order], all_mws[order]

    @staticmethod
    def inject_demand_bid(
        historical_bid_prices: np.ndarray,
        historical_bid_mws: np.ndarray,
        bid_prices: np.ndarray,
        bid_mws: np.ndarray,
    ) -> float:
        """
        Compute total demand MW after injecting our Virtual Demand bid.

        For the inelastic demand approximation, we treat all bids with
        price above the current supply clearing price as price-taking
        (they will clear regardless). We add our bid MW to total demand.

        Returns new total demand MW.
        """
        hist_demand = float(historical_bid_mws.sum())
        our_demand = float(bid_mws.sum())
        return hist_demand + our_demand


# ---------------------------------------------------------------------------
# Market Impact Simulator
# ---------------------------------------------------------------------------

class MarketImpactSimulator:
    """
    Simulates the market impact of injecting virtual orders into the
    historical DAM supply/demand stacks.

    For each ERCOTOrder, it:
      1. Retrieves the historical supply and demand stacks for that
         (timestamp, node) from the ParsedDataStore
      2. Injects the order's 10-tier curve into the appropriate stack
      3. Re-runs the merit-order clearing algorithm
      4. Computes slippage, awarded MW, and per-tier fill details

    The simulator expects the full historical offer/bid DataFrames from
    Phase 1 (ParsedDataStore.dam_offers, ParsedDataStore.dam_bids).
    """

    def __init__(
        self,
        min_stack_mw: float = 10.0,
        use_elastic_demand: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        min_stack_mw : float
            Minimum total historical stack MW required to run simulation.
            Nodes with thin stacks are flagged as unreliable.
        use_elastic_demand : bool
            If True, compute clearing against the full elastic demand stack.
            If False (default), use total bid MW as inelastic demand proxy.
            The inelastic approximation is faster and sufficient for hub nodes.
        """
        self.min_stack_mw = min_stack_mw
        self.use_elastic_demand = use_elastic_demand
        self._engine = MeritOrderEngine()

    # -----------------------------------------------------------------------
    # Stack retrieval
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_supply_stack(
        offers: pd.DataFrame,
        timestamp: pd.Timestamp,
        node: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (prices, mws) for the historical supply stack at (timestamp, node).
        Returns empty arrays if no data found.
        """
        mask = (offers["timestamp"] == timestamp) & (offers["node"] == node)
        stack = offers[mask][["price", "mw"]].sort_values("price")
        if stack.empty:
            return np.array([]), np.array([])
        return stack["price"].values.astype(np.float64), stack["mw"].values.astype(np.float64)

    @staticmethod
    def _get_demand_total(
        bids: pd.DataFrame,
        timestamp: pd.Timestamp,
        node: str,
    ) -> float:
        """
        Total bid MW at (timestamp, node) — used as inelastic demand proxy.
        """
        mask = (bids["timestamp"] == timestamp) & (bids["node"] == node)
        total = bids[mask]["mw"].sum()
        return float(total)

    # -----------------------------------------------------------------------
    # Per-tier award calculation
    # -----------------------------------------------------------------------

    def _compute_tier_awards(
        self,
        tiers: list[Tier],
        clearing_price: float,
        direction: str,
    ) -> tuple[float, list[dict]]:
        """
        Determine which tiers cleared based on the post-injection clearing price.

        For Virtual Supply: tier i clears if tier_price_i ≤ clearing_price
        For Virtual Demand: tier i clears if tier_price_i ≥ clearing_price

        Returns (total_awarded_mw, tier_award_list).
        """
        awarded_mw = 0.0
        tier_awards = []
        for t in tiers:
            if direction == TradeDirection.VIRTUAL_SUPPLY:
                clears = t.price <= clearing_price
            else:
                clears = t.price >= clearing_price

            a_mw = t.mw if clears else 0.0
            awarded_mw += a_mw
            tier_awards.append({
                "tier_id": t.tier_id,
                "price": t.price,
                "offered_mw": t.mw,
                "awarded_mw": a_mw,
                "cleared": clears,
            })
        return awarded_mw, tier_awards

    # -----------------------------------------------------------------------
    # Single order simulation
    # -----------------------------------------------------------------------

    def simulate_order(
        self,
        order: ERCOTOrder,
        offers: pd.DataFrame,
        bids: pd.DataFrame,
        historical_clearing_price: float,
    ) -> ClearingResult:
        """
        Run the market impact simulation for a single ERCOTOrder.

        Parameters
        ----------
        order : ERCOTOrder with a valid TierCurve
        offers : Full historical supply offer DataFrame from ParsedDataStore
        bids : Full historical demand bid DataFrame from ParsedDataStore
        historical_clearing_price : DAM_SPP from historical data (our baseline)

        Returns
        -------
        ClearingResult with true_clearing_price, slippage, awarded_mw
        """
        ts = order.delivery_timestamp
        node = order.node
        curve = order.tier_curve

        if curve is None:
            return ClearingResult(
                order_id=order.order_id, node=node,
                delivery_timestamp=ts, direction=order.direction,
                historical_clearing_price=historical_clearing_price,
                true_clearing_price=historical_clearing_price,
                slippage=0.0, awarded_mw=0.0, fill_rate=0.0,
                tier_awards=[], target_mw=order.target_mw,
            )

        hist_supply_prices, hist_supply_mws = self._get_supply_stack(offers, ts, node)
        demand_mw = self._get_demand_total(bids, ts, node)

        if hist_supply_mws.sum() < self.min_stack_mw or demand_mw < MIN_MW:
            logger.warning(
                "Thin stack — using historical price as clearing price",
                extra={"node": node, "ts": str(ts),
                       "supply_mw": hist_supply_mws.sum(), "demand_mw": demand_mw},
            )
            true_price = historical_clearing_price
            awarded_mw, tier_awards = self._compute_tier_awards(
                curve.tiers, true_price, order.direction
            )
            return ClearingResult(
                order_id=order.order_id, node=node,
                delivery_timestamp=ts, direction=order.direction,
                historical_clearing_price=historical_clearing_price,
                true_clearing_price=true_price,
                slippage=0.0,
                awarded_mw=awarded_mw,
                fill_rate=awarded_mw / max(order.target_mw, MIN_MW),
                tier_awards=tier_awards,
                target_mw=order.target_mw,
            )

        # Extract our curve's prices and MWs
        our_prices = np.array([t.price for t in curve.tiers], dtype=np.float64)
        our_mws = np.array([t.mw for t in curve.tiers], dtype=np.float64)

        if order.direction == TradeDirection.VIRTUAL_SUPPLY:
            # Inject into supply stack and re-clear against same demand
            aug_prices, aug_mws = self._engine.inject_supply_offer(
                hist_supply_prices, hist_supply_mws, our_prices, our_mws
            )
            true_price, _ = self._engine.clear(aug_prices, aug_mws, demand_mw)

        else:
            # Virtual Demand: add our bid MW to total demand, re-clear same supply
            aug_demand = self._engine.inject_demand_bid(
                np.array([]), np.array([]), our_prices, our_mws
            )
            new_total_demand = demand_mw + aug_demand
            true_price, _ = self._engine.clear(
                hist_supply_prices, hist_supply_mws, new_total_demand
            )

        slippage = true_price - historical_clearing_price
        awarded_mw, tier_awards = self._compute_tier_awards(
            curve.tiers, true_price, order.direction
        )
        fill_rate = awarded_mw / max(order.target_mw, MIN_MW)

        return ClearingResult(
            order_id=order.order_id,
            node=node,
            delivery_timestamp=ts,
            direction=order.direction,
            historical_clearing_price=historical_clearing_price,
            true_clearing_price=true_price,
            slippage=slippage,
            awarded_mw=awarded_mw,
            fill_rate=fill_rate,
            tier_awards=tier_awards,
            target_mw=order.target_mw,
        )

    # -----------------------------------------------------------------------
    # Batch simulation
    # -----------------------------------------------------------------------

    def simulate_all(
        self,
        orders: list[ERCOTOrder],
        offers: pd.DataFrame,
        bids: pd.DataFrame,
        dam_spp: pd.DataFrame,
    ) -> list[ClearingResult]:
        """
        Run market impact simulation for all orders from the walk-forward loop.

        Parameters
        ----------
        orders : All ERCOTOrder objects from all walk-forward folds
        offers : ParsedDataStore.dam_offers (full historical supply stack)
        bids : ParsedDataStore.dam_bids (full historical demand stack)
        dam_spp : ParsedDataStore.dam_spp (historical DAM clearing prices)

        Returns
        -------
        List of ClearingResult, one per order.
        """
        # Build a lookup dict for fast historical price retrieval
        spp_lookup: dict[tuple, float] = {}
        for _, row in dam_spp.iterrows():
            key = (row["timestamp"], row["node"])
            spp_lookup[key] = float(row["dam_spp"])

        results: list[ClearingResult] = []
        n_skipped = 0

        for order in orders:
            key = (order.delivery_timestamp, order.node)
            hist_price = spp_lookup.get(key)

            if hist_price is None:
                logger.warning(
                    "Historical DAM price not found — skipping order",
                    extra={"node": order.node, "ts": str(order.delivery_timestamp)},
                )
                n_skipped += 1
                continue

            result = self.simulate_order(order, offers, bids, hist_price)
            results.append(result)

        logger.info(
            "Market impact simulation complete",
            extra={
                "n_simulated": len(results),
                "n_skipped": n_skipped,
                "avg_slippage": round(
                    float(np.mean([r.slippage for r in results])) if results else 0, 4
                ),
                "avg_fill_rate": round(
                    float(np.mean([r.fill_rate for r in results])) if results else 0, 4
                ),
            },
        )
        return results

    # -----------------------------------------------------------------------
    # Slippage analytics
    # -----------------------------------------------------------------------

    @staticmethod
    def slippage_summary(results: list[ClearingResult]) -> pd.DataFrame:
        """
        Aggregate slippage statistics by node and direction.

        Used to identify nodes where our virtual bids have material price
        impact, which may warrant position cap reductions (Phase 5).
        """
        if not results:
            return pd.DataFrame()

        rows = [r.to_series() for r in results]
        df = pd.DataFrame(rows)

        summary = (
            df.groupby(["node", "direction"])
            .agg(
                mean_slippage=("slippage", "mean"),
                std_slippage=("slippage", "std"),
                max_slippage=("slippage", "max"),
                min_slippage=("slippage", "min"),
                avg_fill_rate=("fill_rate", "mean"),
                n_trades=("order_id", "count"),
            )
            .reset_index()
        )
        return summary
