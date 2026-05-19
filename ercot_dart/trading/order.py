"""
ERCOT-Compliant Order Objects and the Phase 3 Trading Orchestrator.

This module defines:
  1. ERCOTOrder — a validated, immutable order record suitable for ERCOT API
     submission and internal trade blotter storage.
  2. TradingOrchestrator — the Phase 3 top-level coordinator that chains
     KellySizer → TierCurveGenerator → ERCOTOrder in a single call.

ERCOT Order Lifecycle
---------------------
  PENDING  → order created, not yet submitted to DAM
  SUBMITTED → sent to ERCOT DAM API before gate closure (10:00 AM)
  AWARDED  → DAM cleared the order (full or partial)
  SETTLED  → RTM settlement complete; P&L is final
  REJECTED → ERCOT rejected the submission (compliance violation or system error)
  CANCELLED → withdrawn before gate closure

The TradingOrchestrator.run() method produces PENDING orders. Downstream
components (ERCOT API client, not part of this codebase) transition
them to SUBMITTED. The backtester (Phase 4) emulates AWARDED and SETTLED
transitions using historical data.

DART P&L Calculation
--------------------
For a SETTLED Virtual Supply order of Q MWh awarded in the DAM:
  P&L = Q × (DAM_SPP_awarded - RTM_SPP_settlement)

For a SETTLED Virtual Demand order:
  P&L = Q × (RTM_SPP_settlement - DAM_SPP_awarded)

Both the awarded price and the settlement price are populated by the
backtester or live settlement feed after delivery.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd

from ercot_dart.config import MAX_OFFER_IDS_PER_NODE, NUM_OFFER_TIERS
from ercot_dart.models.forecasting_engine import CompleteForecast, ForecastingEngine
from ercot_dart.trading.kelly import KellyResult, KellySizer, TradeDirection
from ercot_dart.trading.tier_curve import MWAllocation, TierCurve, TierCurveGenerator
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Order status enum
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    AWARDED = "AWARDED"
    PARTIALLY_AWARDED = "PARTIALLY_AWARDED"
    SETTLED = "SETTLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# ERCOT Order
# ---------------------------------------------------------------------------

@dataclass
class ERCOTOrder:
    """
    Immutable record of one ERCOT DAM virtual offer/bid submission.

    The order encapsulates the 10-tier curve plus all the metadata required
    for pre-trade compliance (Phase 5), trade blotter storage, and post-trade
    P&L attribution.

    Fields that are None at creation are populated post-clearing:
      awarded_mw, awarded_price, rtm_settlement_price, pnl
    """

    # Identifiers
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Order specification
    node: str = ""
    direction: str = TradeDirection.NO_TRADE
    delivery_timestamp: Optional[pd.Timestamp] = None
    tier_curve: Optional[TierCurve] = field(default=None, repr=False)
    target_mw: float = 0.0

    # Status lifecycle
    status: OrderStatus = OrderStatus.PENDING

    # Post-clearing fields (populated by backtester / live settlement)
    awarded_mw: Optional[float] = None
    awarded_price: Optional[float] = None         # DAM clearing price ($/MWh)
    rtm_settlement_price: Optional[float] = None  # Avg RTM price for the hour ($/MWh)
    pnl: Optional[float] = None                   # Realised P&L ($)

    # Risk metadata (from Kelly sizer)
    kelly_result: Optional[KellyResult] = field(default=None, repr=False)
    mu_forecast: float = 0.0
    sigma_forecast: float = 0.0
    prob_profit_forecast: float = 0.0
    regime_at_submission: int = 0

    # -----------------------------------------------------------------------
    # Post-clearing settlement
    # -----------------------------------------------------------------------

    def settle(
        self,
        awarded_mw: float,
        awarded_price: float,
        rtm_price: float,
    ) -> "ERCOTOrder":
        """
        Record DAM award and RTM settlement price, compute realised P&L.

        Creates and returns a new ERCOTOrder with updated fields (treating
        the dataclass as logically immutable after creation).
        """
        if self.direction == TradeDirection.VIRTUAL_SUPPLY:
            pnl = awarded_mw * (awarded_price - rtm_price)
        elif self.direction == TradeDirection.VIRTUAL_DEMAND:
            pnl = awarded_mw * (rtm_price - awarded_price)
        else:
            pnl = 0.0

        import dataclasses
        return dataclasses.replace(
            self,
            awarded_mw=awarded_mw,
            awarded_price=awarded_price,
            rtm_settlement_price=rtm_price,
            pnl=round(pnl, 4),
            status=OrderStatus.SETTLED,
        )

    def to_blotter_row(self) -> pd.Series:
        """Flatten to a single row for the trade blotter DataFrame."""
        d = {
            "order_id": self.order_id,
            "created_at": self.created_at.isoformat(),
            "node": self.node,
            "direction": self.direction,
            "delivery_timestamp": self.delivery_timestamp,
            "target_mw": self.target_mw,
            "status": self.status.value,
            "awarded_mw": self.awarded_mw,
            "awarded_price": self.awarded_price,
            "rtm_settlement_price": self.rtm_settlement_price,
            "pnl": self.pnl,
            "mu_forecast": round(self.mu_forecast, 4),
            "sigma_forecast": round(self.sigma_forecast, 4),
            "prob_profit_forecast": round(self.prob_profit_forecast, 4),
            "regime_at_submission": self.regime_at_submission,
        }
        # Flatten tier curve
        if self.tier_curve is not None:
            for t in self.tier_curve.tiers:
                d[f"MW{t.tier_id}"] = t.mw
                d[f"Price{t.tier_id}"] = t.price
        return pd.Series(d)

    def is_compliant(self) -> tuple[bool, list[str]]:
        """
        Pre-submission compliance check.
        Returns (is_valid, list_of_violations).
        """
        violations: list[str] = []
        if self.direction == TradeDirection.NO_TRADE:
            violations.append("Direction is NO_TRADE — cannot submit")
        if self.target_mw <= 0:
            violations.append(f"target_mw <= 0: {self.target_mw}")
        if self.tier_curve is not None:
            violations.extend(self.tier_curve.validate())
        return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Order blotter
# ---------------------------------------------------------------------------

class OrderBlotter:
    """
    In-memory trade blotter for the current session.

    Stores all ERCOTOrder objects and exposes aggregation methods for
    P&L attribution, node-level exposure, and position reconciliation.
    """

    def __init__(self) -> None:
        self._orders: list[ERCOTOrder] = []

    def add(self, order: ERCOTOrder) -> None:
        self._orders.append(order)

    def get_by_id(self, order_id: str) -> Optional[ERCOTOrder]:
        return next((o for o in self._orders if o.order_id == order_id), None)

    def pending_orders(self) -> list[ERCOTOrder]:
        return [o for o in self._orders if o.status == OrderStatus.PENDING]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._orders:
            return pd.DataFrame()
        return pd.DataFrame([o.to_blotter_row() for o in self._orders])

    def pnl_summary(self) -> pd.DataFrame:
        """Aggregate settled P&L by node and direction."""
        settled = [o for o in self._orders if o.status == OrderStatus.SETTLED]
        if not settled:
            return pd.DataFrame()
        df = pd.DataFrame([o.to_blotter_row() for o in settled])
        return (
            df.groupby(["node", "direction"])
            .agg(
                total_pnl=("pnl", "sum"),
                n_trades=("order_id", "count"),
                avg_awarded_mw=("awarded_mw", "mean"),
                win_rate=("pnl", lambda x: (x > 0).mean()),
            )
            .reset_index()
        )

    def node_exposure(self) -> pd.DataFrame:
        """Current MW exposure (PENDING + SUBMITTED) per node."""
        active = [
            o for o in self._orders
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
        ]
        if not active:
            return pd.DataFrame()
        df = pd.DataFrame([o.to_blotter_row() for o in active])
        return df.groupby(["node", "direction"])["target_mw"].sum().reset_index()


# ---------------------------------------------------------------------------
# Trading Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """
    Configuration for the Phase 3 TradingOrchestrator.

    Mirrors the defaults across KellySizer and TierCurveGenerator so the
    orchestrator can be configured in one place.
    """
    max_position_mw: float = 50.0
    min_position_mw: float = 1.0
    fractional_kelly: float = 0.25
    hurdle_rate: float = 0.50
    min_prob_profit: float = 0.525
    n_tiers: int = NUM_OFFER_TIERS
    mw_allocation: MWAllocation = MWAllocation.EQUAL
    min_mw_per_tier: float = 0.10


class TradingOrchestrator:
    """
    Phase 3 top-level coordinator.

    Chains the forecasting engine → Kelly sizer → tier curve generator →
    ERCOT order objects in a single call per delivery hour.

    Usage
    -----
        orchestrator = TradingOrchestrator(forecasting_engine, config)
        orders, blotter = orchestrator.run(
            feature_matrix=today_features,
            price_anchors={"HB_NORTH": 35.0, "HB_SOUTH": 38.0},
            delivery_timestamp=pd.Timestamp("2024-06-15 14:00", tz="America/Chicago"),
        )
    """

    def __init__(
        self,
        forecasting_engine: ForecastingEngine,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.engine = forecasting_engine
        self.config = config or OrchestratorConfig()

        self._kelly = KellySizer(
            max_position_mw=self.config.max_position_mw,
            min_position_mw=self.config.min_position_mw,
            fractional_kelly=self.config.fractional_kelly,
            hurdle_rate=self.config.hurdle_rate,
            min_prob_profit=self.config.min_prob_profit,
        )
        self._curve_gen = TierCurveGenerator(
            n_tiers=self.config.n_tiers,
            mw_allocation=self.config.mw_allocation,
            min_mw_per_tier=self.config.min_mw_per_tier,
        )
        self.blotter = OrderBlotter()

    def run(
        self,
        feature_matrix: pd.DataFrame,
        price_anchors: dict[str, float],
        delivery_timestamp: Optional[pd.Timestamp] = None,
        nodes: Optional[list[str]] = None,
    ) -> tuple[list[ERCOTOrder], OrderBlotter]:
        """
        Full Phase 3 pipeline for one delivery hour.

        Parameters
        ----------
        feature_matrix : Pre-gate feature rows from Phase 1 ETL.
        price_anchors : dict {node: expected_DAM_SPP} — price anchor per node.
        delivery_timestamp : Target delivery hour (defaults to last timestamp in matrix).
        nodes : Override node list; defaults to engine's target_nodes.

        Returns
        -------
        (orders, blotter)
            orders  — list of ERCOTOrder ready for pre-trade compliance and submission
            blotter — updated OrderBlotter with all new orders added
        """
        target_nodes = nodes or self.engine.config.target_nodes
        orders: list[ERCOTOrder] = []

        for node in target_nodes:
            try:
                # ---- Phase 2 → Forecast --------------------------------
                forecast = self.engine.predict(
                    feature_matrix=feature_matrix,
                    node=node,
                    delivery_timestamp=delivery_timestamp,
                )

                # ---- Kelly sizing --------------------------------------
                kelly_result = self._kelly.size(forecast)

                if not kelly_result.is_tradeable:
                    logger.info(
                        "No trade signal",
                        extra={
                            "node": node,
                            "mu": round(kelly_result.mu, 4),
                            "prob_profit": round(kelly_result.prob_profit, 4),
                        },
                    )
                    continue

                # ---- 10-Tier curve generation --------------------------
                anchor = price_anchors.get(node)
                if anchor is None:
                    logger.warning(
                        "Missing price anchor — skipping node",
                        extra={"node": node},
                    )
                    continue

                curve = self._curve_gen.generate(kelly_result, forecast, anchor)

                # ---- Order construction --------------------------------
                order = ERCOTOrder(
                    node=node,
                    direction=kelly_result.direction,
                    delivery_timestamp=kelly_result.delivery_timestamp,
                    tier_curve=curve,
                    target_mw=kelly_result.target_mw,
                    kelly_result=kelly_result,
                    mu_forecast=kelly_result.mu,
                    sigma_forecast=kelly_result.sigma,
                    prob_profit_forecast=kelly_result.prob_profit,
                    regime_at_submission=kelly_result.regime,
                )

                # ---- Pre-trade compliance check -----------------------
                is_valid, violations = order.is_compliant()
                if not is_valid:
                    logger.warning(
                        "Order failed pre-trade compliance — not added to blotter",
                        extra={"node": node, "violations": violations},
                    )
                    continue

                self.blotter.add(order)
                orders.append(order)

                logger.info(
                    "Order created",
                    extra={
                        "order_id": order.order_id[:8],
                        "node": node,
                        "direction": kelly_result.direction,
                        "target_mw": kelly_result.target_mw,
                        "n_tiers": curve.n_tiers,
                    },
                )

            except Exception as exc:
                logger.error(
                    "Order generation failed",
                    extra={"node": node, "error": str(exc)},
                )

        logger.info(
            "Trading orchestrator run complete",
            extra={
                "n_orders": len(orders),
                "total_mw": sum(o.target_mw for o in orders),
            },
        )
        return orders, self.blotter
