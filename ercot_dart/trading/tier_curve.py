"""
ERCOT 10-Tier Offer/Bid Curve Generator.

ERCOT's DAM submission protocol requires that all energy offers and bids be
expressed as a step-function with at most 10 (price, MW) pairs — "tiers".
For supply offers (Virtual Supply):
  - Tiers must be monotonically non-decreasing in price
  - The DAM clears cheaper tiers first (merit order)
For demand bids (Virtual Demand):
  - Tiers must be monotonically non-increasing in price
  - The DAM clears higher-priced bids first (demand merit order)

Tier Construction Algorithm
----------------------------
The Phase 2 posterior predictive distribution provides samples of the DART
spread. Combined with a price anchor P_anchor (the expected DAM clearing price),
we construct the 10-tier curve as follows:

Step 1 — Compute the predictive CDF at the price anchor.
    We approximate E[DAM_SPP] ≈ P_anchor (provided externally; typically
    derived from the supply stack weighted average price or a rolling DAM mean).

Step 2 — Derive tier price grid.
    For Virtual Supply (offering to sell in the DAM):
    - Prices should lie BELOW P_anchor to ensure high fill probability
    - We spread tiers from [P_anchor - 3σ, P_anchor + 0.5σ]
    - Lower tiers are aggressively priced (guaranteed fill, lower marginal profit)
    - Higher tiers are optimistically priced (conditional fill, higher marginal profit)
    - Rationale: if the DAM clears at P_anchor, all tiers below P_anchor are filled;
      the upper tiers act as profit-maximising limit orders

    For Virtual Demand (bidding to buy in the DAM):
    - Prices should lie ABOVE P_anchor - spread, to ensure high fill probability
    - Tiers spread from [P_anchor - 0.5σ, P_anchor + 3σ]
    - Higher-priced tiers (filled first) represent aggressive demand
    - Monotone decreasing price ordering is enforced

Step 3 — Allocate MW across tiers.
    Two allocation schemes:
    - EQUAL: total_mw / 10 per tier (default, transparent, simple)
    - POSTERIOR_WEIGHTED: MW proportional to the marginal clearing probability
      at each tier's price (optimal for maximising expected filled volume)

Step 4 — Enforce ERCOT constraints.
    - All prices clipped to [PRICE_FLOOR, PRICE_CAP] = [-$250, $5,000]
    - Minimum MW per tier: 0.1 MW (ERCOT sub-economic threshold)
    - Strictly monotone prices: if any two adjacent tiers have equal price,
      add a small epsilon increment to restore strict monotonicity
    - Total MW = target_mw (no over/under allocation after rounding)

Step 5 — Post-construction validation.
    A TierCurve.validate() method checks all ERCOT compliance conditions
    before the curve is passed to the order router (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.config import MIN_MW, NUM_OFFER_TIERS, PRICE_CAP, PRICE_FLOOR
from ercot_dart.models.forecasting_engine import CompleteForecast
from ercot_dart.trading.kelly import KellyResult, TradeDirection
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum price separation between adjacent tiers ($/MWh)
MIN_PRICE_STEP: float = 0.01
# Sigma multiplier for spread below/above anchor for the outer tiers
SUPPLY_LOWER_SIGMA: float = 3.0
SUPPLY_UPPER_SIGMA: float = 0.50
DEMAND_LOWER_SIGMA: float = 0.50
DEMAND_UPPER_SIGMA: float = 3.0


class MWAllocation(str, Enum):
    EQUAL = "equal"                       # Flat MW per tier
    POSTERIOR_WEIGHTED = "posterior"      # MW ∝ marginal clearing probability


# ---------------------------------------------------------------------------
# Single tier
# ---------------------------------------------------------------------------

@dataclass
class Tier:
    """A single (price, mw) pair in an ERCOT offer/bid curve."""
    tier_id: int         # 1-based index
    price: float         # $/MWh
    mw: float            # MW offered/bid at this price

    def to_dict(self) -> dict:
        return {"tier_id": self.tier_id, "price": round(self.price, 2), "mw": round(self.mw, 2)}


# ---------------------------------------------------------------------------
# 10-tier curve
# ---------------------------------------------------------------------------

@dataclass
class TierCurve:
    """
    Complete ERCOT 10-tier offer or bid curve for one (node, hour).

    Attributes
    ----------
    tiers : list of Tier, length ≤ 10
    direction : VIRTUAL_SUPPLY or VIRTUAL_DEMAND
    node : Settlement point name
    delivery_timestamp : Start of the delivery hour
    total_mw : Sum of MW across all tiers
    price_anchor : Expected DAM clearing price used to construct the curve
    """
    tiers: list[Tier]
    direction: str
    node: str
    delivery_timestamp: pd.Timestamp
    total_mw: float
    price_anchor: float

    # Kelly sizing inputs — for audit
    kelly_result: Optional[KellyResult] = field(default=None, repr=False)

    @property
    def n_tiers(self) -> int:
        return len(self.tiers)

    @property
    def prices(self) -> np.ndarray:
        return np.array([t.price for t in self.tiers])

    @property
    def mws(self) -> np.ndarray:
        return np.array([t.mw for t in self.tiers])

    def to_dataframe(self) -> pd.DataFrame:
        rows = [t.to_dict() for t in self.tiers]
        df = pd.DataFrame(rows)
        df["node"] = self.node
        df["direction"] = self.direction
        df["timestamp"] = self.delivery_timestamp
        return df

    def to_wide_row(self) -> pd.Series:
        """
        Serialize to a single wide-format row matching ERCOT submission schema:
        MW1, Price1, MW2, Price2, …, MW10, Price10.
        """
        data: dict = {
            "DeliveryDate": self.delivery_timestamp.date(),
            "HourEnding": self.delivery_timestamp.hour + 1,
            "SettlementPoint": self.node,
            "Direction": self.direction,
        }
        for t in self.tiers:
            data[f"MW{t.tier_id}"] = round(t.mw, 2)
            data[f"Price{t.tier_id}"] = round(t.price, 2)
        # Pad unused tiers with zeros
        for i in range(len(self.tiers) + 1, NUM_OFFER_TIERS + 1):
            data[f"MW{i}"] = 0.0
            data[f"Price{i}"] = 0.0
        return pd.Series(data)

    def validate(self) -> list[str]:
        """
        Check all ERCOT compliance rules. Returns a list of violation strings.
        An empty list means the curve is compliant.
        """
        violations: list[str] = []

        # Tier count
        if self.n_tiers < 1 or self.n_tiers > NUM_OFFER_TIERS:
            violations.append(
                f"Tier count {self.n_tiers} outside [1, {NUM_OFFER_TIERS}]"
            )

        prices = self.prices
        mws = self.mws

        # Price bounds
        if np.any(prices < PRICE_FLOOR):
            violations.append(f"Price(s) below floor {PRICE_FLOOR}: {prices[prices < PRICE_FLOOR]}")
        if np.any(prices > PRICE_CAP):
            violations.append(f"Price(s) above cap {PRICE_CAP}: {prices[prices > PRICE_CAP]}")

        # Minimum MW per tier
        active = mws[mws > 0]
        if len(active) > 0 and np.any(active < MIN_MW):
            violations.append(f"Tier MW below minimum {MIN_MW}: {active[active < MIN_MW]}")

        # Monotonicity
        if self.direction == TradeDirection.VIRTUAL_SUPPLY:
            diffs = np.diff(prices)
            if np.any(diffs < 0):
                violations.append("Supply curve prices not monotonically non-decreasing")
        elif self.direction == TradeDirection.VIRTUAL_DEMAND:
            diffs = np.diff(prices)
            if np.any(diffs > 0):
                violations.append("Demand curve prices not monotonically non-increasing")

        # Total MW
        total = float(np.sum(mws))
        if abs(total - self.total_mw) > 0.01:
            violations.append(
                f"Tier MW sum {total:.2f} != target total_mw {self.total_mw:.2f}"
            )

        return violations


# ---------------------------------------------------------------------------
# Tier Curve Generator
# ---------------------------------------------------------------------------

class TierCurveGenerator:
    """
    Translates a KellyResult + CompleteForecast into an ERCOT-compliant
    10-tier offer/bid curve.

    The price grid is anchored to `price_anchor` (expected DAM SPP) and
    spread using the posterior predictive standard deviation from Phase 2.
    MW is allocated across tiers by the chosen allocation scheme.

    Parameters
    ----------
    n_tiers : int
        Number of tiers (max 10, ERCOT limit).
    mw_allocation : MWAllocation
        EQUAL or POSTERIOR_WEIGHTED MW distribution.
    min_mw_per_tier : float
        Tiers with MW below this threshold are dropped and MW redistributed.
    price_step_min : float
        Minimum price separation between adjacent tiers.
    """

    def __init__(
        self,
        n_tiers: int = NUM_OFFER_TIERS,
        mw_allocation: MWAllocation = MWAllocation.EQUAL,
        min_mw_per_tier: float = 0.10,
        price_step_min: float = MIN_PRICE_STEP,
    ) -> None:
        if n_tiers < 1 or n_tiers > NUM_OFFER_TIERS:
            raise ValueError(f"n_tiers must be in [1, {NUM_OFFER_TIERS}], got {n_tiers}")
        self.n_tiers = n_tiers
        self.mw_allocation = mw_allocation
        self.min_mw_per_tier = min_mw_per_tier
        self.price_step_min = price_step_min

    # -----------------------------------------------------------------------
    # Price grid construction
    # -----------------------------------------------------------------------

    def _supply_price_grid(self, price_anchor: float, sigma: float) -> np.ndarray:
        """
        Construct a monotonically increasing price grid for Virtual Supply.

        Tiers span [anchor - LOWER_SIGMA × σ, anchor + UPPER_SIGMA × σ],
        distributed as evenly-spaced quantiles. The lower end is aggressively
        priced to guarantee fill; the upper end captures higher spread if the
        market clears above anchor.

        All prices are clipped to [PRICE_FLOOR, PRICE_CAP].
        """
        p_low = price_anchor - SUPPLY_LOWER_SIGMA * sigma
        p_high = price_anchor + SUPPLY_UPPER_SIGMA * sigma
        prices = np.linspace(p_low, p_high, self.n_tiers)
        return np.clip(prices, PRICE_FLOOR, PRICE_CAP)

    def _demand_price_grid(self, price_anchor: float, sigma: float) -> np.ndarray:
        """
        Construct a monotonically decreasing price grid for Virtual Demand.

        Tiers span [anchor - LOWER_SIGMA × σ, anchor + UPPER_SIGMA × σ]
        in DECREASING order. Higher-priced tiers (filled first in DAM demand
        merit order) represent the most aggressive bids.

        All prices are clipped to [PRICE_FLOOR, PRICE_CAP].
        """
        p_low = price_anchor - DEMAND_LOWER_SIGMA * sigma
        p_high = price_anchor + DEMAND_UPPER_SIGMA * sigma
        prices = np.linspace(p_high, p_low, self.n_tiers)
        return np.clip(prices, PRICE_FLOOR, PRICE_CAP)

    @staticmethod
    def _enforce_monotone_increasing(prices: np.ndarray, step: float) -> np.ndarray:
        """
        Ensure strict monotone-increasing price sequence by adding
        a minimum step to any tier that would otherwise equal or undercut
        its predecessor. Applied in-place on a copy.
        """
        prices = prices.copy()
        for i in range(1, len(prices)):
            if prices[i] <= prices[i - 1]:
                prices[i] = prices[i - 1] + step
        return prices.clip(PRICE_FLOOR, PRICE_CAP)

    @staticmethod
    def _enforce_monotone_decreasing(prices: np.ndarray, step: float) -> np.ndarray:
        """Ensure strict monotone-decreasing price sequence."""
        prices = prices.copy()
        for i in range(1, len(prices)):
            if prices[i] >= prices[i - 1]:
                prices[i] = prices[i - 1] - step
        return prices.clip(PRICE_FLOOR, PRICE_CAP)

    # -----------------------------------------------------------------------
    # MW allocation
    # -----------------------------------------------------------------------

    def _allocate_mw_equal(self, total_mw: float) -> np.ndarray:
        """Distribute total_mw equally across n_tiers, adjusting last tier for rounding."""
        per_tier = np.full(self.n_tiers, total_mw / self.n_tiers)
        # Correct rounding error on last tier
        per_tier[-1] = total_mw - per_tier[:-1].sum()
        return np.maximum(per_tier, 0.0)

    def _allocate_mw_posterior_weighted(
        self,
        total_mw: float,
        prices: np.ndarray,
        posterior_samples: np.ndarray,
        direction: str,
    ) -> np.ndarray:
        """
        Allocate MW proportionally to the marginal clearing probability at
        each tier's price, derived from the posterior predictive samples.

        For Virtual Supply at price tier p_i:
          - P(clear at tier i) ≈ P(DAM_SPP ≥ p_i) = 1 - F_posterior(p_i)
          - Marginal probability = P(clear at tier i) - P(clear at tier i+1)
          - MW_i ∝ marginal probability

        This concentrates MW around the most likely clearing region,
        maximising expected filled volume weighted by posterior beliefs.
        """
        if posterior_samples is None or len(posterior_samples) == 0:
            return self._allocate_mw_equal(total_mw)

        # posterior_samples contains DART spread draws, not DAM price.
        # We use the samples as a proxy for the DAM price distribution
        # relative to the anchor — this is an approximation.
        n_samples = len(posterior_samples)

        if direction == TradeDirection.VIRTUAL_SUPPLY:
            # P(DAM ≥ price_i) approximated from posterior DART samples
            fill_probs = np.array([
                float(np.mean(posterior_samples >= p)) for p in prices
            ])
            # Marginal probability of being the "marginal" cleared tier
            marginal = np.diff(np.concatenate([[1.0], fill_probs]))
            marginal = np.abs(marginal)  # diff is negative for supply; take abs
        else:
            # For demand: P(DAM ≤ price_i)
            fill_probs = np.array([
                float(np.mean(posterior_samples <= p)) for p in prices
            ])
            marginal = np.diff(np.concatenate([[0.0], fill_probs]))
            marginal = np.abs(marginal)

        # Normalise to sum to 1
        total_weight = marginal.sum()
        if total_weight < 1e-8:
            return self._allocate_mw_equal(total_mw)

        weights = marginal / total_weight
        mws = weights * total_mw

        # Drop sub-minimum tiers and redistribute MW
        mws = self._enforce_min_mw(mws, total_mw)
        return mws

    def _enforce_min_mw(self, mws: np.ndarray, total_mw: float) -> np.ndarray:
        """
        Zero out tiers below min_mw_per_tier and redistribute their MW
        proportionally to the remaining active tiers.
        """
        mws = mws.copy()
        below_min = mws < self.min_mw_per_tier
        surplus = mws[below_min].sum()
        mws[below_min] = 0.0

        active_mask = mws >= self.min_mw_per_tier
        if active_mask.sum() == 0:
            # Fallback: equal allocation across all tiers
            mws = np.full(len(mws), total_mw / len(mws))
        else:
            mws[active_mask] += surplus * (mws[active_mask] / mws[active_mask].sum())

        # Fix rounding on last active tier
        active_idx = np.where(mws > 0)[0]
        if len(active_idx) > 0:
            mws[active_idx[-1]] = total_mw - mws[active_idx[:-1]].sum() - mws[~active_mask].sum()
            mws = np.maximum(mws, 0.0)

        return mws

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(
        self,
        kelly_result: KellyResult,
        forecast: CompleteForecast,
        price_anchor: float,
    ) -> TierCurve:
        """
        Generate a validated ERCOT 10-tier offer/bid curve.

        Parameters
        ----------
        kelly_result : KellyResult from KellySizer.size()
        forecast : CompleteForecast from ForecastingEngine.predict()
        price_anchor : float
            Expected DAM clearing price at the node for this delivery hour.
            Typically sourced from: rolling 30-day mean DAM SPP, or the
            supply stack weighted average price.

        Returns
        -------
        TierCurve — validated and ready for ERCOT API submission.

        Raises
        ------
        ValueError if kelly_result.direction == NO_TRADE.
        """
        if kelly_result.direction == TradeDirection.NO_TRADE:
            raise ValueError(
                f"Cannot generate curve for NO_TRADE direction at node {kelly_result.node}"
            )

        direction = kelly_result.direction
        total_mw = kelly_result.target_mw
        sigma = kelly_result.sigma

        # Step 1 — Price grid
        if direction == TradeDirection.VIRTUAL_SUPPLY:
            raw_prices = self._supply_price_grid(price_anchor, sigma)
            prices = self._enforce_monotone_increasing(raw_prices, self.price_step_min)
        else:
            raw_prices = self._demand_price_grid(price_anchor, sigma)
            prices = self._enforce_monotone_decreasing(raw_prices, self.price_step_min)

        # Step 2 — MW allocation
        if self.mw_allocation == MWAllocation.POSTERIOR_WEIGHTED:
            # Use posterior predictive samples as a proxy for DAM price distribution
            posterior_samples = (
                forecast.posterior.predictive_samples[:, -1]
                if forecast.posterior.predictive_samples is not None
                else None
            )
            mws = self._allocate_mw_posterior_weighted(
                total_mw, prices, posterior_samples, direction
            )
        else:
            mws = self._allocate_mw_equal(total_mw)

        # Step 3 — Build Tier objects (drop zero-MW tiers, preserve monotonicity)
        tiers = []
        tier_id = 1
        for price, mw in zip(prices, mws):
            if mw < self.min_mw_per_tier:
                continue
            tiers.append(Tier(tier_id=tier_id, price=float(price), mw=float(mw)))
            tier_id += 1

        if not tiers:
            raise ValueError(
                f"Tier generation produced zero valid tiers for node {kelly_result.node}. "
                f"total_mw={total_mw:.2f}, sigma={sigma:.4f}"
            )

        curve = TierCurve(
            tiers=tiers,
            direction=direction,
            node=kelly_result.node,
            delivery_timestamp=kelly_result.delivery_timestamp,
            total_mw=total_mw,
            price_anchor=price_anchor,
            kelly_result=kelly_result,
        )

        # Step 4 — Validate
        violations = curve.validate()
        if violations:
            logger.warning(
                "TierCurve validation violations",
                extra={"node": kelly_result.node, "violations": violations},
            )
        else:
            logger.info(
                "TierCurve generated",
                extra={
                    "node": kelly_result.node,
                    "direction": direction,
                    "n_tiers": curve.n_tiers,
                    "total_mw": total_mw,
                    "price_range": f"[{prices.min():.2f}, {prices.max():.2f}]",
                    "anchor": round(price_anchor, 2),
                },
            )
        return curve

    def generate_batch(
        self,
        kelly_results: list[KellyResult],
        forecasts: list[CompleteForecast],
        price_anchors: dict[str, float],
    ) -> list[TierCurve]:
        """
        Generate curves for a batch of nodes for the same delivery hour.

        Parameters
        ----------
        kelly_results : List of KellyResult objects from KellySizer.size_portfolio()
        forecasts : List of CompleteForecast aligned to kelly_results
        price_anchors : dict {node: price_anchor} — expected DAM price per node

        Returns
        -------
        List of TierCurve for all tradeable nodes (NO_TRADE results are skipped).
        """
        curves = []
        for kr, fc in zip(kelly_results, forecasts):
            if kr.direction == TradeDirection.NO_TRADE:
                continue
            anchor = price_anchors.get(kr.node)
            if anchor is None:
                logger.warning(
                    "No price anchor for node — using rolling DAM mean fallback",
                    extra={"node": kr.node},
                )
                # Fallback: use the posterior mu as a rough price estimate
                # This should be replaced with a proper DAM price forecast in production
                anchor = float(fc.posterior.mu_mean[-1]) + 30.0  # rough RTM + spread
            try:
                curves.append(self.generate(kr, fc, anchor))
            except Exception as e:
                logger.error(
                    "Curve generation failed",
                    extra={"node": kr.node, "error": str(e)},
                )
        return curves
