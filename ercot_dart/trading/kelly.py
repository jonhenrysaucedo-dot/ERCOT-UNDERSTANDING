"""
Bayesian Kelly Criterion Position Sizer for ERCOT DART Virtual Trading.

The standard continuous Kelly formula for a normally distributed return is:

    f* = (μ - r) / σ²

where:
    μ  = E[DART spread]    posterior mean from Phase 2 Bayesian model ($/MWh)
    r  = hurdle rate       minimum acceptable spread to deploy capital ($/MWh)
    σ² = Var[DART spread]  posterior predictive variance from MS-GARCH + MCMC

f* is the fraction of maximum position to deploy. It is then scaled by:
  1. A fractional Kelly multiplier k ∈ (0, 1]  — default quarter-Kelly (k=0.25)
  2. A HDI-width penalty  — fat posteriors (high tail uncertainty) reduce size
  3. A regime penalty     — Scarcity regime carries higher gap risk
  4. A P(profit) penalty  — linear taper below a probability threshold
  5. A credible interval asymmetry penalty — lopsided HDI implies model uncertainty

The resulting adjusted fraction f_adj ∈ [0, 1] is multiplied by max_position_mw
to produce the target MW volume, which is then clamped to [min_mw, max_mw].

Mathematical Derivation
-----------------------
Under the Kelly criterion, f* maximises the expected logarithmic growth:

    E[log(1 + f·R)]

For R ~ Normal(μ, σ²), the first-order condition gives f* = (μ - r) / σ².

The Fractional Kelly adjustment replaces f* with k·f* where k < 1. This is
equivalent to a Bayesian investor with Dirichlet-process uncertainty over the
true distribution, converging to Kelly only when the model is perfectly
specified. In practice, quarter-Kelly (k=0.25) is the market standard for
live trading with model uncertainty.

The MCMC posterior credible interval (HDI) provides a natural penalty signal:
if the 95% HDI is much wider than 4σ (the Gaussian reference), the posterior
has heavier tails than assumed, and the theoretical Kelly fraction overstates
the true edge. We penalise by the ratio 4σ / hdi_width, capped at 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.config import MAX_OFFER_IDS_PER_NODE, MIN_MW
from ercot_dart.models.forecasting_engine import CompleteForecast
from ercot_dart.models.hmm import REGIME_LABELS
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Trade direction enum
# ---------------------------------------------------------------------------

class TradeDirection:
    VIRTUAL_SUPPLY: str = "VS"    # Sell DAM, buy RTM — profit when DAM > RTM
    VIRTUAL_DEMAND: str = "VD"    # Buy DAM, sell RTM — profit when RTM > DAM
    NO_TRADE: str = "NO_TRADE"


# ---------------------------------------------------------------------------
# Regime penalty table
# ---------------------------------------------------------------------------

# Regime-conditional Kelly multipliers.
# Scarcity: ORDC adder creates convex payoff but also gap risk — reduce size.
# NegCong: sustained negative prices require directional confidence — moderate reduction.
_REGIME_KELLY_MULTIPLIER: dict[int, float] = {
    0: 1.00,   # Normal
    1: 0.70,   # Scarcity
    2: 0.80,   # Negative Congestion
}


# ---------------------------------------------------------------------------
# Kelly result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KellyResult:
    """
    Complete output of the Kelly sizing calculation.

    All penalty factors are preserved for audit logging and
    compliance reporting (Phase 5).

    Attributes
    ----------
    direction : str
        TradeDirection.VIRTUAL_SUPPLY or VIRTUAL_DEMAND or NO_TRADE.
    raw_kelly_fraction : float
        f* = (μ - r) / σ²  before any adjustments.
    fractional_kelly : float
        f* × k  after the fractional Kelly multiplier.
    adjusted_fraction : float
        Final fraction after all penalties. Clamped to [0, 1].
    target_mw : float
        adjusted_fraction × max_position_mw, clamped to [min_mw, max_mw].
    penalty_hdi : float        HDI width penalty (≤ 1.0)
    penalty_regime : float     Regime multiplier (≤ 1.0)
    penalty_prob : float       P(profit) taper multiplier (≤ 1.0)
    penalty_asymmetry : float  HDI asymmetry multiplier (≤ 1.0)
    mu : float                 Posterior mean DART spread
    sigma : float              Posterior predictive std dev
    prob_profit : float        P(DART > 0)
    regime : int               Hard regime label
    hdi_lower : float
    hdi_upper : float
    node : str
    delivery_timestamp : pd.Timestamp
    """

    direction: str
    raw_kelly_fraction: float
    fractional_kelly: float
    adjusted_fraction: float
    target_mw: float

    # Penalty breakdown
    penalty_hdi: float
    penalty_regime: float
    penalty_prob: float
    penalty_asymmetry: float

    # Inputs
    mu: float
    sigma: float
    prob_profit: float
    regime: int
    hdi_lower: float
    hdi_upper: float
    node: str
    delivery_timestamp: pd.Timestamp

    @property
    def regime_name(self) -> str:
        return REGIME_LABELS.get(self.regime, "Unknown")

    @property
    def is_tradeable(self) -> bool:
        return self.direction != TradeDirection.NO_TRADE and self.target_mw >= MIN_MW

    @property
    def total_penalty(self) -> float:
        return self.penalty_hdi * self.penalty_regime * self.penalty_prob * self.penalty_asymmetry

    def to_series(self) -> pd.Series:
        return pd.Series({
            "timestamp": self.delivery_timestamp,
            "node": self.node,
            "direction": self.direction,
            "target_mw": round(self.target_mw, 2),
            "raw_kelly_fraction": round(self.raw_kelly_fraction, 6),
            "adjusted_fraction": round(self.adjusted_fraction, 6),
            "total_penalty": round(self.total_penalty, 4),
            "mu": round(self.mu, 4),
            "sigma": round(self.sigma, 4),
            "prob_profit": round(self.prob_profit, 4),
            "regime": self.regime_name,
            "hdi_lower": round(self.hdi_lower, 4),
            "hdi_upper": round(self.hdi_upper, 4),
            "is_tradeable": self.is_tradeable,
        })


# ---------------------------------------------------------------------------
# Kelly Sizer
# ---------------------------------------------------------------------------

class KellySizer:
    """
    Continuous Fractional Kelly position sizer for ERCOT DART virtual trades.

    Accepts the CompleteForecast from Phase 2 and returns a KellyResult
    containing the target MW, direction, and full penalty breakdown.

    Parameters
    ----------
    max_position_mw : float
        Maximum MW cap per (node, hour). Must comply with ERCOT position limits.
    min_position_mw : float
        Minimum MW below which we suppress the trade (avoids sub-economic fills).
    fractional_kelly : float
        Kelly fraction multiplier k ∈ (0, 1]. Quarter-Kelly default (0.25).
    hurdle_rate : float
        Minimum DART spread ($/MWh) required to open a position.
        Covers transaction costs, estimated at ~$0.50/MWh.
    min_prob_profit : float
        P(DART > 0) threshold below which no trade is taken.
    hdi_prob : float
        Credible interval probability used when computing the HDI width penalty.
        Should match the hdi_prob used in Phase 2 BayesianDARTModel.predict().
    """

    def __init__(
        self,
        max_position_mw: float = 50.0,
        min_position_mw: float = 1.0,
        fractional_kelly: float = 0.25,
        hurdle_rate: float = 0.50,
        min_prob_profit: float = 0.525,
        hdi_prob: float = 0.95,
    ) -> None:
        if not 0 < fractional_kelly <= 1.0:
            raise ValueError(f"fractional_kelly must be in (0, 1]. Got {fractional_kelly}")
        if max_position_mw <= 0:
            raise ValueError("max_position_mw must be positive")

        self.max_position_mw = max_position_mw
        self.min_position_mw = min_position_mw
        self.fractional_kelly = fractional_kelly
        self.hurdle_rate = hurdle_rate
        self.min_prob_profit = min_prob_profit
        self.hdi_prob = hdi_prob

    # -----------------------------------------------------------------------
    # Penalty calculators
    # -----------------------------------------------------------------------

    def _penalty_hdi_width(self, hdi_lower: float, hdi_upper: float, sigma: float) -> float:
        """
        Penalise positions when the posterior HDI is wider than the
        Gaussian reference (4σ for a 95% interval).

        A wide HDI indicates heavier tails than assumed — the true distribution
        is more uncertain than σ alone suggests, and the Kelly fraction
        derived from σ² overstates the edge.

        penalty = min(1, gaussian_width / actual_hdi_width)
        """
        hdi_width = hdi_upper - hdi_lower
        gaussian_reference = 4.0 * sigma   # 2 × 1.96σ ≈ 4σ for 95% Gaussian
        if hdi_width < 1e-6:
            return 1.0
        penalty = min(1.0, gaussian_reference / hdi_width)
        return max(0.0, penalty)

    def _penalty_regime(self, regime: int) -> float:
        """Return regime-conditional Kelly multiplier."""
        return _REGIME_KELLY_MULTIPLIER.get(regime, 1.0)

    def _penalty_prob_profit(self, prob_profit: float) -> float:
        """
        Linear taper on P(profit):
          - prob_profit < min_prob_profit → penalty = 0 (no trade)
          - prob_profit = 1.0             → penalty = 1.0 (full size)
          - linear interpolation between  → [0, 1]

        This ensures the position scales continuously with conviction rather
        than snapping on/off at a binary threshold.
        """
        if prob_profit < self.min_prob_profit:
            return 0.0
        return min(1.0, (prob_profit - self.min_prob_profit) / (1.0 - self.min_prob_profit))

    def _penalty_hdi_asymmetry(
        self, mu: float, hdi_lower: float, hdi_upper: float
    ) -> float:
        """
        Penalise when the posterior HDI is asymmetric around μ.

        Asymmetry indicates the posterior has a fat tail on one side,
        meaning the model is uncertain about the sign of large outcomes.
        The penalty is 1 - |skew_ratio| where skew_ratio measures how
        far the interval midpoint is from μ relative to the half-width.

        A perfectly centred HDI returns penalty = 1.0.
        An HDI with its midpoint 50% of a half-width away from μ returns 0.5.
        """
        half_width = (hdi_upper - hdi_lower) / 2.0
        if half_width < 1e-6:
            return 1.0
        midpoint = (hdi_upper + hdi_lower) / 2.0
        skew_ratio = abs(midpoint - mu) / half_width
        return max(0.0, 1.0 - skew_ratio)

    # -----------------------------------------------------------------------
    # Core sizing logic
    # -----------------------------------------------------------------------

    def _raw_kelly(self, mu: float, sigma: float) -> float:
        """
        Compute f* = (μ - r) / σ² clipped to [0, MAX_KELLY].

        We clip the raw fraction at 1.0 before applying the fractional
        multiplier, since f* > 1 implies leveraged exposure which is
        not available in ERCOT virtual trading (no margin).
        """
        sigma2 = max(sigma ** 2, 1e-6)
        effective_mu = abs(mu) - self.hurdle_rate
        if effective_mu <= 0:
            return 0.0
        f_star = effective_mu / sigma2
        return float(np.clip(f_star, 0.0, 1.0))

    def _determine_direction(self, mu: float) -> str:
        """
        Determine trade direction from the sign of the expected DART spread.

        μ > hurdle_rate  → Virtual Supply (sell DAM high, buy RTM low)
        μ < -hurdle_rate → Virtual Demand (buy DAM cheap, sell RTM high)
        |μ| ≤ hurdle_rate → No trade (spread doesn't cover transaction costs)
        """
        if mu > self.hurdle_rate:
            return TradeDirection.VIRTUAL_SUPPLY
        elif mu < -self.hurdle_rate:
            return TradeDirection.VIRTUAL_DEMAND
        return TradeDirection.NO_TRADE

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def size(self, forecast: CompleteForecast) -> KellyResult:
        """
        Compute the Kelly-optimal position size for a single delivery hour.

        Parameters
        ----------
        forecast : CompleteForecast from Phase 2 ForecastingEngine.predict()

        Returns
        -------
        KellyResult with target_mw, direction, and full penalty breakdown.
        """
        mu = forecast.mu
        sigma = forecast.sigma
        prob_profit = forecast.prob_profit
        regime = forecast.current_regime
        hdi_lower, hdi_upper = forecast.hdi_95
        node = forecast.node
        ts = forecast.delivery_timestamp

        # Direction
        direction = self._determine_direction(mu)

        # Raw Kelly fraction f*
        raw_kelly = self._raw_kelly(mu, sigma)

        # Fractional Kelly
        frac_kelly = raw_kelly * self.fractional_kelly

        # Penalties
        p_hdi = self._penalty_hdi_width(hdi_lower, hdi_upper, sigma)
        p_regime = self._penalty_regime(regime)
        p_prob = self._penalty_prob_profit(prob_profit)
        p_asym = self._penalty_hdi_asymmetry(mu, hdi_lower, hdi_upper)

        # Final adjusted fraction
        adjusted = frac_kelly * p_hdi * p_regime * p_prob * p_asym
        adjusted = float(np.clip(adjusted, 0.0, 1.0))

        # No-trade gate
        if direction == TradeDirection.NO_TRADE or p_prob == 0.0:
            adjusted = 0.0
            direction = TradeDirection.NO_TRADE

        # Target MW
        target_mw = adjusted * self.max_position_mw
        if target_mw < self.min_position_mw:
            target_mw = 0.0
            direction = TradeDirection.NO_TRADE

        result = KellyResult(
            direction=direction,
            raw_kelly_fraction=raw_kelly,
            fractional_kelly=frac_kelly,
            adjusted_fraction=adjusted,
            target_mw=round(target_mw, 2),
            penalty_hdi=p_hdi,
            penalty_regime=p_regime,
            penalty_prob=p_prob,
            penalty_asymmetry=p_asym,
            mu=mu,
            sigma=sigma,
            prob_profit=prob_profit,
            regime=regime,
            hdi_lower=hdi_lower,
            hdi_upper=hdi_upper,
            node=node,
            delivery_timestamp=ts,
        )

        logger.info(
            "Kelly sizing complete",
            extra={
                "node": node,
                "direction": direction,
                "target_mw": result.target_mw,
                "mu": round(mu, 4),
                "sigma": round(sigma, 4),
                "raw_kelly": round(raw_kelly, 6),
                "adjusted": round(adjusted, 6),
                "total_penalty": round(result.total_penalty, 4),
                "regime": REGIME_LABELS.get(regime),
            },
        )
        return result

    def size_portfolio(
        self, forecasts: list[CompleteForecast]
    ) -> list[KellyResult]:
        """
        Size positions for multiple nodes simultaneously.

        The per-node MW caps are applied independently; cross-node
        correlation (portfolio Kelly) is deferred to Phase 5 risk controls.
        """
        return [self.size(fc) for fc in forecasts]
