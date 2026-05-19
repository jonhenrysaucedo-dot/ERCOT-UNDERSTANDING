"""10-tier bid curve generator — M8.

For each operating hour with a non-zero Kelly position, generates 10 (price, quantity)
pairs forming a monotonic limit-price curve. ERCOT requires strict monotonicity.

Tier structure (per PRD §M8):
    Tiers 1–3  (Base,    ~30% of volume): prices tightly around q50 DAM expected
    Tiers 4–8  (Scaling, ~50% of volume): prices between q50 and q90 (INC) / q10 (DEC)
    Tiers 9–10 (Tail,    ~20% of volume): extreme limit prices beyond q90 / q10

Price bounds (from config/risk.yaml):
    price_cap:   $5000/MWh  (ERCOT offer cap)
    price_floor: -$250/MWh  (ERCOT floor)

Monotonicity:
    INC bids: prices are monotonically INCREASING (higher price = more willing to sell DAM)
    DEC bids: prices are monotonically DECREASING (lower price = more willing to buy DAM)

Walk-forward safety:
    generate_bids() requires timezone-aware as_of_timestamp.
    All inputs (forecast_df, kelly_allocations) come from walk-forward compliant callers.
    Raises WalkForwardViolation on naive datetimes.

Usage:
    gen = TierBidGenerator.from_config("config/risk.yaml")
    bids = gen.generate_bids(forecast_df, kelly_allocations, as_of_timestamp)
    gen.validate_monotonicity(bids)  # raises if ERCOT constraint violated
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
import polars as pl
import structlog
import yaml

from src.ingest.exceptions import WalkForwardViolation

logger = structlog.get_logger(__name__)
UTC = timezone.utc

NUM_TIERS = 10
# Volume fractions: tiers 1-3 = 30%, tiers 4-8 = 50%, tiers 9-10 = 20%
TIER_VOLUME_WEIGHTS = np.array([
    0.10, 0.10, 0.10,           # base: 3 × 10%
    0.10, 0.10, 0.10, 0.10, 0.10,  # scaling: 5 × 10%
    0.10, 0.10,                  # tail: 2 × 10%
])
# Sanity: sums to 1.0
assert abs(TIER_VOLUME_WEIGHTS.sum() - 1.0) < 1e-9


def _require_utc(ts: datetime) -> None:
    if ts.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware; got naive {ts!r}"
        )


class TierBidGenerator:
    """Generates 10-tier monotone limit-price bid curves per hour (M8).

    Walk-forward safety:
        generate_bids() requires timezone-aware as_of_timestamp.
        Inputs are walk-forward compliant by caller contract.
        Raises WalkForwardViolation on naive datetimes.
    """

    def __init__(
        self,
        num_tiers: int = NUM_TIERS,
        price_cap: float = 5000.0,
        price_floor: float = -250.0,
    ) -> None:
        if num_tiers != NUM_TIERS:
            raise ValueError(
                f"ERCOT virtual bid curves must have exactly {NUM_TIERS} tiers; got {num_tiers}"
            )
        self.num_tiers = num_tiers
        self.price_cap = price_cap
        self.price_floor = price_floor

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "TierBidGenerator":
        """Load generator configuration from YAML."""
        with Path(config_path).open() as f:
            cfg = yaml.safe_load(f)
        return cls(
            num_tiers=cfg.get("num_offer_tiers", NUM_TIERS),
            price_cap=cfg["price_cap"],
            price_floor=cfg["price_floor"],
        )

    def generate_bids(
        self,
        forecast_df: pl.DataFrame,
        kelly_allocations: pl.DataFrame,
        as_of_timestamp: datetime,
    ) -> pl.DataFrame:
        """Generate 10-tier bid curves for all hours with non-zero Kelly positions.

        Walk-forward safety:
            as_of_timestamp must be timezone-aware. Inputs must be from
            walk-forward compliant callers (M5 forecast + M7 Kelly sizer).

        Args:
            forecast_df: From DARTBayesianForecaster.forecast() —
                [interval_start_utc, q10, q50, q90, p_positive, p_negative]
            kelly_allocations: From KellySizer.size_positions() —
                [interval_start_utc, direction, position_mw, ...]
            as_of_timestamp: Timezone-aware. Used for audit logging.

        Returns:
            Polars DataFrame (long format, 10 rows per eligible hour) with columns:
                interval_start_utc, hour_ending, direction,
                tier (1–10), price_usd_per_mwh, quantity_mw, is_tail_tier
        """
        _require_utc(as_of_timestamp)
        self._validate_inputs(forecast_df, kelly_allocations)

        joined = kelly_allocations.join(
            forecast_df.select(["interval_start_utc", "q10", "q50", "q90"]),
            on="interval_start_utc",
            how="left",
        )

        rows = []
        for row in joined.filter(pl.col("position_mw") > 0).iter_rows(named=True):
            hour_rows = self._generate_hour_tiers(row)
            rows.extend(hour_rows)

        if not rows:
            return pl.DataFrame(schema={
                "interval_start_utc": pl.Datetime("us", "UTC"),
                "hour_ending": pl.Int32,
                "direction": pl.String,
                "tier": pl.Int32,
                "price_usd_per_mwh": pl.Float64,
                "quantity_mw": pl.Float64,
                "is_tail_tier": pl.Boolean,
            })

        result = pl.DataFrame(rows)
        logger.info(
            "bids_generated",
            as_of=as_of_timestamp.isoformat(),
            n_hours=result["interval_start_utc"].n_unique(),
            total_tiers=len(result),
        )
        return result

    def validate_monotonicity(self, bids: pl.DataFrame) -> None:
        """Validate ERCOT monotonicity constraint per hour.

        INC: prices must be strictly increasing tier 1→10.
        DEC: prices must be strictly decreasing tier 1→10.

        Raises:
            ValueError: if any hour violates monotonicity.
        """
        if len(bids) == 0:
            return

        for (ts, direction), group in bids.group_by(
            ["interval_start_utc", "direction"], maintain_order=True
        ):
            prices = group.sort("tier")["price_usd_per_mwh"].to_numpy()
            if direction == "INC":
                if not np.all(np.diff(prices) > 0):
                    raise ValueError(
                        f"INC bid for {ts} has non-strictly-increasing prices: {prices}"
                    )
            elif direction == "DEC":
                if not np.all(np.diff(prices) < 0):
                    raise ValueError(
                        f"DEC bid for {ts} has non-strictly-decreasing prices: {prices}"
                    )

    # ── internal ────────────────────────────────────────────────────────────

    def _generate_hour_tiers(self, row: dict) -> list[dict]:
        """Build 10 price-quantity pairs for a single hour."""
        direction = row["direction"]
        total_mw = row["position_mw"]
        q50 = row.get("q50") or 0.0
        q10 = row.get("q10") or (q50 - 5.0)
        q90 = row.get("q90") or (q50 + 5.0)

        prices = self._build_price_grid(direction, q50, q10, q90)
        quantities = TIER_VOLUME_WEIGHTS * total_mw

        # Timestamp → ERCOT HourEnding (1-24, CT-based)
        # Use the UTC interval_start + 1h offset + convert to CT
        ts = row["interval_start_utc"]
        # HourEnding is simply the UTC hour + 1 for approximation in v1
        # (full CT conversion happens at the DAM submission layer, not here)
        hour_ending = int((ts.hour + 1) % 24) or 24

        result = []
        for i, (p, q) in enumerate(zip(prices, quantities)):
            result.append({
                "interval_start_utc": ts,
                "hour_ending": hour_ending,
                "direction": direction,
                "tier": i + 1,
                "price_usd_per_mwh": round(float(p), 4),
                "quantity_mw": round(float(q), 4),
                "is_tail_tier": (i + 1) >= 9,
            })
        return result

    def _build_price_grid(
        self,
        direction: str,
        q50: float,
        q10: float,
        q90: float,
    ) -> np.ndarray:
        """Build 10 tier prices satisfying ERCOT monotonicity.

        INC structure (monotone increasing):
            Tiers 1-3:  linspace(q50 - δ_base, q50 + δ_base/2, 3)
            Tiers 4-8:  linspace(q50 + δ_base/2, q90, 5)
            Tiers 9-10: linspace(q90, price_cap, 2) (exclusive of q90)

        DEC structure (monotone decreasing):
            Tiers 1-3:  linspace(q50 + δ_base, q50 - δ_base/2, 3)
            Tiers 4-8:  linspace(q50 - δ_base/2, q10, 5)
            Tiers 9-10: linspace(q10, price_floor, 2)
        """
        delta = max(abs(q90 - q50), abs(q50 - q10), 1.0)

        if direction == "INC":
            base = np.linspace(q50 - delta * 0.5, q50 + delta * 0.25, 3)
            scaling = np.linspace(q50 + delta * 0.25, q90, 6)[1:]  # skip q50+δ/4 (overlap)
            tail_start = q90 + delta * 0.25
            tail = np.linspace(tail_start, self.price_cap, 3)[1:]  # 2 prices
            prices = np.concatenate([base, scaling, tail])
        else:  # DEC
            base = np.linspace(q50 + delta * 0.5, q50 - delta * 0.25, 3)
            scaling = np.linspace(q50 - delta * 0.25, q10, 6)[1:]
            tail_start = q10 - delta * 0.25
            tail = np.linspace(tail_start, self.price_floor, 3)[1:]
            prices = np.concatenate([base, scaling, tail])

        # Clip to ERCOT bounds
        prices = np.clip(prices, self.price_floor, self.price_cap)

        # Enforce strict monotonicity: add small epsilon if needed
        eps = 0.01
        if direction == "INC":
            for i in range(1, len(prices)):
                if prices[i] <= prices[i - 1]:
                    prices[i] = prices[i - 1] + eps
        else:
            for i in range(1, len(prices)):
                if prices[i] >= prices[i - 1]:
                    prices[i] = prices[i - 1] - eps

        # Final clip after epsilon correction
        prices = np.clip(prices, self.price_floor, self.price_cap)

        assert len(prices) == NUM_TIERS, f"Expected {NUM_TIERS} prices, got {len(prices)}"
        return prices

    def _validate_inputs(
        self,
        forecast_df: pl.DataFrame,
        kelly_allocations: pl.DataFrame,
    ) -> None:
        required_f = {"interval_start_utc", "q10", "q50", "q90"}
        required_k = {"interval_start_utc", "direction", "position_mw"}

        missing_f = required_f - set(forecast_df.columns)
        if missing_f:
            raise ValueError(f"forecast_df missing: {missing_f}")

        missing_k = required_k - set(kelly_allocations.columns)
        if missing_k:
            raise ValueError(f"kelly_allocations missing: {missing_k}")
