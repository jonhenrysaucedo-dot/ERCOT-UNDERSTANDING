"""
Pre-Trade Compliance Filters for ERCOT DAM Virtual Trading.

ERCOT Market Rules impose hard limits on DAM submissions. Violations result
in order rejection by the ERCOT settlement engine, so all checks must pass
BEFORE submission (pre-trade), not after (post-trade).

Applicable ERCOT Protocols (PUCT Substantive Rules §25.501)
------------------------------------------------------------
1. Offer ID Limit — PUCT §25.501(f)(4):
   "A QSE may submit no more than 35 Energy Only Offer IDs per Settlement
   Point per Market Day."

   Implication:
   - For a 24-hour trading day, 35 offers per node is almost never binding
     when trading a single delivery hour per offer.
   - BUT: a QSE running multiple strategies (DAM energy, ancillary services,
     virtual trading) shares the 35-slot pool. Our compliance engine
     must track daily slot consumption ACROSS all submissions from the QSE.
   - Each TierCurve submitted = 1 Offer ID consumed.

2. Price Bounds — PUCT §25.501(d):
   - Offers: $-250/MWh ≤ price ≤ $5,000/MWh (Low Cap / HCAP)
   - Bids:   $-250/MWh ≤ price ≤ $5,000/MWh

3. Minimum Offer Quantity — ERCOT Nodal Operating Guide §6.5.7:
   - Minimum MW per offer tier: 0.1 MW

4. Settlement Point Eligibility — ERCOT Nodal Protocols §4.6.1:
   - Virtual trades are only permitted at Hub Settlement Points (HB_*) and
     Load Zone Settlement Points (LZ_*). Resource-specific nodes are ineligible.
   - Node type must be in {HB, LZ, DC_TIE}.

5. Gate Closure — ERCOT Nodal Protocols §4.6.3:
   - All DAM offers/bids must be submitted by 10:00 AM CPT on the day
     BEFORE the delivery date.
   - The compliance engine checks submission timestamps against this deadline.

Compliance Report
-----------------
Each pre-trade check returns a ComplianceViolation. The ComplianceEngine
aggregates these into a ComplianceReport with a binary pass/fail verdict
and a ranked list of violations. Orders that fail MUST NOT be submitted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd

from ercot_dart.config import (
    DAM_GATE_CLOSE,
    MAX_OFFER_IDS_PER_NODE,
    MIN_MW,
    NUM_OFFER_TIERS,
    PRICE_CAP,
    PRICE_FLOOR,
)
from ercot_dart.trading.order import ERCOTOrder
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# Settlement point types eligible for virtual trading
ELIGIBLE_NODE_PREFIXES: tuple[str, ...] = ("HB_", "LZ_", "DC_")


# ---------------------------------------------------------------------------
# Violation severity
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "CRITICAL"    # Must block submission
    WARNING = "WARNING"      # Flagged but not blocking
    INFO = "INFO"


# ---------------------------------------------------------------------------
# Violation record
# ---------------------------------------------------------------------------

@dataclass
class ComplianceViolation:
    rule: str
    severity: Severity
    message: str
    order_id: str
    node: str
    delivery_timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity.value,
            "message": self.message,
            "order_id": self.order_id[:8],
            "node": self.node,
            "delivery_timestamp": str(self.delivery_timestamp),
        }


# ---------------------------------------------------------------------------
# Compliance report
# ---------------------------------------------------------------------------

@dataclass
class ComplianceReport:
    """
    Aggregated compliance verdict for a batch of orders.

    Orders in `approved` passed all CRITICAL checks.
    Orders in `blocked` failed at least one CRITICAL check.
    """
    approved: list[ERCOTOrder] = field(default_factory=list)
    blocked: list[ERCOTOrder] = field(default_factory=list)
    violations: list[ComplianceViolation] = field(default_factory=list)

    @property
    def n_approved(self) -> int:
        return len(self.approved)

    @property
    def n_blocked(self) -> int:
        return len(self.blocked)

    @property
    def critical_violations(self) -> list[ComplianceViolation]:
        return [v for v in self.violations if v.severity == Severity.CRITICAL]

    def summary(self) -> pd.DataFrame:
        if not self.violations:
            return pd.DataFrame()
        return pd.DataFrame([v.to_dict() for v in self.violations])

    def log(self) -> None:
        logger.info(
            "Compliance report",
            extra={
                "approved": self.n_approved,
                "blocked": self.n_blocked,
                "critical_violations": len(self.critical_violations),
                "warnings": sum(1 for v in self.violations if v.severity == Severity.WARNING),
            },
        )
        for v in self.critical_violations:
            logger.warning(
                "CRITICAL compliance violation",
                extra=v.to_dict(),
            )


# ---------------------------------------------------------------------------
# Individual compliance checks
# ---------------------------------------------------------------------------

class _OfferIDLimitChecker:
    """
    Tracks cumulative Offer ID consumption per (node, delivery_date)
    and flags orders that would exceed the 35 Offer ID limit.

    The checker is stateful within a trading session: it accumulates
    submitted order counts from the ERCOT submission blotter and checks
    new orders against the remaining capacity.
    """

    def __init__(self, max_offers: int = MAX_OFFER_IDS_PER_NODE) -> None:
        self.max_offers = max_offers
        # {(node, date): count_already_submitted}
        self._submitted_counts: dict[tuple, int] = defaultdict(int)

    def record_submission(self, node: str, delivery_date: pd.Timestamp) -> None:
        """Call this when an order is confirmed submitted to ERCOT."""
        key = (node, delivery_date.normalize())
        self._submitted_counts[key] += 1

    def check(self, order: ERCOTOrder) -> Optional[ComplianceViolation]:
        if order.delivery_timestamp is None:
            return None
        key = (order.node, order.delivery_timestamp.normalize())
        current = self._submitted_counts[key]
        if current >= self.max_offers:
            return ComplianceViolation(
                rule="OFFER_ID_LIMIT",
                severity=Severity.CRITICAL,
                message=(
                    f"Node {order.node} on {order.delivery_timestamp.date()} "
                    f"has {current}/{self.max_offers} Offer IDs consumed. "
                    f"Submission would exceed PUCT §25.501(f)(4) limit."
                ),
                order_id=order.order_id,
                node=order.node,
                delivery_timestamp=order.delivery_timestamp,
            )
        return None


class _PriceBoundsChecker:
    """
    Verifies all tier prices are within [PRICE_FLOOR, PRICE_CAP].
    """

    def check(self, order: ERCOTOrder) -> list[ComplianceViolation]:
        violations = []
        if order.tier_curve is None:
            return violations
        for tier in order.tier_curve.tiers:
            if tier.price < PRICE_FLOOR or tier.price > PRICE_CAP:
                violations.append(ComplianceViolation(
                    rule="PRICE_BOUNDS",
                    severity=Severity.CRITICAL,
                    message=(
                        f"Tier {tier.tier_id} price {tier.price:.2f} outside "
                        f"[{PRICE_FLOOR}, {PRICE_CAP}] $/MWh."
                    ),
                    order_id=order.order_id,
                    node=order.node,
                    delivery_timestamp=order.delivery_timestamp,
                ))
        return violations


class _MinimumMWChecker:
    """
    Verifies all active (non-zero) tiers meet the 0.1 MW minimum.
    """

    def check(self, order: ERCOTOrder) -> list[ComplianceViolation]:
        violations = []
        if order.tier_curve is None:
            return violations
        for tier in order.tier_curve.tiers:
            if 0 < tier.mw < MIN_MW:
                violations.append(ComplianceViolation(
                    rule="MINIMUM_MW",
                    severity=Severity.CRITICAL,
                    message=(
                        f"Tier {tier.tier_id} MW {tier.mw:.3f} < "
                        f"minimum {MIN_MW} MW (ERCOT NOG §6.5.7)."
                    ),
                    order_id=order.order_id,
                    node=order.node,
                    delivery_timestamp=order.delivery_timestamp,
                ))
        return violations


class _MonotonicityChecker:
    """
    Verifies tier prices are monotonically non-decreasing (supply) or
    non-increasing (demand).
    """

    def check(self, order: ERCOTOrder) -> Optional[ComplianceViolation]:
        if order.tier_curve is None:
            return None
        violations = order.tier_curve.validate()
        mono_violations = [v for v in violations if "monoton" in v.lower()]
        if mono_violations:
            return ComplianceViolation(
                rule="MONOTONICITY",
                severity=Severity.CRITICAL,
                message="; ".join(mono_violations),
                order_id=order.order_id,
                node=order.node,
                delivery_timestamp=order.delivery_timestamp,
            )
        return None


class _EligibleNodeChecker:
    """
    Verifies the settlement point is eligible for virtual trading
    (Hub or Load Zone node, not a resource-specific node).
    """

    def check(self, order: ERCOTOrder) -> Optional[ComplianceViolation]:
        node = order.node.upper()
        if not any(node.startswith(prefix) for prefix in ELIGIBLE_NODE_PREFIXES):
            return ComplianceViolation(
                rule="INELIGIBLE_NODE",
                severity=Severity.CRITICAL,
                message=(
                    f"Node '{order.node}' is not eligible for virtual trading. "
                    f"Only Hub (HB_*), Load Zone (LZ_*), and DC Tie (DC_*) nodes "
                    f"are permitted under ERCOT Nodal Protocols §4.6.1."
                ),
                order_id=order.order_id,
                node=order.node,
                delivery_timestamp=order.delivery_timestamp,
            )
        return None


class _GateClosureChecker:
    """
    Verifies that the order was created before the 10:00 AM CPT gate closure
    for the target delivery date.

    In production: `submission_time` is the current wall-clock time.
    In backtesting: `submission_time` is set to 09:00 AM of the target date.
    """

    def check(
        self,
        order: ERCOTOrder,
        submission_time: Optional[datetime] = None,
    ) -> Optional[ComplianceViolation]:
        if order.delivery_timestamp is None:
            return None

        now = submission_time or datetime.now(timezone.utc)
        # Gate closes at 10:00 AM CPT on the delivery date - 1 day
        gate_day = order.delivery_timestamp.normalize() - pd.Timedelta(days=1)
        gate_close_utc = pd.Timestamp(
            gate_day.year, gate_day.month, gate_day.day,
            DAM_GATE_CLOSE.hour, DAM_GATE_CLOSE.minute,
            tz="America/Chicago"
        ).tz_convert("UTC")

        if pd.Timestamp(now).tz_localize("UTC" if now.tzinfo is None else None) > gate_close_utc:
            return ComplianceViolation(
                rule="GATE_CLOSURE",
                severity=Severity.CRITICAL,
                message=(
                    f"Submission attempted after gate closure "
                    f"({gate_close_utc.isoformat()} UTC) "
                    f"for delivery {order.delivery_timestamp.date()}."
                ),
                order_id=order.order_id,
                node=order.node,
                delivery_timestamp=order.delivery_timestamp,
            )
        return None


class _DuplicateOrderChecker:
    """
    Prevents submitting duplicate offers for the same (node, delivery_hour).
    Two offers for the same hour at the same node on the same day
    would consume two Offer IDs for identical exposure.
    """

    def __init__(self) -> None:
        # {(node, delivery_timestamp): order_id}
        self._submitted: dict[tuple, str] = {}

    def record(self, order: ERCOTOrder) -> None:
        key = (order.node, order.delivery_timestamp)
        self._submitted[key] = order.order_id

    def check(self, order: ERCOTOrder) -> Optional[ComplianceViolation]:
        key = (order.node, order.delivery_timestamp)
        existing_id = self._submitted.get(key)
        if existing_id is not None:
            return ComplianceViolation(
                rule="DUPLICATE_ORDER",
                severity=Severity.CRITICAL,
                message=(
                    f"Duplicate order detected for node '{order.node}' "
                    f"at {order.delivery_timestamp}. "
                    f"Existing order ID: {existing_id[:8]}."
                ),
                order_id=order.order_id,
                node=order.node,
                delivery_timestamp=order.delivery_timestamp,
            )
        return None


# ---------------------------------------------------------------------------
# Compliance Engine
# ---------------------------------------------------------------------------

class ComplianceEngine:
    """
    Pre-trade compliance gate for ERCOT DAM virtual order submissions.

    Runs all applicable ERCOT compliance checks on a batch of ERCOTOrder
    objects and returns a ComplianceReport partitioning orders into
    `approved` (safe to submit) and `blocked` (must not submit).

    The engine is stateful within a session — it tracks cumulative
    Offer ID consumption and submitted orders to enforce daily limits.

    Usage
    -----
        engine = ComplianceEngine()
        report = engine.check_batch(orders)
        for order in report.approved:
            submit_to_ercot(order)
            engine.record_submission(order)
        report.log()
    """

    def __init__(self, max_offers_per_node: int = MAX_OFFER_IDS_PER_NODE) -> None:
        self._offer_id_checker = _OfferIDLimitChecker(max_offers_per_node)
        self._price_bounds = _PriceBoundsChecker()
        self._min_mw = _MinimumMWChecker()
        self._monotonicity = _MonotonicityChecker()
        self._eligible_node = _EligibleNodeChecker()
        self._gate_closure = _GateClosureChecker()
        self._duplicate = _DuplicateOrderChecker()

    def check_order(
        self,
        order: ERCOTOrder,
        submission_time: Optional[datetime] = None,
    ) -> list[ComplianceViolation]:
        """Run all compliance checks on a single order. Returns list of violations."""
        violations: list[ComplianceViolation] = []

        # CRITICAL checks (any one blocks the order)
        v = self._eligible_node.check(order)
        if v:
            violations.append(v)

        v = self._gate_closure.check(order, submission_time)
        if v:
            violations.append(v)

        v = self._offer_id_checker.check(order)
        if v:
            violations.append(v)

        v = self._duplicate.check(order)
        if v:
            violations.append(v)

        v = self._monotonicity.check(order)
        if v:
            violations.append(v)

        violations.extend(self._price_bounds.check(order))
        violations.extend(self._min_mw.check(order))

        return violations

    def check_batch(
        self,
        orders: list[ERCOTOrder],
        submission_time: Optional[datetime] = None,
    ) -> ComplianceReport:
        """
        Run compliance checks on a batch of orders.

        Returns a ComplianceReport with `approved` and `blocked` lists.
        Orders are approved only if they have zero CRITICAL violations.
        """
        report = ComplianceReport()

        for order in orders:
            violations = self.check_order(order, submission_time)
            report.violations.extend(violations)

            critical = [v for v in violations if v.severity == Severity.CRITICAL]
            if critical:
                report.blocked.append(order)
            else:
                report.approved.append(order)

        report.log()
        return report

    def record_submission(self, order: ERCOTOrder) -> None:
        """
        Record a successful submission so that daily Offer ID counts
        and duplicate detection are updated for subsequent orders.
        """
        if order.delivery_timestamp is not None:
            self._offer_id_checker.record_submission(
                order.node, order.delivery_timestamp
            )
        self._duplicate.record(order)
