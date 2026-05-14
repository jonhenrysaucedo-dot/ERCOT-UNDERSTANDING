"""
Wash Trade Detection for Electrically Similar ERCOT Nodes.

Background
----------
ERCOT's nodal pricing structure creates hundreds of Settlement Points with
varying degrees of electrical similarity. Two nodes are "electrically similar"
when their LMPs move in lockstep — typically because they are on the same
radial line, in the same load pocket, or share the same busbar.

Wash Trading Risk
-----------------
A wash trade in ERCOT virtual trading occurs when a QSE simultaneously holds:
  - A Virtual Supply position at Node A  (sell DAM, buy RTM)
  - A Virtual Demand position at Node B  (buy DAM, sell RTM)

AND Nodes A and B are electrically similar (LMP correlation ≥ threshold).

The result is that the DAM purchases and sales nearly offset each other,
generating minimal directional exposure while still consuming Offer ID slots
and potentially affecting clearing prices. ERCOT PUCT §25.501 prohibits
"anti-competitive virtual trading" and the CFTC has issued guidance that
wash trades that obscure price discovery are manipulative.

Detection Methodology
---------------------
We define "electrical similarity" using the historical LMP correlation
between two nodes:

    ρ(A, B) = Corr(LMP_A, LMP_B)  over a trailing 30-day window

If ρ(A, B) ≥ correlation_threshold (default 0.95), nodes A and B are
flagged as electrically similar.

A wash trade is flagged when the same QSE submits offsetting directions
(VS at A, VD at B) at two electrically similar nodes for the same delivery hour.

Additional Heuristics
---------------------
Beyond raw correlation, we use two additional signals:
  1. Shift Factor Proximity: |SFP_A - SFP_B| < shift_factor_threshold
     (indicates they react identically to the same grid injection)
  2. Hub Type Match: If both nodes are of the same type (both HB_* or both LZ_*)
     in the same zone, they are presumed similar even with < 30 days of data.

Regulatory Context
------------------
This detector generates WARNING-level violations (not CRITICAL) because:
  - Electrically similar does not mean identical — basis risk still exists
  - A trading desk may have legitimate hedging reasons for cross-node positions
  - The final call on wash trade intent requires human review

CRITICAL violations are reserved for situations where the correlation
is ≥ 0.99 (essentially the same node under a different name), or where
the same node appears in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.compliance.filters import ComplianceViolation, Severity
from ercot_dart.trading.kelly import TradeDirection
from ercot_dart.trading.order import ERCOTOrder
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# Correlation thresholds
WASH_TRADE_WARNING_CORR: float = 0.95    # Warning threshold
WASH_TRADE_CRITICAL_CORR: float = 0.99   # Critical threshold (essentially same node)

# Minimum observations required to compute correlation
MIN_CORR_OBS: int = 168   # 1 week of hourly data


# ---------------------------------------------------------------------------
# Node similarity matrix
# ---------------------------------------------------------------------------

class NodeSimilarityMatrix:
    """
    Computes and caches pairwise LMP correlations between settlement points.

    The matrix is re-computed on each model refit (weekly) using the
    trailing 30-day window from the training data, ensuring no forward leakage.

    The similarity score combines:
      - LMP Pearson correlation (primary signal)
      - Shift Factor Proxy proximity (secondary signal)
      - Same-zone same-type heuristic (tertiary fallback)
    """

    def __init__(
        self,
        corr_window_days: int = 30,
        sfp_proximity_threshold: float = 0.05,
    ) -> None:
        self.corr_window_days = corr_window_days
        self.sfp_proximity_threshold = sfp_proximity_threshold
        self._corr_matrix: Optional[pd.DataFrame] = None
        self._sfp_matrix: Optional[pd.DataFrame] = None
        self._nodes: list[str] = []

    def fit(
        self,
        dam_spp: pd.DataFrame,
        feature_matrix: Optional[pd.DataFrame] = None,
        cutoff: Optional[pd.Timestamp] = None,
    ) -> "NodeSimilarityMatrix":
        """
        Fit the similarity matrix from historical DAM SPP data.

        Parameters
        ----------
        dam_spp : Historical DAM settlement prices with columns
                  [timestamp, node, dam_spp].
        feature_matrix : Optional — provides shift_factor_proxy per node.
        cutoff : Only use data before this timestamp (walk-forward safe).
        """
        df = dam_spp.copy()
        if cutoff is not None:
            df = df[df["timestamp"] < cutoff]

        window_start = df["timestamp"].max() - pd.Timedelta(days=self.corr_window_days)
        df = df[df["timestamp"] >= window_start]

        # Pivot to wide: rows=timestamp, cols=node
        pivot = df.pivot_table(index="timestamp", columns="node", values="dam_spp")
        pivot = pivot.dropna(axis=1, thresh=MIN_CORR_OBS)

        self._nodes = list(pivot.columns)
        self._corr_matrix = pivot.corr(method="pearson")

        # Shift factor proximity matrix
        if feature_matrix is not None and "shift_factor_proxy" in feature_matrix.columns:
            sfp_mean = (
                feature_matrix[feature_matrix["timestamp"] < (cutoff or pd.Timestamp.now(tz="UTC"))]
                .groupby("node")["shift_factor_proxy"]
                .mean()
            )
            sfp_mean = sfp_mean.reindex(self._nodes)
            sfp_diff = pd.DataFrame(
                np.abs(sfp_mean.values[:, None] - sfp_mean.values[None, :]),
                index=self._nodes,
                columns=self._nodes,
            )
            self._sfp_matrix = sfp_diff

        logger.info(
            "Node similarity matrix fitted",
            extra={"n_nodes": len(self._nodes), "window_days": self.corr_window_days},
        )
        return self

    def correlation(self, node_a: str, node_b: str) -> float:
        """Return LMP correlation between two nodes. Returns NaN if unavailable."""
        if self._corr_matrix is None:
            return float("nan")
        if node_a not in self._corr_matrix.index or node_b not in self._corr_matrix.columns:
            return float("nan")
        return float(self._corr_matrix.loc[node_a, node_b])

    def sfp_proximity(self, node_a: str, node_b: str) -> float:
        """Return |SFP_A - SFP_B|. Returns NaN if unavailable."""
        if self._sfp_matrix is None:
            return float("nan")
        if node_a not in self._sfp_matrix.index or node_b not in self._sfp_matrix.columns:
            return float("nan")
        return float(self._sfp_matrix.loc[node_a, node_b])

    def are_similar(self, node_a: str, node_b: str) -> tuple[bool, float, str]:
        """
        Return (is_similar, correlation, reason_string).

        Checks correlation threshold first, then SFP proximity fallback,
        then same-type heuristic.
        """
        corr = self.correlation(node_a, node_b)

        if not np.isnan(corr):
            if corr >= WASH_TRADE_WARNING_CORR:
                return True, corr, f"LMP correlation {corr:.4f} ≥ {WASH_TRADE_WARNING_CORR}"

        # SFP proximity fallback
        sfp_diff = self.sfp_proximity(node_a, node_b)
        if not np.isnan(sfp_diff) and sfp_diff < self.sfp_proximity_threshold:
            return True, corr, f"|SFP_diff| {sfp_diff:.4f} < {self.sfp_proximity_threshold}"

        # Same-type same-zone heuristic (e.g. HB_NORTH vs HB_BUSAVG)
        def _zone(node: str) -> str:
            for zone in ["NORTH", "SOUTH", "WEST", "HOUSTON"]:
                if zone in node.upper():
                    return zone
            return ""

        def _ntype(node: str) -> str:
            return node.split("_")[0].upper() if "_" in node else ""

        if _ntype(node_a) == _ntype(node_b) and _zone(node_a) == _zone(node_b) != "":
            return True, corr, f"Same type ({_ntype(node_a)}) and zone ({_zone(node_a)})"

        return False, corr, "Not similar"

    def get_similar_pairs(
        self, nodes: list[str], threshold: float = WASH_TRADE_WARNING_CORR
    ) -> list[tuple[str, str, float]]:
        """
        Return all (node_a, node_b, correlation) pairs above the threshold.
        Excludes self-pairs.
        """
        if self._corr_matrix is None:
            return []
        pairs = []
        available = [n for n in nodes if n in self._corr_matrix.index]
        for i, a in enumerate(available):
            for b in available[i + 1:]:
                corr = float(self._corr_matrix.loc[a, b])
                if not np.isnan(corr) and corr >= threshold:
                    pairs.append((a, b, corr))
        return sorted(pairs, key=lambda x: -x[2])


# ---------------------------------------------------------------------------
# Wash trade detector
# ---------------------------------------------------------------------------

@dataclass
class WashTradeCandidate:
    """Describes a potential wash trade pair."""
    order_vs: ERCOTOrder
    order_vd: ERCOTOrder
    correlation: float
    reason: str
    severity: Severity


class WashTradeDetector:
    """
    Detects potential wash trades between electrically similar ERCOT nodes.

    For each batch of orders for the same delivery hour, it identifies
    all (VS_node, VD_node) pairs where the nodes are electrically similar
    and generates ComplianceViolations accordingly.

    Usage
    -----
        detector = WashTradeDetector(similarity_matrix)
        violations, candidates = detector.check(orders)
    """

    def __init__(self, similarity_matrix: NodeSimilarityMatrix) -> None:
        self.similarity = similarity_matrix

    def check(
        self, orders: list[ERCOTOrder]
    ) -> tuple[list[ComplianceViolation], list[WashTradeCandidate]]:
        """
        Check a batch of same-hour orders for wash trading patterns.

        Returns
        -------
        (violations, candidates)
        violations : ComplianceViolation list (WARNING or CRITICAL)
        candidates : WashTradeCandidate list for human review
        """
        violations: list[ComplianceViolation] = []
        candidates: list[WashTradeCandidate] = []

        # Partition into VS and VD orders
        vs_orders = [o for o in orders if o.direction == TradeDirection.VIRTUAL_SUPPLY]
        vd_orders = [o for o in orders if o.direction == TradeDirection.VIRTUAL_DEMAND]

        # Self-direction check: same node appearing in both directions
        vs_nodes = {o.node for o in vs_orders}
        vd_nodes = {o.node for o in vd_orders}
        same_node = vs_nodes & vd_nodes

        for node in same_node:
            vs_order = next(o for o in vs_orders if o.node == node)
            vd_order = next(o for o in vd_orders if o.node == node)
            violations.append(ComplianceViolation(
                rule="WASH_TRADE_SAME_NODE",
                severity=Severity.CRITICAL,
                message=(
                    f"Same node '{node}' appears in both VS and VD directions "
                    f"for delivery hour {vs_order.delivery_timestamp}. "
                    f"This is a direct wash trade."
                ),
                order_id=vs_order.order_id,
                node=node,
                delivery_timestamp=vs_order.delivery_timestamp,
            ))

        # Cross-node similarity check
        for vs_order in vs_orders:
            for vd_order in vd_orders:
                if vs_order.node == vd_order.node:
                    continue  # already handled above

                is_similar, corr, reason = self.similarity.are_similar(
                    vs_order.node, vd_order.node
                )
                if not is_similar:
                    continue

                severity = (
                    Severity.CRITICAL if corr >= WASH_TRADE_CRITICAL_CORR
                    else Severity.WARNING
                )

                candidate = WashTradeCandidate(
                    order_vs=vs_order,
                    order_vd=vd_order,
                    correlation=corr,
                    reason=reason,
                    severity=severity,
                )
                candidates.append(candidate)

                violations.append(ComplianceViolation(
                    rule="WASH_TRADE_SIMILAR_NODES",
                    severity=severity,
                    message=(
                        f"Potential wash trade: VS at '{vs_order.node}' vs "
                        f"VD at '{vd_order.node}' — {reason}. "
                        f"Recommend human review before submission."
                    ),
                    order_id=vs_order.order_id,
                    node=vs_order.node,
                    delivery_timestamp=vs_order.delivery_timestamp,
                ))

        if violations:
            logger.warning(
                "Wash trade check flagged orders",
                extra={
                    "n_violations": len(violations),
                    "critical": sum(1 for v in violations if v.severity == Severity.CRITICAL),
                    "warning": sum(1 for v in violations if v.severity == Severity.WARNING),
                },
            )

        return violations, candidates
