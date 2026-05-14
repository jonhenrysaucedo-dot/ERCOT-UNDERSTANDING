"""
Walk-Forward Validation Framework for ERCOT DART Backtesting.

Walk-forward validation is the only correct backtesting methodology for
time-series strategies. Unlike k-fold cross-validation, it never allows
future data to contaminate training — each fold's test set strictly follows
its training set chronologically.

Two Modes
---------
1. EXPANDING WINDOW (anchored origin):
   Fold 1: train [T0, T1], test [T1, T1+step]
   Fold 2: train [T0, T1+step], test [T1+step, T1+2·step]
   ...
   The training set grows monotonically. Best for strategies where more
   history improves regime detection (HMM parameter stability).

2. ROLLING WINDOW (fixed-length):
   Fold 1: train [T0, T0+W], test [T0+W, T0+W+step]
   Fold 2: train [T0+step, T0+W+step], test [T0+W+step, T0+W+2·step]
   ...
   The training window slides. Best for strategies requiring recency
   (MS-GARCH parameters adapt to volatility regime changes).

Look-Ahead Bias Prevention
--------------------------
The gate-closure constraint from Phase 1 ensures feature construction
never leaks same-day information. The walk-forward splitter provides an
additional layer: the FeatureEngineer is given ONLY the train-window
DataFrame, so scaler.fit(), HMM.fit(), and MCMC.fit() all see only past data.

The test-window features are then transformed using the train-fitted scaler
(transform-only, no refit), and predictions are made strictly out-of-sample.

ERCOT-Specific Considerations
------------------------------
- The HMM requires at least 168 observations (1 week hourly) to converge
- The MCMC sampler requires at least 200 observations for stable posteriors
- We enforce a minimum train_days parameter to guard these constraints
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Optional

import pandas as pd
import numpy as np

from ercot_dart.etl.pipeline import ParsedDataStore
from ercot_dart.models.forecasting_engine import ForecastingConfig, ForecastingEngine
from ercot_dart.trading.order import ERCOTOrder, OrchestratorConfig, TradingOrchestrator
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward modes
# ---------------------------------------------------------------------------

class WindowMode(str, Enum):
    EXPANDING = "expanding"
    ROLLING = "rolling"


# ---------------------------------------------------------------------------
# Split descriptor
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplit:
    """
    Defines one (train, test) time-window pair.

    All boundary timestamps are inclusive on the left, exclusive on the right.
    """
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp        # exclusive — last training observation is just before
    test_start: pd.Timestamp       # = train_end
    test_end: pd.Timestamp         # exclusive

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"train [{self.train_start.date()} → {self.train_end.date()}] "
            f"test [{self.test_start.date()} → {self.test_end.date()}]"
        )


# ---------------------------------------------------------------------------
# Split generator
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """
    Generates chronologically ordered (train, test) splits from a
    timestamp-indexed feature matrix.

    Parameters
    ----------
    min_train_days : int
        Minimum number of training days before we allow testing.
        Gates HMM convergence and MCMC posterior stability.
    test_days : int
        Number of days in each test fold (step size of the walk).
    mode : WindowMode
        EXPANDING (growing train set) or ROLLING (fixed-size train window).
    max_train_days : int, optional
        Maximum training window size for ROLLING mode. Ignored for EXPANDING.
    """

    def __init__(
        self,
        min_train_days: int = 120,
        test_days: int = 7,
        mode: WindowMode = WindowMode.EXPANDING,
        max_train_days: Optional[int] = None,
    ) -> None:
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.mode = mode
        self.max_train_days = max_train_days  # only used in ROLLING

    def generate(
        self,
        feature_matrix: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> list[WalkForwardSplit]:
        """
        Generate all valid (train, test) splits from the feature matrix.

        Returns splits in chronological order. The first test fold starts
        at min_train_days after the dataset origin.
        """
        ts = feature_matrix[timestamp_col]
        origin = ts.min().normalize()
        end = ts.max().normalize() + pd.Timedelta(days=1)

        test_step = pd.Timedelta(days=self.test_days)
        min_train_td = pd.Timedelta(days=self.min_train_days)

        splits: list[WalkForwardSplit] = []
        fold_id = 0
        test_start = origin + min_train_td

        while test_start + test_step <= end:
            test_end = test_start + test_step

            if self.mode == WindowMode.EXPANDING:
                train_start = origin
            else:
                # Rolling: clamp train window to max_train_days
                lookback = pd.Timedelta(days=self.max_train_days or self.min_train_days)
                train_start = max(origin, test_start - lookback)

            splits.append(WalkForwardSplit(
                fold_id=fold_id,
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            ))
            fold_id += 1
            test_start = test_end

        logger.info(
            "Walk-forward splits generated",
            extra={
                "n_folds": len(splits),
                "mode": self.mode,
                "min_train_days": self.min_train_days,
                "test_days": self.test_days,
            },
        )
        return splits

    def split_dataframe(
        self,
        feature_matrix: pd.DataFrame,
        split: WalkForwardSplit,
        timestamp_col: str = "timestamp",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Slice the feature matrix into train and test DataFrames for a single fold.

        The test set contains only pre-gate features (enforced by Phase 1).
        """
        ts = feature_matrix[timestamp_col]
        train_mask = (ts >= split.train_start) & (ts < split.train_end)
        test_mask = (ts >= split.test_start) & (ts < split.test_end)
        return feature_matrix[train_mask].copy(), feature_matrix[test_mask].copy()


# ---------------------------------------------------------------------------
# Fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """
    Results from a single walk-forward fold.

    Attributes
    ----------
    split : WalkForwardSplit
    orders : ERCOTOrder list — all orders generated during the test window
    n_no_trade : int — hours where no trade signal was generated
    fit_elapsed_s : float — time to refit all models (HMM + GARCH + MCMC)
    predict_elapsed_s : float — time to generate all forecasts + orders
    """
    split: WalkForwardSplit
    orders: list[ERCOTOrder]
    n_no_trade: int
    fit_elapsed_s: float
    predict_elapsed_s: float
    error: Optional[str] = None

    @property
    def n_orders(self) -> int:
        return len(self.orders)

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Orchestrates the full walk-forward backtest loop.

    For each fold:
      1. Slice feature_matrix into train / test windows
      2. Refit ForecastingEngine on train data (HMM → GARCH → MCMC)
      3. For each delivery hour in the test window:
           a. Extract pre-gate feature row
           b. Call TradingOrchestrator.run() → ERCOTOrder(s)
      4. Collect all orders across folds for market impact simulation

    The validator does NOT call MarketImpactSimulator or RTMSettlementAggregator
    — those are invoked by BacktestEngine after all orders are collected, since
    market impact requires access to the full parsed bid/offer stacks (ParsedDataStore).

    Parameters
    ----------
    splitter : WalkForwardSplitter
    forecasting_config : ForecastingConfig passed to each ForecastingEngine refit
    orchestrator_config : OrchestratorConfig for KellySizer + TierCurveGenerator
    price_anchor_fn : Callable[[str, pd.Timestamp, pd.DataFrame], float]
        Function that returns a price anchor (expected DAM SPP) given
        (node, delivery_timestamp, train_feature_matrix). Defaults to
        rolling 30-day mean DAM SPP from the training window.
    """

    def __init__(
        self,
        splitter: Optional[WalkForwardSplitter] = None,
        forecasting_config: Optional[ForecastingConfig] = None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        price_anchor_fn: Optional[Callable] = None,
    ) -> None:
        self.splitter = splitter or WalkForwardSplitter()
        self.forecasting_config = forecasting_config or ForecastingConfig()
        self.orchestrator_config = orchestrator_config or OrchestratorConfig()
        self.price_anchor_fn = price_anchor_fn or self._default_price_anchor

    # -----------------------------------------------------------------------
    # Default price anchor: rolling 30-day mean DAM SPP from training data
    # -----------------------------------------------------------------------

    @staticmethod
    def _default_price_anchor(
        node: str,
        delivery_timestamp: pd.Timestamp,
        train_df: pd.DataFrame,
    ) -> float:
        """
        Compute the rolling 30-day mean DAM SPP from the training window.

        This is the simplest non-forward-looking price anchor: the recent
        average of observed DAM prices at the node gives a stable baseline
        around which the 10-tier curve is constructed.
        """
        node_data = train_df[train_df["node"] == node]
        if "dam_spp" not in node_data.columns or node_data.empty:
            return 30.0   # ERCOT all-in avg energy price fallback

        cutoff = delivery_timestamp - pd.Timedelta(days=30)
        recent = node_data[node_data["timestamp"] >= cutoff]["dam_spp"]
        if len(recent) < 24:
            recent = node_data["dam_spp"]   # use full training window

        return float(recent.mean())

    # -----------------------------------------------------------------------
    # Delivery hour iterator
    # -----------------------------------------------------------------------

    @staticmethod
    def _delivery_hours(test_df: pd.DataFrame) -> list[pd.Timestamp]:
        """
        Extract the unique delivery timestamps in the test window,
        sorted chronologically.

        We iterate hour-by-hour rather than by unique timestamp to ensure
        each delivery hour produces exactly one set of orders.
        """
        if "timestamp" not in test_df.columns:
            return []
        return sorted(test_df["timestamp"].unique())

    # -----------------------------------------------------------------------
    # Single fold execution
    # -----------------------------------------------------------------------

    def _run_fold(
        self,
        fold_id: int,
        split: WalkForwardSplit,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> FoldResult:
        """Fit on train, predict on test for one walk-forward fold."""

        # ---- Refit forecasting engine ----------------------------------
        t_fit_start = _time.monotonic()
        engine = ForecastingEngine(config=self.forecasting_config)
        try:
            engine.fit(train_df)
        except Exception as e:
            logger.error(
                "Engine fit failed on fold",
                extra={"fold": fold_id, "error": str(e)},
            )
            return FoldResult(
                split=split, orders=[], n_no_trade=0,
                fit_elapsed_s=0.0, predict_elapsed_s=0.0,
                error=str(e),
            )
        fit_elapsed = _time.monotonic() - t_fit_start

        orchestrator = TradingOrchestrator(engine, self.orchestrator_config)

        # ---- Predict on each test delivery hour -----------------------
        all_orders: list[ERCOTOrder] = []
        n_no_trade = 0
        t_pred_start = _time.monotonic()

        delivery_hours = self._delivery_hours(test_df)

        for delivery_ts in delivery_hours:
            # Pre-gate features: only rows strictly before this delivery hour
            # (the 10:00 AM gate cuts off same-day features — already enforced
            # by Phase 1 FeatureEngineer._gate_filter(), but we re-apply here
            # as a belt-and-suspenders guard)
            pregate_df = test_df[test_df["timestamp"] < delivery_ts].copy()
            # Also include train data so rolling features have sufficient history
            pregate_df = pd.concat([train_df, pregate_df], ignore_index=True)
            pregate_df = pregate_df[
                pregate_df["timestamp"] < delivery_ts
            ].drop_duplicates(subset=["timestamp", "node"]).sort_values(["timestamp", "node"])

            if pregate_df.empty:
                n_no_trade += 1
                continue

            # Build price anchors from training data for each target node
            price_anchors = {
                node: self.price_anchor_fn(node, delivery_ts, train_df)
                for node in self.forecasting_config.target_nodes
            }

            try:
                orders, _ = orchestrator.run(
                    feature_matrix=pregate_df,
                    price_anchors=price_anchors,
                    delivery_timestamp=delivery_ts,
                )
                if orders:
                    all_orders.extend(orders)
                else:
                    n_no_trade += 1
            except Exception as e:
                logger.warning(
                    "Order generation failed for delivery hour",
                    extra={"fold": fold_id, "ts": str(delivery_ts), "error": str(e)},
                )
                n_no_trade += 1

        predict_elapsed = _time.monotonic() - t_pred_start

        logger.info(
            "Fold complete",
            extra={
                "fold": fold_id,
                "n_orders": len(all_orders),
                "n_no_trade": n_no_trade,
                "fit_s": round(fit_elapsed, 1),
                "predict_s": round(predict_elapsed, 1),
            },
        )

        return FoldResult(
            split=split,
            orders=all_orders,
            n_no_trade=n_no_trade,
            fit_elapsed_s=fit_elapsed,
            predict_elapsed_s=predict_elapsed,
        )

    # -----------------------------------------------------------------------
    # Full validation loop
    # -----------------------------------------------------------------------

    def run(
        self,
        feature_matrix: pd.DataFrame,
        max_folds: Optional[int] = None,
        timestamp_col: str = "timestamp",
    ) -> list[FoldResult]:
        """
        Execute the complete walk-forward validation loop.

        Parameters
        ----------
        feature_matrix : Full Phase 1 feature matrix (all dates, all nodes).
        max_folds : Optionally cap the number of folds (useful for smoke tests).
        timestamp_col : Timestamp column name.

        Returns
        -------
        List of FoldResult — one per fold. Pass to BacktestEngine for
        market impact simulation and P&L settlement.
        """
        splits = self.splitter.generate(feature_matrix, timestamp_col)
        if max_folds is not None:
            splits = splits[:max_folds]

        logger.info(
            "Walk-forward validation starting",
            extra={"n_folds": len(splits), "mode": self.splitter.mode},
        )

        t0 = _time.monotonic()
        results: list[FoldResult] = []

        for split in splits:
            logger.info(
                "Running fold",
                extra={"fold": split.fold_id, "split": repr(split)},
            )
            train_df, test_df = self.splitter.split_dataframe(
                feature_matrix, split, timestamp_col
            )

            if len(train_df) < 168:   # < 1 week of hourly data
                logger.warning(
                    "Fold skipped — insufficient training data",
                    extra={"fold": split.fold_id, "train_rows": len(train_df)},
                )
                continue

            result = self._run_fold(split.fold_id, split, train_df, test_df)
            results.append(result)

        total_elapsed = _time.monotonic() - t0
        total_orders = sum(r.n_orders for r in results)
        successful_folds = sum(1 for r in results if r.success)

        logger.info(
            "Walk-forward validation complete",
            extra={
                "total_elapsed_s": round(total_elapsed, 1),
                "successful_folds": successful_folds,
                "total_folds": len(results),
                "total_orders": total_orders,
            },
        )
        return results
