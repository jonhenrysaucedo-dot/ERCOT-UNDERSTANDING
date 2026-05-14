"""
Backtest Performance Analytics and Top-Level BacktestEngine.

This module provides:
  1. BacktestResult — the unified output container aggregating all walk-forward
     fold results, clearing simulations, and settlement data.
  2. PerformanceAnalyzer — computes regime-conditional performance attribution,
     slippage decomposition, and model diagnostics.
  3. BacktestEngine — the top-level Phase 4 coordinator that chains:
       WalkForwardValidator → MarketImpactSimulator → RTMSettlementAggregator
       → VectorbtPortfolioBuilder → BacktestResult

Performance Attribution
-----------------------
We decompose P&L along three axes:
  1. Regime: How does Sharpe ratio differ across Normal / Scarcity / NegCong?
  2. Node: Which settlement points contribute the most alpha?
  3. Hour-of-Day: Does the strategy perform better during peak (7-22) or off-peak?

This attribution directly informs:
  - Phase 5 CUSUM drift detection (regime-specific monitoring)
  - Position limit adjustments per node / regime
  - Gate-closure window tuning

Benchmark
---------
The naive benchmark is a "constant VS" strategy that places equal-MW
Virtual Supply orders at every node for every hour, without any signal
filtering. Comparing the Kelly strategy to this benchmark isolates the
contribution of the probabilistic model from simple directional exposure.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.backtest.market_impact import ClearingResult, MarketImpactSimulator
from ercot_dart.backtest.settlement import (
    RTMSettlementAggregator,
    VectorbtPortfolioBuilder,
    VectorbtPortfolioResult,
)
from ercot_dart.backtest.walk_forward import FoldResult, WalkForwardValidator
from ercot_dart.etl.pipeline import ParsedDataStore
from ercot_dart.models.forecasting_engine import ForecastingConfig
from ercot_dart.trading.order import ERCOTOrder, OrchestratorConfig
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backtest Result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Unified output of the Phase 4 BacktestEngine.

    Attributes
    ----------
    fold_results : list[FoldResult]
        Walk-forward fold metadata and raw order lists.
    clearing_results : list[ClearingResult]
        Market impact simulation results for every simulated order.
    settled_trades : pd.DataFrame
        Fully settled trade records with realised P&L and impl. shortfall.
    portfolio : VectorbtPortfolioResult
        vectorbt portfolio object and derived metrics.
    elapsed_s : float
        Total wall-clock time for the full backtest run.
    """
    fold_results: list[FoldResult]
    clearing_results: list[ClearingResult]
    settled_trades: pd.DataFrame
    portfolio: VectorbtPortfolioResult
    elapsed_s: float

    @property
    def total_pnl(self) -> float:
        return self.portfolio.total_pnl

    @property
    def n_folds(self) -> int:
        return len(self.fold_results)

    @property
    def n_orders(self) -> int:
        return sum(r.n_orders for r in self.fold_results)

    @property
    def all_orders(self) -> list[ERCOTOrder]:
        return [o for r in self.fold_results for o in r.orders]

    def summary(self) -> pd.Series:
        fold_errors = sum(1 for r in self.fold_results if not r.success)
        avg_fill_rate = (
            float(np.mean([cr.fill_rate for cr in self.clearing_results]))
            if self.clearing_results else 0.0
        )
        avg_slippage = (
            float(np.mean([cr.slippage for cr in self.clearing_results]))
            if self.clearing_results else 0.0
        )
        s = self.portfolio.summary()
        s["n_folds"] = self.n_folds
        s["n_fold_errors"] = fold_errors
        s["n_orders_generated"] = self.n_orders
        s["n_orders_cleared"] = len(self.clearing_results)
        s["avg_fill_rate"] = round(avg_fill_rate, 4)
        s["avg_slippage"] = round(avg_slippage, 4)
        s["elapsed_s"] = round(self.elapsed_s, 1)
        return s

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if not self.settled_trades.empty:
            self.settled_trades.to_parquet(directory / "settled_trades.parquet", index=False)
        pd.DataFrame([r.to_series() for r in self.clearing_results]).to_parquet(
            directory / "clearing_results.parquet", index=False
        )
        self.summary().to_json(directory / "backtest_summary.json")
        logger.info("BacktestResult saved", extra={"directory": str(directory)})


# ---------------------------------------------------------------------------
# Performance Analyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """
    Decomposes BacktestResult P&L along regime, node, and time-of-day axes.

    All methods accept the settled_trades DataFrame and the clearing_results
    list as inputs, making them composable without re-running the backtest.
    """

    @staticmethod
    def regime_attribution(settled_trades: pd.DataFrame) -> pd.DataFrame:
        """
        P&L attribution by market regime at order submission.

        Requires settled_trades to have a 'regime_at_submission' column,
        populated from KellyResult.regime when the order was created.
        """
        if settled_trades.empty or "regime_at_submission" not in settled_trades.columns:
            return pd.DataFrame()

        regime_map = {0: "Normal", 1: "Scarcity", 2: "NegativeCongestion"}
        df = settled_trades.copy()
        df["regime_name"] = df["regime_at_submission"].map(regime_map)

        attribution = (
            df.groupby("regime_name")
            .agg(
                total_pnl=("realised_pnl", "sum"),
                mean_pnl_per_mwh=(
                    "realised_pnl",
                    lambda x: float(x.sum() / df.loc[x.index, "awarded_mw"].sum())
                    if df.loc[x.index, "awarded_mw"].sum() > 0 else 0.0
                ),
                n_trades=("order_id", "count"),
                win_rate=("realised_pnl", lambda x: (x > 0).mean()),
                mean_dart_spread=("dart_spread_realised", "mean"),
            )
            .reset_index()
        )
        return attribution

    @staticmethod
    def node_attribution(settled_trades: pd.DataFrame) -> pd.DataFrame:
        """P&L attribution by settlement point node."""
        if settled_trades.empty:
            return pd.DataFrame()
        return (
            settled_trades.groupby(["node", "direction"])
            .agg(
                total_pnl=("realised_pnl", "sum"),
                n_trades=("order_id", "count"),
                avg_awarded_mw=("awarded_mw", "mean"),
                avg_fill_rate=("fill_rate", "mean"),
                avg_slippage=("slippage", "mean"),
                win_rate=("realised_pnl", lambda x: (x > 0).mean()),
            )
            .reset_index()
            .sort_values("total_pnl", ascending=False)
        )

    @staticmethod
    def hourly_attribution(settled_trades: pd.DataFrame) -> pd.DataFrame:
        """P&L attribution by hour-of-day (peak vs. off-peak analysis)."""
        if settled_trades.empty:
            return pd.DataFrame()
        df = settled_trades.copy()
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["period"] = df["hour_of_day"].apply(
            lambda h: "On-Peak" if 7 <= h <= 22 else "Off-Peak"
        )
        return (
            df.groupby(["hour_of_day", "period"])
            .agg(
                mean_pnl=("realised_pnl", "mean"),
                total_pnl=("realised_pnl", "sum"),
                n_trades=("order_id", "count"),
                win_rate=("realised_pnl", lambda x: (x > 0).mean()),
            )
            .reset_index()
            .sort_values("hour_of_day")
        )

    @staticmethod
    def slippage_decomposition(clearing_results: list[ClearingResult]) -> pd.DataFrame:
        """
        Decompose total implementation shortfall into:
          - Slippage (price impact from market injection)
          - Partial fill (MW not awarded)
          - Node liquidity (ratio of our MW to total stack MW)
        """
        if not clearing_results:
            return pd.DataFrame()

        rows = []
        for cr in clearing_results:
            slippage_cost = cr.awarded_mw * abs(cr.slippage)
            unfilled_mw = cr.target_mw - cr.awarded_mw
            rows.append({
                "node": cr.node,
                "direction": cr.direction,
                "slippage_cost_usd": round(slippage_cost, 4),
                "unfilled_mw": round(unfilled_mw, 2),
                "fill_rate": round(cr.fill_rate, 4),
                "slippage_per_mwh": round(cr.slippage, 4),
            })

        df = pd.DataFrame(rows)
        return (
            df.groupby(["node", "direction"])
            .agg(
                total_slippage_cost=("slippage_cost_usd", "sum"),
                avg_slippage_per_mwh=("slippage_per_mwh", "mean"),
                avg_fill_rate=("fill_rate", "mean"),
                total_unfilled_mw=("unfilled_mw", "sum"),
                n_trades=("fill_rate", "count"),
            )
            .reset_index()
        )

    @staticmethod
    def benchmark_vs_naive(
        settled_trades: pd.DataFrame,
        naive_pnl_per_mwh: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compare the Kelly strategy P&L against a naive constant-VS benchmark.

        The naive benchmark earns `naive_pnl_per_mwh` (e.g., the mean
        historical DART spread) on every available MW-hour, regardless of
        signal. The information ratio measures how much the Kelly signal
        adds above this baseline.

        IR = (strategy_mean_pnl - benchmark_mean_pnl) / tracking_error
        """
        if settled_trades.empty:
            return pd.DataFrame()

        daily_strat = (
            settled_trades.assign(date=settled_trades["timestamp"].dt.normalize())
            .groupby("date")["realised_pnl"]
            .sum()
        )
        daily_benchmark = pd.Series(
            naive_pnl_per_mwh * settled_trades.groupby(
                settled_trades["timestamp"].dt.normalize()
            )["awarded_mw"].sum().values,
            index=daily_strat.index,
        )

        tracking_error = (daily_strat - daily_benchmark).std()
        excess = (daily_strat - daily_benchmark).mean()
        ir = excess / tracking_error if tracking_error > 1e-8 else 0.0

        return pd.DataFrame({
            "strategy_total_pnl": [daily_strat.sum()],
            "benchmark_total_pnl": [daily_benchmark.sum()],
            "alpha_pnl": [daily_strat.sum() - daily_benchmark.sum()],
            "information_ratio": [round(ir * np.sqrt(252), 4)],
            "strategy_daily_sharpe": [
                round(daily_strat.mean() / daily_strat.std() * np.sqrt(252), 4)
                if daily_strat.std() > 1e-8 else 0.0
            ],
        })


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Phase 4 top-level coordinator.

    Chains:
      WalkForwardValidator → MarketImpactSimulator → RTMSettlementAggregator
      → VectorbtPortfolioBuilder → BacktestResult

    Usage
    -----
        engine = BacktestEngine(
            forecasting_config=ForecastingConfig(mcmc_draws=500),
            orchestrator_config=OrchestratorConfig(max_position_mw=25.0),
        )
        result = engine.run(
            feature_matrix=feature_matrix,
            parsed_store=parsed_store,
            rtm_15min=rtm_df,
        )
        print(result.summary())
    """

    def __init__(
        self,
        forecasting_config: Optional[ForecastingConfig] = None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        min_train_days: int = 120,
        test_days: int = 7,
        window_mode: str = "expanding",
        init_cash: float = 1_000_000.0,
    ) -> None:
        from ercot_dart.backtest.walk_forward import WalkForwardSplitter, WindowMode

        self.forecasting_config = forecasting_config or ForecastingConfig()
        self.orchestrator_config = orchestrator_config or OrchestratorConfig()
        self.init_cash = init_cash

        self._splitter = WalkForwardSplitter(
            min_train_days=min_train_days,
            test_days=test_days,
            mode=WindowMode(window_mode),
        )
        self._validator = WalkForwardValidator(
            splitter=self._splitter,
            forecasting_config=self.forecasting_config,
            orchestrator_config=self.orchestrator_config,
        )
        self._impact_sim = MarketImpactSimulator()
        self._settlement = RTMSettlementAggregator()
        self._portfolio_builder = VectorbtPortfolioBuilder()

    def run(
        self,
        feature_matrix: pd.DataFrame,
        parsed_store: ParsedDataStore,
        rtm_15min: pd.DataFrame,
        max_folds: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> BacktestResult:
        """
        Execute the full Phase 4 backtest pipeline.

        Parameters
        ----------
        feature_matrix : Phase 1 feature matrix (all dates, all nodes).
        parsed_store : ParsedDataStore with dam_offers, dam_bids, dam_spp.
        rtm_15min : 15-minute SCED LMP DataFrame for RTM settlement.
        max_folds : Cap the number of walk-forward folds (for smoke tests).
        output_dir : If provided, saves BacktestResult artefacts to this path.

        Returns
        -------
        BacktestResult with full P&L and analytics.
        """
        t0 = _time.monotonic()
        logger.info("BacktestEngine starting")

        # ---- Step 1: Walk-forward validation loop ----------------------
        logger.info("Step 1: Walk-forward validation")
        fold_results = self._validator.run(feature_matrix, max_folds=max_folds)
        all_orders = [o for r in fold_results for o in r.orders]
        logger.info(
            "Walk-forward complete",
            extra={"n_folds": len(fold_results), "n_orders": len(all_orders)},
        )

        if not all_orders:
            logger.warning("No orders generated — returning empty BacktestResult")
            empty_portfolio = VectorbtPortfolioResult(
                portfolio=None,
                equity_curve=pd.Series(dtype=float),
                settled_trades=pd.DataFrame(),
                init_cash=self.init_cash,
            )
            return BacktestResult(
                fold_results=fold_results,
                clearing_results=[],
                settled_trades=pd.DataFrame(),
                portfolio=empty_portfolio,
                elapsed_s=_time.monotonic() - t0,
            )

        # ---- Step 2: Market impact simulation -------------------------
        logger.info("Step 2: Market impact simulation")
        clearing_results = self._impact_sim.simulate_all(
            orders=all_orders,
            offers=parsed_store.dam_offers,
            bids=parsed_store.dam_bids,
            dam_spp=parsed_store.dam_spp,
        )

        # Attach clearing results back to orders (settle)
        clearing_by_id = {cr.order_id: cr for cr in clearing_results}
        settled_orders = []
        for order in all_orders:
            cr = clearing_by_id.get(order.order_id)
            if cr is None or cr.awarded_mw <= 0:
                continue
            settled_orders.append(order.settle(
                awarded_mw=cr.awarded_mw,
                awarded_price=cr.true_clearing_price,
                rtm_price=0.0,    # placeholder — filled in next step
            ))

        # ---- Step 3: RTM settlement aggregation ----------------------
        logger.info("Step 3: RTM settlement aggregation")
        nodes = list(set(o.node for o in all_orders))
        rtm_hourly = self._settlement.aggregate(rtm_15min, nodes=nodes)
        settled_trades = self._settlement.settle_clearing_results(
            clearing_results, rtm_hourly
        )

        # Enrich settled_trades with regime metadata from Kelly results
        if not settled_trades.empty:
            regime_map = {
                o.order_id: o.regime_at_submission for o in all_orders
            }
            settled_trades["regime_at_submission"] = (
                settled_trades["order_id"].map(regime_map)
            )

        # ---- Step 4: vectorbt portfolio construction -----------------
        logger.info("Step 4: Building vectorbt portfolio")
        portfolio = self._portfolio_builder.build(settled_trades, self.init_cash)

        # ---- Step 5: Assemble result --------------------------------
        elapsed = _time.monotonic() - t0
        result = BacktestResult(
            fold_results=fold_results,
            clearing_results=clearing_results,
            settled_trades=settled_trades,
            portfolio=portfolio,
            elapsed_s=elapsed,
        )

        logger.info(
            "BacktestEngine complete",
            extra={
                "elapsed_s": round(elapsed, 1),
                "total_pnl": round(result.total_pnl, 2),
                "n_orders": result.n_orders,
                "sharpe": round(portfolio.sharpe_ratio(), 4),
            },
        )

        if output_dir is not None:
            result.save(Path(output_dir))

        return result
