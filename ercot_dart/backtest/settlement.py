"""
RTM Settlement Aggregator and vectorbt Portfolio Builder.

ERCOT DAM awards are hourly; RTM settlement prices are 15-minute intervals.
This module bridges the two:

  1. RTM Price Aggregation
     The 15-minute Locational Marginal Prices (from SCED) must be aggregated
     to hourly prices to match the granularity of DAM awards. ERCOT settles
     virtual positions using the simple arithmetic mean of the four 15-minute
     RTM prices within each DAM settlement interval:

         RTM_hourly = (1/4) × Σ_{i=1}^{4} RTM_{15min,i}

     This is the official ERCOT settlement formula per PUCT protocols.

  2. vectorbt Portfolio Construction
     We use vectorbt's Portfolio.from_orders() interface to:
       - Represent each awarded virtual position as a long/short entry
       - Compute the MTM P&L at settlement using the RTM price
       - Aggregate into an equity curve and compute risk metrics

     vectorbt's vectorised engine handles thousands of trades across
     multiple nodes without looping, making it far faster than a
     pandas-based settlement loop.

  3. Implementation Shortfall Tracking
     The implementation shortfall is the gap between the theoretical P&L
     (computed at historical DAM prices, zero slippage) and the realised P&L
     (computed at true clearing prices with market impact). This metric is
     monitored by the CUSUM drift detector in Phase 5.

     Implementation Shortfall = Theoretical P&L - Realised P&L
     = awarded_mw × (historical_dam_spp - rtm_spp)
       - awarded_mw × (true_clearing_price - rtm_spp)
     = awarded_mw × (historical_dam_spp - true_clearing_price)
     = awarded_mw × (-slippage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.backtest.market_impact import ClearingResult
from ercot_dart.trading.kelly import TradeDirection
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RTM Settlement Aggregator
# ---------------------------------------------------------------------------

class RTMSettlementAggregator:
    """
    Aggregates 15-minute SCED RTM prices to hourly settlement prices.

    ERCOT's official settlement formula:
        hourly_rtm_spp = mean(15min_rtm_spp_HE1, ..., 15min_rtm_spp_HE4)

    where HE = Hour Ending; all four intervals fall within the same
    delivery hour.

    The resulting hourly RTM prices are joined against ClearingResults
    to compute realised P&L for each virtual trade.
    """

    def aggregate(
        self,
        rtm_15min: pd.DataFrame,
        nodes: Optional[list[str]] = None,
        timestamp_col: str = "timestamp",
        lmp_col: str = "rtm_15min_lmp",
        node_col: str = "node",
    ) -> pd.DataFrame:
        """
        Aggregate 15-minute RTM prices to hourly means per node.

        Parameters
        ----------
        rtm_15min : DataFrame with columns [timestamp, node, rtm_15min_lmp]
            Output of SCEDSettlementPriceParser (before hourly aggregation).
            Timestamps are the START of each 15-minute interval.
        nodes : Optional filter list. If None, aggregates all nodes.

        Returns
        -------
        DataFrame with columns [timestamp, node, rtm_spp]
            timestamp = start of the hourly interval (HourEnding - 1 hour)
        """
        df = rtm_15min.copy()
        if nodes is not None:
            df = df[df[node_col].isin(nodes)]

        # Floor timestamp to the hour to group 15-min intervals
        df["hour"] = df[timestamp_col].dt.floor("1h")

        hourly = (
            df.groupby(["hour", node_col])[lmp_col]
            .mean()
            .reset_index()
            .rename(columns={"hour": "timestamp", lmp_col: "rtm_spp"})
        )

        logger.info(
            "RTM aggregation complete",
            extra={"hourly_rows": len(hourly), "nodes": hourly[node_col].nunique()},
        )
        return hourly

    def settle_clearing_results(
        self,
        clearing_results: list[ClearingResult],
        rtm_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join ClearingResults with hourly RTM prices and compute P&L.

        P&L formula:
          Virtual Supply:  P&L = awarded_mw × (true_clearing_price - rtm_spp)
          Virtual Demand:  P&L = awarded_mw × (rtm_spp - true_clearing_price)

        Also computes:
          theoretical_pnl : P&L using historical (unadjusted) clearing price
          implementation_shortfall : theoretical_pnl - realised_pnl
                                   = awarded_mw × (-slippage)

        Returns
        -------
        DataFrame with one row per settled trade.
        """
        if not clearing_results:
            return pd.DataFrame()

        cr_df = pd.DataFrame([r.to_series() for r in clearing_results])

        rtm_lookup = rtm_hourly.set_index(["timestamp", "node"])["rtm_spp"]

        settled_rows = []
        for _, row in cr_df.iterrows():
            key = (row["timestamp"], row["node"])
            rtm_price = rtm_lookup.get(key)

            if rtm_price is None or np.isnan(rtm_price):
                logger.warning(
                    "RTM price not found for settlement",
                    extra={"node": row["node"], "ts": str(row["timestamp"])},
                )
                continue

            awarded_mw = row["awarded_mw"]
            true_price = row["true_clearing_price"]
            hist_price = row["historical_price"]
            direction = row["direction"]

            if direction == TradeDirection.VIRTUAL_SUPPLY:
                realised_pnl = awarded_mw * (true_price - rtm_price)
                theoretical_pnl = awarded_mw * (hist_price - rtm_price)
            else:
                realised_pnl = awarded_mw * (rtm_price - true_price)
                theoretical_pnl = awarded_mw * (rtm_price - hist_price)

            impl_shortfall = theoretical_pnl - realised_pnl

            settled_rows.append({
                "order_id": row["order_id"],
                "timestamp": row["timestamp"],
                "node": row["node"],
                "direction": direction,
                "awarded_mw": awarded_mw,
                "dam_clearing_price": true_price,
                "rtm_spp": rtm_price,
                "dart_spread_realised": (
                    true_price - rtm_price if direction == TradeDirection.VIRTUAL_SUPPLY
                    else rtm_price - true_price
                ),
                "realised_pnl": round(realised_pnl, 4),
                "theoretical_pnl": round(theoretical_pnl, 4),
                "implementation_shortfall": round(impl_shortfall, 4),
                "slippage": row["slippage"],
                "fill_rate": row["fill_rate"],
            })

        if not settled_rows:
            return pd.DataFrame()

        settled = pd.DataFrame(settled_rows)
        logger.info(
            "Settlement complete",
            extra={
                "n_settled": len(settled),
                "total_pnl": round(float(settled["realised_pnl"].sum()), 2),
                "mean_impl_shortfall": round(
                    float(settled["implementation_shortfall"].mean()), 4
                ),
            },
        )
        return settled


# ---------------------------------------------------------------------------
# vectorbt Portfolio Builder
# ---------------------------------------------------------------------------

class VectorbtPortfolioBuilder:
    """
    Constructs a vectorbt Portfolio from settled DART trade records.

    vectorbt's Portfolio.from_orders() represents each virtual trade as:
      - A SIZE (in MWh) order at ENTRY_PRICE (DAM clearing price)
      - A SIZE (in MWh) order at EXIT_PRICE (RTM settlement price)

    For Virtual Supply:
      - ENTRY: sell awarded_mw MWh at dam_clearing_price  (direction = SELL)
      - EXIT:  buy  awarded_mw MWh at rtm_spp            (direction = BUY)
      Net cash = awarded_mw × (dam_clearing_price - rtm_spp) = DART spread × MWh

    For Virtual Demand:
      - ENTRY: buy  awarded_mw MWh at dam_clearing_price  (direction = BUY)
      - EXIT:  sell awarded_mw MWh at rtm_spp            (direction = SELL)
      Net cash = awarded_mw × (rtm_spp - dam_clearing_price)

    The portfolio is indexed by timestamp and grouped by node so that
    vectorbt can compute per-node and aggregate performance metrics.
    """

    def build(
        self,
        settled_trades: pd.DataFrame,
        init_cash: float = 1_000_000.0,
        freq: str = "1h",
    ) -> "VectorbtPortfolioResult":
        """
        Build a vectorbt Portfolio from the settled trades DataFrame.

        Parameters
        ----------
        settled_trades : Output of RTMSettlementAggregator.settle_clearing_results()
        init_cash : Starting capital in USD (used for return normalisation).
        freq : Time frequency for the equity curve index.

        Returns
        -------
        VectorbtPortfolioResult with the portfolio object and summary stats.
        """
        try:
            import vectorbt as vbt
        except ImportError:
            logger.warning("vectorbt not installed — returning raw P&L DataFrame only")
            return VectorbtPortfolioResult(
                portfolio=None,
                equity_curve=self._manual_equity_curve(settled_trades, init_cash),
                settled_trades=settled_trades,
                init_cash=init_cash,
            )

        if settled_trades.empty:
            return VectorbtPortfolioResult(
                portfolio=None,
                equity_curve=pd.Series(dtype=float),
                settled_trades=settled_trades,
                init_cash=init_cash,
            )

        # Build the order records for vectorbt
        # Each DART trade is two synthetic orders: entry (DAM) + exit (RTM)
        order_records = self._build_order_records(settled_trades)

        # Create a price series for all unique timestamps
        all_timestamps = pd.date_range(
            start=settled_trades["timestamp"].min(),
            end=settled_trades["timestamp"].max(),
            freq=freq,
        )

        # Use DAM clearing price as the "price" series for entry
        # and RTM price as the "close" series for exit marking
        # We use a combined price column: DAM for open, RTM for close
        price_df = settled_trades.pivot_table(
            index="timestamp",
            columns="node",
            values="dam_clearing_price",
            aggfunc="mean",
        ).reindex(all_timestamps).ffill()

        try:
            portfolio = vbt.Portfolio.from_orders(
                close=price_df,
                size=order_records.get("size"),
                price=order_records.get("price"),
                direction=order_records.get("direction"),
                init_cash=init_cash,
                freq=freq,
            )
        except Exception as e:
            logger.warning(
                "vectorbt Portfolio construction failed — falling back to manual equity curve",
                extra={"error": str(e)},
            )
            portfolio = None

        equity_curve = self._manual_equity_curve(settled_trades, init_cash)

        result = VectorbtPortfolioResult(
            portfolio=portfolio,
            equity_curve=equity_curve,
            settled_trades=settled_trades,
            init_cash=init_cash,
        )
        logger.info(
            "Portfolio built",
            extra={
                "total_pnl": round(float(settled_trades["realised_pnl"].sum()), 2),
                "n_trades": len(settled_trades),
                "final_equity": round(float(equity_curve.iloc[-1]), 2) if len(equity_curve) > 0 else 0,
            },
        )
        return result

    def _build_order_records(self, settled_trades: pd.DataFrame) -> dict:
        """
        Translate settled trades into vectorbt order record arrays.

        Returns a dict of numpy arrays: {size, price, direction}
        aligned to the timestamp index of the portfolio.
        """
        # Two orders per trade (entry + exit), interleaved
        n = len(settled_trades)
        sizes = np.empty(n * 2)
        prices = np.empty(n * 2)
        directions = np.empty(n * 2, dtype=object)
        timestamps = []

        for i, row in settled_trades.reset_index(drop=True).iterrows():
            ts = row["timestamp"]
            mw = row["awarded_mw"]
            dam_p = row["dam_clearing_price"]
            rtm_p = row["rtm_spp"]
            d = row["direction"]

            if d == TradeDirection.VIRTUAL_SUPPLY:
                # Entry: SELL at DAM; Exit: BUY at RTM
                sizes[2 * i] = mw
                prices[2 * i] = dam_p
                directions[2 * i] = "sell"
                sizes[2 * i + 1] = mw
                prices[2 * i + 1] = rtm_p
                directions[2 * i + 1] = "buy"
            else:
                # Entry: BUY at DAM; Exit: SELL at RTM
                sizes[2 * i] = mw
                prices[2 * i] = dam_p
                directions[2 * i] = "buy"
                sizes[2 * i + 1] = mw
                prices[2 * i + 1] = rtm_p
                directions[2 * i + 1] = "sell"

            timestamps.extend([ts, ts])

        return {"size": sizes, "price": prices, "direction": directions,
                "timestamps": timestamps}

    @staticmethod
    def _manual_equity_curve(
        settled_trades: pd.DataFrame,
        init_cash: float,
    ) -> pd.Series:
        """
        Compute a cumulative equity curve from settled P&L without vectorbt.
        Sorted by timestamp, indexed by timestamp.
        """
        if settled_trades.empty:
            return pd.Series(dtype=float)

        pnl_ts = (
            settled_trades.groupby("timestamp")["realised_pnl"]
            .sum()
            .sort_index()
        )
        equity = init_cash + pnl_ts.cumsum()
        return equity


# ---------------------------------------------------------------------------
# Portfolio result container
# ---------------------------------------------------------------------------

@dataclass
class VectorbtPortfolioResult:
    """
    Container for the vectorbt Portfolio and derived analytics.

    If vectorbt is unavailable, portfolio is None and equity_curve is
    computed manually from the settled_trades P&L.
    """
    portfolio: Optional[object]          # vbt.Portfolio or None
    equity_curve: pd.Series
    settled_trades: pd.DataFrame
    init_cash: float

    @property
    def total_pnl(self) -> float:
        return float(self.settled_trades["realised_pnl"].sum()) if not self.settled_trades.empty else 0.0

    @property
    def total_return_pct(self) -> float:
        return (self.total_pnl / self.init_cash) * 100 if self.init_cash > 0 else 0.0

    @property
    def n_trades(self) -> int:
        return len(self.settled_trades)

    @property
    def win_rate(self) -> float:
        if self.settled_trades.empty:
            return 0.0
        return float((self.settled_trades["realised_pnl"] > 0).mean())

    def pnl_by_node(self) -> pd.DataFrame:
        if self.settled_trades.empty:
            return pd.DataFrame()
        return (
            self.settled_trades.groupby(["node", "direction"])["realised_pnl"]
            .agg(["sum", "mean", "count"])
            .rename(columns={"sum": "total_pnl", "mean": "avg_pnl", "count": "n_trades"})
            .reset_index()
        )

    def monthly_pnl(self) -> pd.DataFrame:
        if self.settled_trades.empty:
            return pd.DataFrame()
        df = self.settled_trades.copy()
        df["month"] = df["timestamp"].dt.to_period("M")
        return (
            df.groupby("month")["realised_pnl"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "monthly_pnl", "count": "n_trades"})
            .reset_index()
        )

    def implementation_shortfall_total(self) -> float:
        if "implementation_shortfall" not in self.settled_trades.columns:
            return 0.0
        return float(self.settled_trades["implementation_shortfall"].sum())

    def sharpe_ratio(self, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
        """
        Annualised Sharpe ratio computed from the hourly P&L series.

        Uses hourly granularity: periods_per_year = 8760.
        """
        if self.settled_trades.empty:
            return 0.0
        hourly_pnl = (
            self.settled_trades.groupby("timestamp")["realised_pnl"].sum()
        )
        if hourly_pnl.std() < 1e-8:
            return 0.0
        sharpe = (
            (hourly_pnl.mean() - risk_free_rate / periods_per_year)
            / hourly_pnl.std()
            * np.sqrt(periods_per_year)
        )
        return float(sharpe)

    def max_drawdown(self) -> float:
        """Maximum drawdown of the cumulative P&L equity curve."""
        if self.equity_curve.empty:
            return 0.0
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        return float(drawdown.min())

    def calmar_ratio(self) -> float:
        """Annualised return / |Max Drawdown|."""
        mdd = abs(self.max_drawdown())
        if mdd < 1e-8:
            return 0.0
        n_hours = len(self.equity_curve)
        ann_return = self.total_return_pct / 100 * (8760 / max(n_hours, 1))
        return ann_return / mdd

    def summary(self) -> pd.Series:
        return pd.Series({
            "total_pnl_usd": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "sharpe_ratio": round(self.sharpe_ratio(), 4),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 4),
            "calmar_ratio": round(self.calmar_ratio(), 4),
            "impl_shortfall_usd": round(self.implementation_shortfall_total(), 2),
        })
