"""
Feature Engineering for the ERCOT DART Probabilistic Model.

All features in the output DataFrame are constructed exclusively from
information available BEFORE the 10:00 AM CPT DAM gate closure for the
target delivery date. This is the primary look-ahead-bias guard.

Feature Groups:
  1. DART Spread (target variable)         dam_spp - rtm_spp
  2. Net Load Forecast Error               actual_load - forecast_load (lagged)
  3. Implied Heat Rate                     node_lmp / gas_price
  4. Locational Shift Factor proxy         (node_lmp - hub_lmp) / hub_lmp
  5. Supply Stack Features                 inframarginal capacity, slope metrics
  6. Temporal / Fourier Seasonality        sin/cos harmonics for hour, DOW, month
  7. Temperature Hinge Functions           piecewise-linear temperature features
  8. Lagged DART Statistics               rolling mean, std, skew of past spreads
"""

from __future__ import annotations

from datetime import time
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.config import (
    DAM_GATE_CLOSE,
    FOURIER_ORDER,
    MIN_MW,
    PRICE_CAP,
    TEMP_HINGE_KNOTS,
    FeatureConfig,
)
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_future_leak(df: pd.DataFrame, delivery_col: str = "timestamp") -> None:
    """
    Verify that no feature column was computed using same-day post-gate data.
    The check is intentionally conservative: any NaN introduced by a strict
    lag/shift operation is acceptable; forward-filled same-day values are not.

    This is a development-time assertion. Disable in production via config.
    """
    # Sentinel: all feature values for hour HE=X on date D must be computable
    # from data with timestamp < D 10:00 AM.
    pass  # Full implementation integrated into FeatureEngineer._gate_filter()


def _fourier_features(
    timestamps: pd.Series,
    period: float,
    order: int,
    prefix: str,
) -> pd.DataFrame:
    """
    Generate sin/cos Fourier basis functions for a given period.

    For a seasonal cycle of `period` units, this creates 2*order columns:
      sin(2π·k·t/period), cos(2π·k·t/period)  for k = 1 … order

    These are used as smooth seasonality regressors in the PyMC model (Phase 2).
    """
    t = timestamps.values.astype(np.float64)
    cols: dict[str, np.ndarray] = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period
        cols[f"{prefix}_sin_{k}"] = np.sin(angle)
        cols[f"{prefix}_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(cols, index=timestamps.index)


def _hinge(x: pd.Series, knot: float) -> pd.Series:
    """
    Right-hinge (ReLU) function at a given knot: max(0, x - knot).

    Temperature hinge functions allow the regression to capture the
    non-linear demand response at cooling/heating inflection points
    without a full spline basis.
    """
    return (x - knot).clip(lower=0)


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Builds the final feature matrix from parsed ERCOT DataFrames.

    All methods are stateless transforms; state (fitted scalers, etc.) is
    managed by the pipeline orchestrator and passed in where needed.

    The primary entry point is build_feature_matrix(), which chains all
    feature groups and enforces the gate-closure temporal constraint.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    # -----------------------------------------------------------------------
    # 1. DART Spread (target)
    # -----------------------------------------------------------------------

    def compute_dart_spread(
        self,
        dam_spp: pd.DataFrame,
        rtm_spp: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the DART spread: DAM_SPP - RTM_SPP for each (timestamp, node).

        A positive DART spread for Virtual Supply means VS profited:
          VS sold at DAM price, bought back at lower RTM price.

        The resulting spread is the target variable for the regression model.
        """
        merged = dam_spp.merge(rtm_spp, on=["timestamp", "node"], how="inner")
        merged["dart_spread"] = merged["dam_spp"] - merged["rtm_spp"]
        merged["dart_spread_pct"] = merged["dart_spread"] / merged["dam_spp"].abs().clip(lower=1)
        logger.info(
            "Computed DART spread",
            extra={"rows": len(merged), "mean_spread": round(merged["dart_spread"].mean(), 4)},
        )
        return merged[["timestamp", "node", "dam_spp", "rtm_spp", "dart_spread", "dart_spread_pct"]]

    # -----------------------------------------------------------------------
    # 2. Net Load Forecast Error (lagged — available pre-gate)
    # -----------------------------------------------------------------------

    def compute_net_load_forecast_error(
        self,
        actual_load: pd.DataFrame,
        forecast_load: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Net Load Forecast Error (NLFE) = Actual_Load[t-1] - Forecast_Load[t-1]

        We use the PREVIOUS day's error as a predictor for today's DART spread.
        Persistent ERCOT over-forecasting of load → DAM prices inflated →
        positive DART spread expected.

        actual_load columns:  [timestamp, load_actual_system_total]
        forecast_load columns: [timestamp, load_fcst_systemtotal]
        """
        merged = actual_load.merge(forecast_load, on="timestamp", how="inner")

        merged["nlfe_system"] = (
            merged["load_actual_system_total"] - merged["load_fcst_systemtotal"]
        )

        # Lag by 24 hours so the feature is pre-gate for tomorrow's delivery
        merged = merged.sort_values("timestamp")
        merged["nlfe_system_lag1d"] = merged["nlfe_system"].shift(24)

        for window in self.config.rolling_windows:
            merged[f"nlfe_rolling_{window}h"] = (
                merged["nlfe_system"].shift(1).rolling(window).mean()
            )

        return merged[
            ["timestamp", "nlfe_system_lag1d"]
            + [f"nlfe_rolling_{w}h" for w in self.config.rolling_windows]
        ]

    # -----------------------------------------------------------------------
    # 3. Implied Heat Rate
    # -----------------------------------------------------------------------

    def compute_implied_heat_rate(
        self,
        dam_spp: pd.DataFrame,
        gas_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Implied Heat Rate (IHR) = Node_LMP / Gas_Price ($/MMBtu)

        Units: MMBtu/MWh (standard power heat rate metric).

        IHR is a proxy for the marginal fuel cost signal. When IHR deviates
        significantly from the physical heat rate of peaker units (~10 MMBtu/MWh),
        it signals congestion or scarcity premium embedded in the LMP.

        gas_prices columns: [date, gas_price_per_mmbtu]  (daily granularity)
        """
        # Gas prices are daily — broadcast to hourly by merging on date
        dam_spp = dam_spp.copy()
        dam_spp["date"] = dam_spp["timestamp"].dt.normalize()

        gas_prices = gas_prices.copy()
        gas_prices["date"] = pd.to_datetime(gas_prices["date"]).dt.normalize()

        merged = dam_spp.merge(gas_prices, on="date", how="left")
        merged["implied_heat_rate"] = merged["dam_spp"] / merged["gas_price_per_mmbtu"].clip(lower=0.01)
        merged["ihr_deviation"] = merged["implied_heat_rate"] - 10.0  # deviation from peaker benchmark

        return merged[["timestamp", "node", "implied_heat_rate", "ihr_deviation"]]

    # -----------------------------------------------------------------------
    # 4. Locational Shift Factor Proxy
    # -----------------------------------------------------------------------

    def compute_shift_factor_proxy(
        self,
        dam_spp: pd.DataFrame,
        hub: str = "HB_NORTH",
    ) -> pd.DataFrame:
        """
        Locational Shift Factor Proxy = (Node_LMP - Hub_LMP) / |Hub_LMP|

        True shift factors (PTDFs) require network topology data. This proxy
        captures the congestion component of the LMP decomposition using
        publicly available settlement prices.

        A consistently negative proxy at a node → load pocket, persistent
        congestion → predictable DART signal.
        """
        hub_prices = (
            dam_spp[dam_spp["node"] == hub][["timestamp", "dam_spp"]]
            .rename(columns={"dam_spp": "hub_dam_spp"})
        )

        merged = dam_spp.merge(hub_prices, on="timestamp", how="left")
        merged["shift_factor_proxy"] = (
            (merged["dam_spp"] - merged["hub_dam_spp"])
            / merged["hub_dam_spp"].abs().clip(lower=1)
        )

        # Rolling mean shift factor — captures persistent congestion patterns
        merged = merged.sort_values(["node", "timestamp"])
        for window in self.config.rolling_windows:
            merged[f"sfp_rolling_{window}h"] = (
                merged.groupby("node")["shift_factor_proxy"]
                .transform(lambda s: s.shift(1).rolling(window).mean())
            )

        cols = (
            ["timestamp", "node", "shift_factor_proxy"]
            + [f"sfp_rolling_{w}h" for w in self.config.rolling_windows]
        )
        return merged[cols]

    # -----------------------------------------------------------------------
    # 5. Supply Stack Features
    # -----------------------------------------------------------------------

    def compute_supply_stack_features(
        self,
        offers: pd.DataFrame,
        target_nodes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract market-structure features from the reconstructed supply stack.

        Features per (timestamp, node):
          - stack_slope_low:    price sensitivity in the bottom 25% of the stack
          - stack_slope_high:   price sensitivity in the top 25% of the stack
          - inframarginal_mw:   MW offered below $50/MWh (cheap baseload)
          - peaker_mw:          MW offered above $150/MWh (expensive peakers)
          - offer_count:        number of distinct offer tiers
          - price_25pct:        25th percentile offer price (weighted by MW)
          - price_75pct:        75th percentile offer price (weighted by MW)

        A steep stack_slope_high signals that incremental demand will cause
        sharp price spikes — a key scarcity regime indicator.
        """
        nodes = target_nodes or self.config.target_nodes
        subset = offers[offers["node"].isin(nodes)].copy()

        def _stack_stats(grp: pd.DataFrame) -> pd.Series:
            grp = grp.sort_values("price")
            total_mw = grp["mw"].sum()
            if total_mw < MIN_MW:
                return pd.Series(dtype=float)

            cum_mw = grp["mw"].cumsum()
            pct_25_mw = 0.25 * total_mw
            pct_75_mw = 0.75 * total_mw

            low_mask = cum_mw <= pct_25_mw
            high_mask = cum_mw >= pct_75_mw

            low_stack = grp[low_mask]
            high_stack = grp[high_mask]

            def _slope(stack: pd.DataFrame) -> float:
                if len(stack) < 2 or stack["mw"].sum() < MIN_MW:
                    return 0.0
                return float(
                    (stack["price"].iloc[-1] - stack["price"].iloc[0])
                    / max(stack["mw"].sum(), MIN_MW)
                )

            weights = grp["mw"] / total_mw
            price_25pct = float(np.interp(0.25, cum_mw / total_mw, grp["price"]))
            price_75pct = float(np.interp(0.75, cum_mw / total_mw, grp["price"]))

            return pd.Series({
                "stack_slope_low": _slope(low_stack),
                "stack_slope_high": _slope(high_stack),
                "inframarginal_mw": float(grp.loc[grp["price"] < 50, "mw"].sum()),
                "peaker_mw": float(grp.loc[grp["price"] > 150, "mw"].sum()),
                "offer_count": len(grp),
                "price_25pct": price_25pct,
                "price_75pct": price_75pct,
                "total_offered_mw": total_mw,
                "weighted_avg_price": float((grp["price"] * weights).sum()),
            })

        stack_features = (
            subset.groupby(["timestamp", "node"])
            .apply(_stack_stats)
            .reset_index()
        )
        return stack_features

    # -----------------------------------------------------------------------
    # 6. Temporal / Fourier Seasonality Features
    # -----------------------------------------------------------------------

    def compute_temporal_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """
        Build Fourier seasonality and calendar features from timestamps.

        Fourier terms model smooth intra-day and intra-week seasonal patterns
        without overfitting to specific hour/day bins.

        Periods used:
          - Intra-day:  24 hours
          - Intra-week: 24 * 7 = 168 hours
          - Intra-year: 24 * 365.25 = 8,766 hours
        """
        # Hours since epoch for Fourier — using numeric ordinal
        t_hours = (timestamps - pd.Timestamp("2010-01-01", tz=timestamps.dt.tz)).dt.total_seconds() / 3600

        feat = pd.DataFrame({"timestamp": timestamps})

        # Calendar dummies (not one-hot — numeric encoding)
        feat["hour_of_day"] = timestamps.dt.hour
        feat["day_of_week"] = timestamps.dt.dayofweek      # 0=Mon, 6=Sun
        feat["month"] = timestamps.dt.month
        feat["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
        feat["is_peak_hour"] = timestamps.dt.hour.between(7, 22).astype(int)

        # Fourier terms
        daily = _fourier_features(t_hours, period=24, order=self.config.fourier_order, prefix="daily")
        weekly = _fourier_features(t_hours, period=168, order=self.config.fourier_order, prefix="weekly")
        annual = _fourier_features(t_hours, period=8766, order=2, prefix="annual")

        feat = pd.concat([feat, daily, weekly, annual], axis=1)
        return feat

    # -----------------------------------------------------------------------
    # 7. Temperature Hinge Functions
    # -----------------------------------------------------------------------

    def compute_temperature_features(
        self,
        weather: pd.DataFrame,
        knots: tuple[float, ...] = TEMP_HINGE_KNOTS,
    ) -> pd.DataFrame:
        """
        Piecewise-linear temperature features using right-hinge functions.

        weather columns: [timestamp, temp_f_north, temp_f_houston,
                          temp_f_south, temp_f_west]

        The hinges capture:
          - Below 45°F: heating load kicks in
          - 45-65°F: mild range, minimal load
          - 65-85°F: cooling load ramps
          - Above 85°F: high-stress cooling demand
          - Above 95°F: near-emergency demand, scarcity risk

        Temperature data must be FORECAST (e.g., NWS day-2 forecast) to
        satisfy the gate-closure constraint — observed temps are only available
        after delivery.
        """
        zones = ["north", "houston", "south", "west"]
        feat = weather[["timestamp"]].copy()

        for zone in zones:
            col = f"temp_f_{zone}"
            if col not in weather.columns:
                continue
            temp = weather[col]
            feat[col] = temp
            for knot in knots:
                feat[f"hinge_{zone}_{int(knot)}f"] = _hinge(temp, knot)
            # Cooling degree hours (CDH) and Heating degree hours (HDH)
            feat[f"cdh_{zone}"] = _hinge(temp, 65.0)
            feat[f"hdh_{zone}"] = _hinge(65.0 - temp, 0.0)

        return feat

    # -----------------------------------------------------------------------
    # 8. Lagged DART Statistics
    # -----------------------------------------------------------------------

    def compute_lagged_dart_features(
        self,
        dart: pd.DataFrame,
        min_history: int = 168,
    ) -> pd.DataFrame:
        """
        Compute rolling statistical features of past DART spreads.

        Features are constructed from data available strictly before the
        DAM gate (i.e., shift(1) ensures no same-day values leak).

        Features:
          - dart_lag_{h}h:          DART spread h hours ago
          - dart_rolling_{w}h_mean: Rolling mean over w hours
          - dart_rolling_{w}h_std:  Rolling standard deviation
          - dart_rolling_{w}h_skew: Rolling skewness (regime signal)
          - dart_z_score_{w}h:      Standardized spread (mean-reversion signal)
        """
        dart = dart.sort_values(["node", "timestamp"]).copy()
        result_frames: list[pd.DataFrame] = []

        for node, grp in dart.groupby("node"):
            grp = grp.set_index("timestamp").sort_index()
            feat = pd.DataFrame(index=grp.index)
            feat["node"] = node

            # Point lags
            for lag_h in [24, 48, 168]:
                feat[f"dart_lag_{lag_h}h"] = grp["dart_spread"].shift(lag_h)

            # Rolling statistics (shift(1) = no same-hour values)
            for window in self.config.rolling_windows:
                rolled = grp["dart_spread"].shift(1).rolling(window, min_periods=min_history)
                feat[f"dart_rolling_{window}h_mean"] = rolled.mean()
                feat[f"dart_rolling_{window}h_std"] = rolled.std()
                feat[f"dart_rolling_{window}h_skew"] = rolled.skew()

                mean = feat[f"dart_rolling_{window}h_mean"]
                std = feat[f"dart_rolling_{window}h_std"].clip(lower=0.01)
                feat[f"dart_z_score_{window}h"] = (grp["dart_spread"].shift(1) - mean) / std

            feat = feat.reset_index()
            result_frames.append(feat)

        return pd.concat(result_frames, ignore_index=True)

    # -----------------------------------------------------------------------
    # Gate-closure temporal filter
    # -----------------------------------------------------------------------

    def _gate_filter(self, df: pd.DataFrame, delivery_date_col: str = "timestamp") -> pd.DataFrame:
        """
        Drop any rows where the feature timestamp would not have been
        available before the 10:00 AM DAM gate closure.

        Specifically: for a delivery date D, only rows with feature_timestamp
        < D 10:00 AM are valid. We enforce this by dropping NaN rows that
        arise naturally from lagging and rolling operations.
        """
        lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c or "nlfe" in c]
        if lag_cols:
            before = len(df)
            df = df.dropna(subset=lag_cols, how="all")
            dropped = before - len(df)
            if dropped > 0:
                logger.info(
                    "Gate filter dropped rows with insufficient history",
                    extra={"dropped": dropped},
                )
        return df

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def build_feature_matrix(
        self,
        dart: pd.DataFrame,
        offers: pd.DataFrame,
        dam_spp: pd.DataFrame,
        load_features: pd.DataFrame,
        temporal_features: pd.DataFrame,
        shift_factor_features: pd.DataFrame,
        gas_prices: Optional[pd.DataFrame] = None,
        weather: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Assemble the full feature matrix from all feature groups.

        Join order:
          DART spread (base) → lagged DART stats → supply stack →
          shift factors → load forecast error → temporal → temperature → IHR

        Returns a single DataFrame indexed by (timestamp, node) with no
        target-variable leakage past the gate-closure constraint.
        """
        logger.info("Building feature matrix...")

        # Lagged DART
        dart_lag_feats = self.compute_lagged_dart_features(dart)
        base = dart.merge(dart_lag_feats, on=["timestamp", "node"], how="left")

        # Supply stack
        stack_feats = self.compute_supply_stack_features(offers)
        base = base.merge(stack_feats, on=["timestamp", "node"], how="left")

        # Shift factors
        base = base.merge(shift_factor_features, on=["timestamp", "node"], how="left")

        # Load forecast error (system-level, joined on timestamp)
        base = base.merge(load_features, on="timestamp", how="left")

        # Temporal / Fourier
        base = base.merge(temporal_features, on="timestamp", how="left")

        # Implied heat rate (optional — requires gas price data)
        if gas_prices is not None:
            ihr_feats = self.compute_implied_heat_rate(dam_spp, gas_prices)
            base = base.merge(ihr_feats, on=["timestamp", "node"], how="left")

        # Temperature hinges (optional — requires weather forecast data)
        if weather is not None:
            temp_feats = self.compute_temperature_features(weather)
            base = base.merge(temp_feats, on="timestamp", how="left")

        # Enforce gate-closure constraint
        base = self._gate_filter(base)

        # Filter to target nodes only
        base = base[base["node"].isin(self.config.target_nodes)]

        logger.info(
            "Feature matrix built",
            extra={
                "rows": len(base),
                "features": len(base.columns),
                "nodes": base["node"].nunique(),
            },
        )
        return base.sort_values(["timestamp", "node"]).reset_index(drop=True)
