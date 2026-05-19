"""
CUSUM Change-Point Detection for DART Model Drift Monitoring.

The CUSUM (Cumulative SUM) control chart is the gold-standard sequential
hypothesis test for detecting a persistent shift in the mean of a process.
Unlike point-in-time tests (e.g., rolling Z-score), CUSUM accumulates
evidence across time, making it sensitive to small sustained shifts
(e.g., gradual model decay) while remaining robust to transient noise.

Mathematical Specification
--------------------------
Let x_t be the monitored metric at time t. Define:

    Upper CUSUM: S_t^+ = max(0, S_{t-1}^+ + (x_t - μ_0 - k))
    Lower CUSUM: S_t^- = max(0, S_{t-1}^- - (x_t - μ_0 + k))

where:
    μ_0 : in-control mean (the "target" value when the model is healthy)
    k   : allowance parameter = δ / 2  (half the detectable shift size δ)
    δ   : the minimum shift in σ units we want to detect reliably

Alarm condition:
    S_t^+ > H  → upward shift detected (mean has increased above μ_0 + δ)
    S_t^- > H  → downward shift detected (mean has decreased below μ_0 - δ)

Decision interval H is chosen to control the in-control Average Run Length
(ARL₀). For H = 5 and k = 0.5, ARL₀ ≈ 465 observations, meaning we expect
one false alarm per 465 in-control observations.

Monitored Metrics
-----------------
We run independent CUSUM detectors on five metrics, each with a calibrated
(μ_0, k, H) triplet derived from the Phase 4 backtest statistics:

1. Implementation Shortfall ($/MWh):
   μ_0 = 0 (zero IS = no market impact in-control)
   Upward drift → our bids are moving prices against us → reduce position
   Trigger: retrain model, reduce max_position_mw by 20%

2. Win Rate (daily rolling 7-day):
   μ_0 = backtest_win_rate (e.g., 0.58)
   Downward drift → model is losing directional accuracy
   Trigger: suspend trading, retrain immediately

3. DART Forecast Bias (realised_dart - forecast_mu):
   μ_0 = 0 (unbiased forecast in-control)
   Upward/downward drift → systematic over/under-prediction → retrain
   Trigger: retrain model with fresh prior

4. Fill Rate (awarded_mw / target_mw):
   μ_0 = backtest_fill_rate (e.g., 0.85)
   Downward drift → our offers are priced too aggressively → widen curve
   Trigger: adjust tier_curve_generator sigma multiplier

5. Regime Prediction Stability (entropy of regime probabilities):
   μ_0 = backtest_regime_entropy
   Upward drift → HMM is becoming uncertain → retrain HMM
   Trigger: retrain HMM only (not full MCMC)

Retraining Actions
------------------
Each alarm triggers one of three retraining actions:
  RETRAIN_FULL   : Refit HMM + GARCH + MCMC on an expanded training window
  RETRAIN_HMM    : Refit HMM only (fast, < 30 seconds)
  ADJUST_PARAMS  : Modify KellySizer/TierCurveGenerator parameters in-place
  SUSPEND        : Halt all trading at the affected node until manual review
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Retraining action enum
# ---------------------------------------------------------------------------

class RetrainAction(str, Enum):
    RETRAIN_FULL = "RETRAIN_FULL"
    RETRAIN_HMM = "RETRAIN_HMM"
    ADJUST_PARAMS = "ADJUST_PARAMS"
    SUSPEND = "SUSPEND"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# CUSUM state for a single metric
# ---------------------------------------------------------------------------

@dataclass
class CUSUMState:
    """
    State of one CUSUM detector instance.

    Attributes
    ----------
    metric_name : str
    s_plus : float     Upper CUSUM statistic (detects upward shift)
    s_minus : float    Lower CUSUM statistic (detects downward shift)
    mu_0 : float       In-control target mean
    k : float          Allowance (half the detectable shift)
    h : float          Decision interval (alarm threshold)
    t : int            Number of observations processed
    alarm_up : bool    Upward alarm active
    alarm_down : bool  Downward alarm active
    last_alarm_t : int Observation count at which last alarm fired
    """
    metric_name: str
    s_plus: float = 0.0
    s_minus: float = 0.0
    mu_0: float = 0.0
    k: float = 0.5
    h: float = 5.0
    t: int = 0
    alarm_up: bool = False
    alarm_down: bool = False
    last_alarm_t: int = -1

    # History for plotting (bounded ring buffer)
    _history_x: list[float] = field(default_factory=list, repr=False)
    _history_splus: list[float] = field(default_factory=list, repr=False)
    _history_sminus: list[float] = field(default_factory=list, repr=False)
    _history_timestamps: list[pd.Timestamp] = field(default_factory=list, repr=False)
    _max_history: int = field(default=1000, repr=False)

    @property
    def is_alarm(self) -> bool:
        return self.alarm_up or self.alarm_down

    @property
    def alarm_direction(self) -> str:
        if self.alarm_up:
            return "UP"
        if self.alarm_down:
            return "DOWN"
        return "NONE"

    def update(self, x: float, timestamp: Optional[pd.Timestamp] = None) -> "CUSUMState":
        """
        Process one new observation x and update the CUSUM statistics.

        The update equations are:
            S_t^+ = max(0, S_{t-1}^+ + (x - μ_0 - k))
            S_t^- = max(0, S_{t-1}^- - (x - μ_0 + k))

        Alarms are set when S_t^+ > H or S_t^- > H.
        """
        self.t += 1
        self.s_plus = max(0.0, self.s_plus + (x - self.mu_0 - self.k))
        self.s_minus = max(0.0, self.s_minus - (x - self.mu_0 + self.k))

        self.alarm_up = self.s_plus > self.h
        self.alarm_down = self.s_minus > self.h

        if self.is_alarm and self.last_alarm_t < self.t:
            self.last_alarm_t = self.t
            logger.warning(
                "CUSUM alarm triggered",
                extra={
                    "metric": self.metric_name,
                    "direction": self.alarm_direction,
                    "s_plus": round(self.s_plus, 4),
                    "s_minus": round(self.s_minus, 4),
                    "h": self.h,
                    "t": self.t,
                },
            )

        # Ring-buffer history
        self._history_x.append(x)
        self._history_splus.append(self.s_plus)
        self._history_sminus.append(self.s_minus)
        if timestamp is not None:
            self._history_timestamps.append(timestamp)
        if len(self._history_x) > self._max_history:
            self._history_x.pop(0)
            self._history_splus.pop(0)
            self._history_sminus.pop(0)
            if self._history_timestamps:
                self._history_timestamps.pop(0)

        return self

    def reset(self) -> "CUSUMState":
        """Reset CUSUM statistics after a model retraining event."""
        self.s_plus = 0.0
        self.s_minus = 0.0
        self.alarm_up = False
        self.alarm_down = False
        logger.info("CUSUM reset after retraining", extra={"metric": self.metric_name})
        return self

    def history_dataframe(self) -> pd.DataFrame:
        """Return CUSUM history as a DataFrame for plotting."""
        n = len(self._history_x)
        df = pd.DataFrame({
            "obs_index": range(max(0, self.t - n), self.t),
            "x": self._history_x,
            "s_plus": self._history_splus,
            "s_minus": self._history_sminus,
            "h": self.h,
        })
        if self._history_timestamps and len(self._history_timestamps) == n:
            df["timestamp"] = self._history_timestamps
        df["alarm"] = (df["s_plus"] > self.h) | (df["s_minus"] > self.h)
        return df


# ---------------------------------------------------------------------------
# Metric specification
# ---------------------------------------------------------------------------

@dataclass
class MetricSpec:
    """
    Specification for a single CUSUM-monitored metric.

    Parameters
    ----------
    name : str          Human-readable metric name
    mu_0 : float        In-control target mean
    sigma_0 : float     In-control standard deviation (used to set k)
    delta_sigma : float Minimum detectable shift in σ units (default 1σ)
    h_sigma : float     Decision interval in σ units (default 5σ)
    action_up : RetrainAction  Action if upward drift detected
    action_down : RetrainAction  Action if downward drift detected
    """
    name: str
    mu_0: float
    sigma_0: float
    delta_sigma: float = 1.0
    h_sigma: float = 5.0
    action_up: RetrainAction = RetrainAction.RETRAIN_FULL
    action_down: RetrainAction = RetrainAction.RETRAIN_FULL

    @property
    def k(self) -> float:
        """Allowance = δ/2 in original units."""
        return self.delta_sigma * self.sigma_0 / 2.0

    @property
    def h(self) -> float:
        """Decision interval in original units."""
        return self.h_sigma * self.sigma_0


# ---------------------------------------------------------------------------
# Default metric specifications (calibrated from Phase 4 backtest)
# ---------------------------------------------------------------------------

def default_metric_specs(backtest_stats: Optional[dict] = None) -> list[MetricSpec]:
    """
    Return default CUSUM metric specifications.

    If backtest_stats is provided, μ_0 values are set from actual backtest
    performance rather than theoretical defaults.

    backtest_stats keys: win_rate, fill_rate, impl_shortfall_per_mwh,
                         dart_forecast_bias, regime_entropy
    """
    s = backtest_stats or {}
    return [
        MetricSpec(
            name="implementation_shortfall",
            mu_0=s.get("impl_shortfall_per_mwh", 0.0),
            sigma_0=s.get("impl_shortfall_std", 1.5),
            delta_sigma=1.0,
            h_sigma=5.0,
            action_up=RetrainAction.ADJUST_PARAMS,   # IS rising → reduce size
            action_down=RetrainAction.NONE,            # Negative IS = we're helping ourselves
        ),
        MetricSpec(
            name="win_rate_7d",
            mu_0=s.get("win_rate", 0.55),
            sigma_0=s.get("win_rate_std", 0.05),
            delta_sigma=1.0,
            h_sigma=4.0,
            action_up=RetrainAction.NONE,
            action_down=RetrainAction.SUSPEND,         # Win rate falling → suspend + retrain
        ),
        MetricSpec(
            name="dart_forecast_bias",
            mu_0=s.get("dart_forecast_bias", 0.0),
            sigma_0=s.get("dart_forecast_bias_std", 2.0),
            delta_sigma=1.0,
            h_sigma=5.0,
            action_up=RetrainAction.RETRAIN_FULL,
            action_down=RetrainAction.RETRAIN_FULL,
        ),
        MetricSpec(
            name="fill_rate",
            mu_0=s.get("fill_rate", 0.85),
            sigma_0=s.get("fill_rate_std", 0.10),
            delta_sigma=1.0,
            h_sigma=5.0,
            action_up=RetrainAction.NONE,
            action_down=RetrainAction.ADJUST_PARAMS,   # Fill rate falling → widen curve
        ),
        MetricSpec(
            name="regime_entropy",
            mu_0=s.get("regime_entropy", 0.9),
            sigma_0=s.get("regime_entropy_std", 0.15),
            delta_sigma=1.0,
            h_sigma=5.0,
            action_up=RetrainAction.RETRAIN_HMM,       # HMM uncertain → refit HMM
            action_down=RetrainAction.NONE,
        ),
    ]


# ---------------------------------------------------------------------------
# Drift alarm record
# ---------------------------------------------------------------------------

@dataclass
class DriftAlarm:
    """
    A single drift alarm event.

    Attributes
    ----------
    metric_name : str
    direction : str         "UP" or "DOWN"
    s_value : float         CUSUM statistic at alarm time
    recommended_action : RetrainAction
    timestamp : pd.Timestamp  Wall-clock time of the alarm
    node : str              Node where the alarm originated (or "SYSTEM")
    obs_count : int         Number of observations processed before alarm
    """
    metric_name: str
    direction: str
    s_value: float
    recommended_action: RetrainAction
    timestamp: pd.Timestamp
    node: str = "SYSTEM"
    obs_count: int = 0
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "metric": self.metric_name,
            "direction": self.direction,
            "s_value": round(self.s_value, 4),
            "action": self.recommended_action.value,
            "timestamp": str(self.timestamp),
            "node": self.node,
            "obs_count": self.obs_count,
        }


# ---------------------------------------------------------------------------
# CUSUM Monitor (one node or system-wide)
# ---------------------------------------------------------------------------

class CUSUMMonitor:
    """
    Multi-metric CUSUM monitor for one settlement point or system-wide.

    Maintains an independent CUSUMState for each monitored metric and
    fires DriftAlarm events whenever a threshold is crossed.

    Usage
    -----
        monitor = CUSUMMonitor("HB_NORTH", metric_specs)
        # After each settlement batch:
        monitor.update("implementation_shortfall", is_value, timestamp)
        monitor.update("win_rate_7d", win_rate, timestamp)
        alarms = monitor.check_alarms()
        if alarms:
            handle_retraining(alarms)
    """

    def __init__(
        self,
        node: str,
        metric_specs: Optional[list[MetricSpec]] = None,
    ) -> None:
        self.node = node
        self._specs: dict[str, MetricSpec] = {}
        self._states: dict[str, CUSUMState] = {}
        self._alarms: list[DriftAlarm] = []

        for spec in (metric_specs or default_metric_specs()):
            self._register_metric(spec)

    def _register_metric(self, spec: MetricSpec) -> None:
        self._specs[spec.name] = spec
        self._states[spec.name] = CUSUMState(
            metric_name=spec.name,
            mu_0=spec.mu_0,
            k=spec.k,
            h=spec.h,
        )

    def update(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Optional[DriftAlarm]:
        """
        Feed one new observation to the named CUSUM detector.

        Returns a DriftAlarm if the update triggers an alarm, else None.
        """
        if metric_name not in self._states:
            logger.warning(
                "Unknown metric — ignoring",
                extra={"metric": metric_name, "node": self.node},
            )
            return None

        state = self._states[metric_name]
        was_alarm = state.is_alarm
        state.update(value, timestamp)

        # Fire alarm on new alarm (not repeated on subsequent updates)
        if state.is_alarm and not was_alarm:
            spec = self._specs[metric_name]
            action = (
                spec.action_up if state.alarm_up else spec.action_down
            )
            s_val = state.s_plus if state.alarm_up else state.s_minus

            alarm = DriftAlarm(
                metric_name=metric_name,
                direction=state.alarm_direction,
                s_value=s_val,
                recommended_action=action,
                timestamp=timestamp or pd.Timestamp.now(tz="UTC"),
                node=self.node,
                obs_count=state.t,
            )
            self._alarms.append(alarm)
            return alarm

        return None

    def update_from_settled_trades(
        self,
        settled_trades: pd.DataFrame,
        window_days: int = 7,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> list[DriftAlarm]:
        """
        Batch-update all metrics from the latest window of settled trades
        for this monitor's node.

        This is called daily after settlement data arrives, typically at
        ~14:00 CPT (after RTM settlement closes for the previous delivery day).

        Parameters
        ----------
        settled_trades : Settled trades DataFrame from Phase 4 settlement.
        window_days : Rolling window for rate metrics (default 7 days).
        timestamp : Observation timestamp for history records.
        """
        ts = timestamp or pd.Timestamp.now(tz="UTC")
        node_trades = settled_trades[settled_trades["node"] == self.node].copy()

        if node_trades.empty:
            return []

        cutoff = node_trades["timestamp"].max() - pd.Timedelta(days=window_days)
        window = node_trades[node_trades["timestamp"] >= cutoff]

        alarms: list[DriftAlarm] = []

        # 1. Implementation shortfall per MWh
        if "implementation_shortfall" in node_trades.columns and "awarded_mw" in node_trades.columns:
            total_is = float(window["implementation_shortfall"].sum())
            total_mw = float(window["awarded_mw"].sum())
            is_per_mwh = total_is / max(total_mw, 0.01)
            alarm = self.update("implementation_shortfall", is_per_mwh, ts)
            if alarm:
                alarms.append(alarm)

        # 2. Win rate (7-day rolling)
        if "realised_pnl" in window.columns and len(window) > 0:
            win_rate = float((window["realised_pnl"] > 0).mean())
            alarm = self.update("win_rate_7d", win_rate, ts)
            if alarm:
                alarms.append(alarm)

        # 3. DART forecast bias (realised spread - forecast mean)
        if "dart_spread_realised" in window.columns and "mu_forecast" not in window.columns:
            # mu_forecast may be missing in settlement DF — skip gracefully
            pass
        elif "dart_spread_realised" in window.columns and "mu_forecast" in window.columns:
            bias = float((window["dart_spread_realised"] - window["mu_forecast"]).mean())
            alarm = self.update("dart_forecast_bias", bias, ts)
            if alarm:
                alarms.append(alarm)

        # 4. Fill rate
        if "fill_rate" in window.columns and len(window) > 0:
            avg_fill = float(window["fill_rate"].mean())
            alarm = self.update("fill_rate", avg_fill, ts)
            if alarm:
                alarms.append(alarm)

        return alarms

    def update_regime_entropy(
        self,
        regime_proba: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Optional[DriftAlarm]:
        """
        Update the regime entropy metric from the HMM posterior probabilities.

        Shannon entropy: H = -Σ_k p_k log(p_k)
        High entropy → HMM is uncertain about which regime we're in.
        Rising entropy → HMM needs retraining.

        regime_proba : shape (n_regimes,) soft probability vector.
        """
        proba = np.clip(regime_proba, 1e-10, 1.0)
        entropy = float(-np.sum(proba * np.log(proba)))
        return self.update("regime_entropy", entropy, timestamp)

    def check_alarms(self) -> list[DriftAlarm]:
        """Return all active (unacknowledged) alarms."""
        return [a for a in self._alarms if not a.acknowledged]

    def acknowledge_alarm(self, metric_name: str) -> None:
        """Mark alarms for a metric as acknowledged and reset the CUSUM state."""
        for alarm in self._alarms:
            if alarm.metric_name == metric_name:
                alarm.acknowledged = True
        if metric_name in self._states:
            self._states[metric_name].reset()

    def history(self, metric_name: str) -> pd.DataFrame:
        """Return the CUSUM history DataFrame for a metric."""
        if metric_name not in self._states:
            return pd.DataFrame()
        return self._states[metric_name].history_dataframe()

    def status_report(self) -> pd.DataFrame:
        """Summary of current CUSUM state for all metrics."""
        rows = []
        for name, state in self._states.items():
            rows.append({
                "node": self.node,
                "metric": name,
                "s_plus": round(state.s_plus, 4),
                "s_minus": round(state.s_minus, 4),
                "h": state.h,
                "alarm": state.is_alarm,
                "alarm_direction": state.alarm_direction,
                "t": state.t,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# System-wide Drift Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    System-wide CUSUM drift detector managing monitors for all active nodes.

    The DriftDetector is the top-level Phase 5 component. It:
      1. Maintains one CUSUMMonitor per active trading node
      2. Collects daily metric updates from settled trades
      3. Fires DriftAlarm events and routes them to the retraining dispatcher
      4. Provides a unified status dashboard across all nodes and metrics

    Usage (nightly, after settlement)
    ----------------------------------
        detector = DriftDetector(nodes=["HB_NORTH", "HB_SOUTH"])
        detector.calibrate(backtest_result)          # set μ_0 from backtest stats
        alarms = detector.update_all(settled_trades)  # feed new data
        if detector.should_retrain():
            retraining_dispatcher.run(alarms)
    """

    def __init__(
        self,
        nodes: Optional[list[str]] = None,
        backtest_stats: Optional[dict] = None,
    ) -> None:
        self._nodes = nodes or []
        self._backtest_stats = backtest_stats
        self._monitors: dict[str, CUSUMMonitor] = {}
        self._alarm_log: list[DriftAlarm] = []
        self._retraining_callbacks: list[Callable[[list[DriftAlarm]], None]] = []

        for node in self._nodes:
            self._init_monitor(node)

    def _init_monitor(self, node: str) -> CUSUMMonitor:
        specs = default_metric_specs(self._backtest_stats)
        monitor = CUSUMMonitor(node, specs)
        self._monitors[node] = monitor
        return monitor

    def calibrate(self, backtest_stats: dict) -> None:
        """
        Re-calibrate all monitor μ_0 values from backtest statistics.

        Call this after the Phase 4 BacktestEngine.run() completes to
        set empirically grounded thresholds rather than theoretical defaults.
        """
        self._backtest_stats = backtest_stats
        for node in self._monitors:
            specs = default_metric_specs(backtest_stats)
            self._monitors[node] = CUSUMMonitor(node, specs)
        logger.info(
            "DriftDetector calibrated from backtest stats",
            extra={"nodes": list(self._monitors.keys())},
        )

    def add_node(self, node: str) -> None:
        if node not in self._monitors:
            self._init_monitor(node)

    def register_retraining_callback(
        self, callback: Callable[[list[DriftAlarm]], None]
    ) -> None:
        """
        Register a function to call when retraining is triggered.
        The callback receives the list of active DriftAlarm objects.

        Example callback: lambda alarms: forecasting_engine.fit(fresh_data)
        """
        self._retraining_callbacks.append(callback)

    def update_all(
        self,
        settled_trades: pd.DataFrame,
        regime_probas: Optional[dict[str, np.ndarray]] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> list[DriftAlarm]:
        """
        Feed the latest settled trades to all node monitors.

        Parameters
        ----------
        settled_trades : Settled trades DataFrame from Phase 4 settlement.
        regime_probas : Optional dict {node: regime_proba_array} from the
                        latest HMM prediction for entropy monitoring.
        timestamp : Observation timestamp (defaults to now).

        Returns
        -------
        All new DriftAlarm objects fired this update cycle.
        """
        ts = timestamp or pd.Timestamp.now(tz="UTC")
        new_alarms: list[DriftAlarm] = []

        for node, monitor in self._monitors.items():
            alarms = monitor.update_from_settled_trades(settled_trades, timestamp=ts)
            new_alarms.extend(alarms)

            if regime_probas and node in regime_probas:
                alarm = monitor.update_regime_entropy(regime_probas[node], ts)
                if alarm:
                    new_alarms.append(alarm)

        self._alarm_log.extend(new_alarms)

        if new_alarms:
            logger.warning(
                "Drift alarms fired",
                extra={
                    "n_alarms": len(new_alarms),
                    "actions": list({a.recommended_action.value for a in new_alarms}),
                    "nodes": list({a.node for a in new_alarms}),
                },
            )
            # Fire retraining callbacks
            if self.should_retrain(new_alarms):
                for cb in self._retraining_callbacks:
                    try:
                        cb(new_alarms)
                    except Exception as e:
                        logger.error(
                            "Retraining callback failed",
                            extra={"error": str(e)},
                        )

        return new_alarms

    @staticmethod
    def should_retrain(alarms: list[DriftAlarm]) -> bool:
        """
        Return True if any alarm warrants a model retraining action.
        ADJUST_PARAMS and NONE do not trigger retraining.
        """
        retraining_actions = {
            RetrainAction.RETRAIN_FULL,
            RetrainAction.RETRAIN_HMM,
            RetrainAction.SUSPEND,
        }
        return any(a.recommended_action in retraining_actions for a in alarms)

    def active_alarms(self) -> list[DriftAlarm]:
        """Return all unacknowledged alarms across all nodes."""
        return [a for a in self._alarm_log if not a.acknowledged]

    def acknowledge_all(self, metric_name: str) -> None:
        """Acknowledge and reset a specific metric across all nodes."""
        for monitor in self._monitors.values():
            monitor.acknowledge_alarm(metric_name)

    def status_report(self) -> pd.DataFrame:
        """Unified status dashboard for all nodes and metrics."""
        frames = [m.status_report() for m in self._monitors.values()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def alarm_history(self) -> pd.DataFrame:
        """Return the full historical alarm log as a DataFrame."""
        if not self._alarm_log:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self._alarm_log])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save alarm history
        self.alarm_history().to_parquet(
            path.with_suffix(".parquet"), index=False
        ) if not self.alarm_history().empty else None
        # Save CUSUM states as JSON
        states = {}
        for node, monitor in self._monitors.items():
            states[node] = {
                name: {
                    "s_plus": s.s_plus,
                    "s_minus": s.s_minus,
                    "t": s.t,
                    "mu_0": s.mu_0,
                    "k": s.k,
                    "h": s.h,
                }
                for name, s in monitor._states.items()
            }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(states, f, indent=2)
        logger.info("DriftDetector saved", extra={"path": str(path)})
