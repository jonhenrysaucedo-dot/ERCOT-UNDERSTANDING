"""
Phase 2 Forecasting Engine — Orchestrates HMM + MS-GARCH + Bayesian MCMC.

The ForecastingEngine chains the three model components into a single
walk-forward-safe interface:

    1. RegimeDetector (HMM)
         Input:  feature_matrix
         Output: RegimeForecast (regime labels, soft probabilities, transition matrix)

    2. MSGARCHVolatility
         Input:  dart_spread series + RegimeForecast
         Output: MSGARCHResult (per-observation conditional vol σ_t)

    3. BayesianDARTModel (PyMC MCMC)
         Input:  feature_matrix + σ_t + regime labels
         Output: PosteriorForecast (μ, σ, HDI, P(profit))

The engine exposes fit() and predict() so Phase 3 (Kelly sizer) and
Phase 4 (walk-forward backtester) can call it uniformly without knowing
which sub-model is active.

Walk-Forward Protocol
---------------------
In walk-forward validation (Phase 4), this engine is called as:

    engine = ForecastingEngine(config)
    engine.fit(train_window)                    # uses only past data
    forecast = engine.predict(test_row)         # strictly out-of-sample

The train/test split is managed externally by the backtester; this
engine is stateless between calls except for its stored sub-models.
"""

from __future__ import annotations

import pickle
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ercot_dart.models.bayesian import BayesianDARTModel, PosteriorForecast
from ercot_dart.models.garch import MSGARCHResult, MSGARCHVolatility
from ercot_dart.models.hmm import RegimeDetector, RegimeForecast
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ForecastingConfig:
    """
    Configuration for the Phase 2 Forecasting Engine.

    hmm_node : Node used to fit the HMM. Regime detection is system-wide,
               so a liquid hub node (HB_NORTH or HB_BUSAVG) is appropriate.
    target_nodes : Nodes for which we generate DART forecasts.
    mcmc_draws / mcmc_chains : NUTS sampler settings. Reduce for speed
                               during walk-forward validation.
    fractional_kelly : Phase 3 Kelly fraction multiplier (passed through
                       to the Kelly sizer, stored here for convenience).
    """
    hmm_node: str = "HB_NORTH"
    target_nodes: list[str] = field(default_factory=lambda: [
        "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON",
    ])
    n_regimes: int = 3
    hmm_n_iter: int = 200
    mcmc_draws: int = 1000
    mcmc_tune: int = 1000
    mcmc_chains: int = 4
    mcmc_target_accept: float = 0.90
    random_seed: int = 42
    fractional_kelly: float = 0.25         # Phase 3 — quarter-Kelly


# ---------------------------------------------------------------------------
# Combined forecast output
# ---------------------------------------------------------------------------

@dataclass
class CompleteForecast:
    """
    Aggregated output from all three Phase 2 model components.

    This is the object passed to the Kelly sizer (Phase 3).

    Attributes
    ----------
    posterior : PosteriorForecast
        Full posterior distribution over DART spreads.
    regime : RegimeForecast
        HMM regime labels and soft probabilities.
    garch : MSGARCHResult
        Conditional volatility series from MS-GARCH.
    node : str
        Target settlement point node.
    delivery_timestamp : pd.Timestamp
        The delivery hour being forecast (must be after gate closure).
    """
    posterior: PosteriorForecast
    regime: RegimeForecast
    garch: MSGARCHResult
    node: str
    delivery_timestamp: pd.Timestamp

    @property
    def mu(self) -> float:
        """Posterior mean DART spread — primary Kelly input."""
        return float(self.posterior.mu_mean[-1])

    @property
    def sigma(self) -> float:
        """Posterior standard deviation — Kelly denominator."""
        return float(self.posterior.sigma_mean[-1])

    @property
    def prob_profit(self) -> float:
        """P(DART > 0) — trade viability filter."""
        return float(self.posterior.prob_positive[-1])

    @property
    def current_regime(self) -> int:
        """Hard regime label for the delivery hour."""
        return int(self.regime.regime[-1])

    @property
    def hdi_95(self) -> tuple[float, float]:
        """95% Highest Density Interval on the DART spread."""
        return (
            float(self.posterior.hdi_lower[-1]),
            float(self.posterior.hdi_upper[-1]),
        )

    def to_series(self) -> pd.Series:
        lower, upper = self.hdi_95
        return pd.Series({
            "timestamp": self.delivery_timestamp,
            "node": self.node,
            "mu": self.mu,
            "sigma": self.sigma,
            "prob_profit": self.prob_profit,
            "regime": self.current_regime,
            "hdi_lower_95": lower,
            "hdi_upper_95": upper,
        })


# ---------------------------------------------------------------------------
# Forecasting Engine
# ---------------------------------------------------------------------------

class ForecastingEngine:
    """
    Phase 2 model pipeline: HMM → MS-GARCH → Bayesian MCMC.

    The engine is designed to be serialised and reloaded between trading
    days. On each morning (after 60-day files drop, before gate closure):
      1. engine.fit(training_window)  — refit all three models
      2. engine.predict(today_features) — generate forecasts for each node

    The fit/predict split ensures strict look-ahead-bias isolation.
    """

    def __init__(self, config: Optional[ForecastingConfig] = None) -> None:
        self.config = config or ForecastingConfig()

        self._regime_detector = RegimeDetector(
            n_regimes=self.config.n_regimes,
            n_iter=self.config.hmm_n_iter,
            random_state=self.config.random_seed,
        )
        self._garch = MSGARCHVolatility()
        self._bayesian: dict[str, BayesianDARTModel] = {}  # one model per target node

        self._is_fitted: bool = False
        self._last_regime_forecast: Optional[RegimeForecast] = None
        self._last_garch_result: Optional[MSGARCHResult] = None

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(
        self,
        feature_matrix: pd.DataFrame,
        dart_col: str = "dart_spread",
    ) -> "ForecastingEngine":
        """
        Fit all three sub-models on the training feature matrix.

        Call sequence:
          HMM.fit() → MS-GARCH.fit() → BayesianDARTModel.fit() per node

        Parameters
        ----------
        feature_matrix : Output of Phase 1 ETLPipeline.run(), containing
                         features and the dart_spread target column.
        dart_col : Name of the DART spread target column.
        """
        t0 = _time.monotonic()
        logger.info("ForecastingEngine fitting started")

        # ---- Step 1: HMM regime detection ----------------------------------
        logger.info("Step 1: Fitting HMM regime detector")
        self._regime_detector.fit(feature_matrix, node=self.config.hmm_node)

        # Predict regimes for the full training set
        regime_forecast = self._regime_detector.predict(feature_matrix)
        self._last_regime_forecast = regime_forecast

        # Attach regime labels to the feature matrix (aligned by timestamp)
        regime_df = regime_forecast.as_dataframe()[["timestamp", "regime"]]
        feature_matrix = feature_matrix.merge(regime_df, on="timestamp", how="left")

        # ---- Step 2: MS-GARCH volatility -----------------------------------
        logger.info("Step 2: Fitting MS-GARCH volatility model")

        # Use the primary hub node for GARCH fitting
        hub_data = feature_matrix[
            feature_matrix["node"] == self.config.hmm_node
        ].sort_values("timestamp").dropna(subset=[dart_col])

        hub_returns = hub_data[dart_col]
        hub_regimes_aligned = self._align_regimes(regime_forecast, hub_data)

        # Build a hub-specific RegimeForecast aligned to hub_data timestamps
        from ercot_dart.models.hmm import RegimeForecast, REGIME_LABELS as _RL
        hub_regime_forecast = RegimeForecast(
            regime=hub_regimes_aligned,
            regime_name=[_RL.get(r, "Normal") for r in hub_regimes_aligned],
            regime_proba=np.zeros((len(hub_regimes_aligned), self.config.n_regimes)),
            transition_matrix=regime_forecast.transition_matrix,
            means=regime_forecast.means,
            covars=regime_forecast.covars,
            timestamps=hub_data["timestamp"].reset_index(drop=True),
        )
        garch_result = self._garch.fit(hub_returns, hub_regime_forecast)
        self._last_garch_result = garch_result

        # ---- Step 3: Bayesian MCMC per target node -------------------------
        logger.info("Step 3: Fitting Bayesian DART models per node")
        for node in self.config.target_nodes:
            node_data = feature_matrix[
                feature_matrix["node"] == node
            ].sort_values("timestamp").dropna(subset=[dart_col])

            if len(node_data) < 100:
                logger.warning(
                    "Insufficient training data for node — skipping",
                    extra={"node": node, "n_obs": len(node_data)},
                )
                continue

            # Align GARCH sigma to node observations
            node_regimes = self._align_regimes(regime_forecast, node_data)
            node_sigma = self._interpolate_garch_sigma(garch_result, node_data)

            bayes_model = BayesianDARTModel(
                draws=self.config.mcmc_draws,
                tune=self.config.mcmc_tune,
                chains=self.config.mcmc_chains,
                target_accept=self.config.mcmc_target_accept,
                random_seed=self.config.random_seed,
            )
            bayes_model.fit(
                feature_matrix=node_data,
                sigma_garch=node_sigma,
                regimes=node_regimes,
                target_col=dart_col,
            )
            self._bayesian[node] = bayes_model
            logger.info("Bayesian model fitted", extra={"node": node})

        elapsed = _time.monotonic() - t0
        self._is_fitted = True
        logger.info("ForecastingEngine fitting complete", extra={"elapsed_s": round(elapsed, 1)})
        return self

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        feature_matrix: pd.DataFrame,
        node: str,
        delivery_timestamp: Optional[pd.Timestamp] = None,
    ) -> CompleteForecast:
        """
        Generate a CompleteForecast for a single node and delivery period.

        Parameters
        ----------
        feature_matrix : Feature rows for the delivery period (pre-gate).
        node : Target settlement point to forecast.
        delivery_timestamp : Target delivery timestamp (defaults to last row).

        Returns
        -------
        CompleteForecast with posterior, regime, and GARCH sub-forecasts.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if node not in self._bayesian:
            raise KeyError(f"No Bayesian model for node '{node}'. Available: {list(self._bayesian)}")

        # Regime prediction
        regime_forecast = self._regime_detector.predict(feature_matrix, node=node)

        # GARCH 1-step ahead from last known state
        node_data = feature_matrix[feature_matrix["node"] == node].sort_values("timestamp")
        node_sigma = self._interpolate_garch_sigma(self._last_garch_result, node_data)
        node_regimes = regime_forecast.regime

        # Posterior predictive
        posterior = self._bayesian[node].predict(
            feature_matrix=node_data,
            sigma_garch=node_sigma,
            regimes=node_regimes,
        )

        ts = delivery_timestamp or node_data["timestamp"].iloc[-1]

        garch_stub = MSGARCHResult(
            params=self._last_garch_result.params,
            conditional_vol=node_sigma,
            conditional_var=node_sigma ** 2,
            regimes=node_regimes,
            timestamps=node_data["timestamp"].reset_index(drop=True),
        )

        return CompleteForecast(
            posterior=posterior,
            regime=regime_forecast,
            garch=garch_stub,
            node=node,
            delivery_timestamp=ts,
        )

    def predict_all_nodes(
        self,
        feature_matrix: pd.DataFrame,
        delivery_timestamp: Optional[pd.Timestamp] = None,
    ) -> list[CompleteForecast]:
        """Predict for all configured target nodes. Returns list of CompleteForecast."""
        forecasts = []
        for node in self.config.target_nodes:
            if node not in self._bayesian:
                continue
            try:
                forecasts.append(
                    self.predict(feature_matrix, node=node, delivery_timestamp=delivery_timestamp)
                )
            except Exception as e:
                logger.error("Prediction failed for node", extra={"node": node, "error": str(e)})
        return forecasts

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _align_regimes(
        regime_forecast: RegimeForecast,
        node_data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Align HMM regime labels to a node-specific subset of the feature matrix
        by matching on timestamps. Returns regime array in node_data order.
        """
        regime_df = regime_forecast.as_dataframe()[["timestamp", "regime"]]
        merged = node_data[["timestamp"]].merge(regime_df, on="timestamp", how="left")
        regimes = merged["regime"].fillna(0).values.astype(int)
        return regimes

    @staticmethod
    def _interpolate_garch_sigma(
        garch_result: MSGARCHResult,
        node_data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Map GARCH conditional vol (fitted on hub node) to a target node's
        timestamps via forward-fill. If timestamp coverage is partial,
        fall back to the long-run vol for the dominant regime.
        """
        garch_df = garch_result.as_dataframe()[["timestamp", "conditional_vol"]]
        merged = node_data[["timestamp"]].merge(garch_df, on="timestamp", how="left")
        sigma = merged["conditional_vol"].ffill().bfill()

        # Fallback: use long-run vol if still NaN
        if sigma.isna().any():
            dominant_regime = int(np.bincount(garch_result.regimes).argmax())
            lr_vol = garch_result.params[dominant_regime].long_run_vol
            sigma = sigma.fillna(lr_vol)

        return sigma.values.astype(np.float64)

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def regime_statistics(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Delegate to RegimeDetector.regime_statistics()."""
        return self._regime_detector.regime_statistics(feature_matrix)

    def garch_summary(self) -> pd.DataFrame:
        """Return MS-GARCH parameter summary if fitted."""
        if self._last_garch_result is None:
            raise RuntimeError("No GARCH result available. Call fit() first.")
        return self._last_garch_result.regime_vol_summary()

    def convergence_summary(self, node: str) -> pd.DataFrame:
        """Return MCMC convergence diagnostics for a node."""
        if node not in self._bayesian:
            raise KeyError(f"No model for node '{node}'.")
        return self._bayesian[node].convergence_summary()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._regime_detector.save(directory / "regime_detector.pkl")
        self._garch.save(directory / "garch.pkl")

        for node, model in self._bayesian.items():
            safe_name = node.replace("/", "_")
            model.save(directory / f"bayesian_{safe_name}")

        meta = {
            "config": self.config,
            "is_fitted": self._is_fitted,
            "fitted_nodes": list(self._bayesian.keys()),
        }
        with open(directory / "engine_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("ForecastingEngine saved", extra={"directory": str(directory)})

    @classmethod
    def load(cls, directory: Path) -> "ForecastingEngine":
        directory = Path(directory)
        with open(directory / "engine_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        engine = cls(config=meta["config"])
        engine._regime_detector = RegimeDetector.load(directory / "regime_detector.pkl")
        engine._garch = MSGARCHVolatility.load(directory / "garch.pkl")

        for node in meta["fitted_nodes"]:
            safe_name = node.replace("/", "_")
            model_path = directory / f"bayesian_{safe_name}"
            if model_path.exists():
                engine._bayesian[node] = BayesianDARTModel.load(model_path)

        engine._is_fitted = meta["is_fitted"]
        logger.info("ForecastingEngine loaded", extra={"directory": str(directory)})
        return engine
