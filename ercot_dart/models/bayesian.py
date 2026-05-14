"""
Bayesian MCMC Regression Model for DART Spread Forecasting.

This module implements a hierarchical Bayesian regression using PyMC (v5+).
The model outputs a full posterior probability distribution over the DART
spread for each (timestamp, node), not a point estimate. This posterior is
the direct input to the Kelly sizer in Phase 3.

Model Specification
-------------------
The DART spread y_t at time t is modeled as:

    y_t ~ Normal(μ_t, σ_t)

    μ_t = α                          (global intercept)
          + X_t · β                  (fundamental regressors)
          + δ_{S_t}                  (regime-specific intercept shift)

    σ_t = σ_{GARCH,t}               (MS-GARCH conditional volatility)

Priors
------
    α           ~ Normal(0, 5)                        (weakly informative intercept)
    β_j         ~ Normal(0, σ_β)                      (shrinkage prior on features)
    σ_β         ~ HalfNormal(1)                       (hierarchical hyperprior)
    δ_k         ~ Normal(0, 2)   k ∈ {0,1,2}          (regime intercept shifts)

The σ_GARCH is treated as a fixed known quantity (passed as data), derived
from the MS-GARCH conditional volatility. This correctly propagates regime-
driven uncertainty into the posterior without making σ a latent variable
(which would make NUTS sampling intractable at ERCOT scale).

Posterior Predictive
--------------------
After sampling, we draw from the posterior predictive:
    ỹ ~ Normal(μ̃, σ̃)

where μ̃ and σ̃ are themselves samples from the posterior. This yields a
distribution of DART spread predictions that fully accounts for:
  1. Fundamental uncertainty (β posterior width)
  2. Regime uncertainty (δ posterior width)
  3. Conditional volatility uncertainty (σ_GARCH)

The resulting posterior samples are used in Phase 3 to compute:
  - E[DART | X] = posterior mean μ
  - Var[DART | X] = posterior variance σ²
  - P(DART > threshold) — probability that the trade is profitable
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature selection for Bayesian regression
# ---------------------------------------------------------------------------

# These features are the regressors X in the Bayesian model.
# They are a superset of the HMM features and capture fundamental drivers.
BAYESIAN_FEATURES: list[str] = [
    # Fourier seasonality
    "daily_sin_1", "daily_cos_1", "daily_sin_2", "daily_cos_2",
    "weekly_sin_1", "weekly_cos_1", "weekly_sin_2", "weekly_cos_2",
    "annual_sin_1", "annual_cos_1",
    # Calendar
    "is_weekend", "is_peak_hour",
    # Temperature hinges (zonal)
    "hinge_north_65f", "hinge_north_85f", "hinge_north_95f",
    "hinge_houston_65f", "hinge_houston_85f", "hinge_houston_95f",
    # Load forecast error
    "nlfe_system_lag1d", "nlfe_rolling_24h",
    # Shift factor
    "shift_factor_proxy", "sfp_rolling_168h",
    # Supply stack
    "stack_slope_high", "inframarginal_mw", "peaker_mw",
    # Implied heat rate
    "implied_heat_rate", "ihr_deviation",
    # Lagged DART
    "dart_lag_24h", "dart_lag_168h",
    "dart_rolling_168h_mean", "dart_rolling_168h_std",
    "dart_z_score_168h",
]


# ---------------------------------------------------------------------------
# Posterior forecast output
# ---------------------------------------------------------------------------

@dataclass
class PosteriorForecast:
    """
    Full posterior distribution over the DART spread for one or more
    (timestamp, node) observations.

    Attributes
    ----------
    mu_posterior : np.ndarray shape (n_draws, T)
        Posterior samples of the expected DART spread.
    sigma_posterior : np.ndarray shape (n_draws, T)
        Posterior samples of the predictive standard deviation.
    predictive_samples : np.ndarray shape (n_draws, T)
        Draws from the posterior predictive distribution.
    mu_mean : np.ndarray shape (T,)
        Posterior mean of μ — point estimate used for Kelly sizing.
    sigma_mean : np.ndarray shape (T,)
        Posterior mean of σ — uncertainty estimate used for Kelly sizing.
    hdi_lower : np.ndarray shape (T,)
        Lower bound of the 95% Highest Density Interval (HDI).
    hdi_upper : np.ndarray shape (T,)
        Upper bound of the 95% HDI.
    prob_positive : np.ndarray shape (T,)
        P(DART spread > 0) — probability of a profitable Virtual Supply trade.
    timestamps : pd.Series
    nodes : pd.Series
    """

    mu_posterior: np.ndarray
    sigma_posterior: np.ndarray
    predictive_samples: np.ndarray
    mu_mean: np.ndarray
    sigma_mean: np.ndarray
    hdi_lower: np.ndarray
    hdi_upper: np.ndarray
    prob_positive: np.ndarray
    timestamps: pd.Series = field(default_factory=pd.Series)
    nodes: pd.Series = field(default_factory=pd.Series)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "node": self.nodes,
            "mu_mean": self.mu_mean,
            "sigma_mean": self.sigma_mean,
            "hdi_lower_95": self.hdi_lower,
            "hdi_upper_95": self.hdi_upper,
            "prob_positive": self.prob_positive,
        })

    def kelly_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mu, sigma) arrays — the two sufficient statistics
        for the continuous Kelly formula in Phase 3.

        mu    = E[DART spread]
        sigma = std(DART spread) from the posterior predictive
        """
        return self.mu_mean, self.sigma_mean


# ---------------------------------------------------------------------------
# Bayesian DART Model
# ---------------------------------------------------------------------------

class BayesianDARTModel:
    """
    PyMC hierarchical Bayesian regression for DART spread forecasting.

    The model uses NUTS (No-U-Turn Sampler) for efficient sampling of
    the posterior. For production inference where wall-clock time is
    constrained by the 10:00 AM gate closure, we expose:
      - draws / chains / target_accept  for accuracy vs. speed tradeoff
      - use_map_init to warm-start NUTS from the MAP estimate

    Usage
    -----
        model = BayesianDARTModel()
        idata = model.fit(X_train, y_train, sigma_garch_train, regimes_train)
        forecast = model.predict(X_new, sigma_garch_new, regimes_new)
    """

    def __init__(
        self,
        features: Optional[list[str]] = None,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.90,
        random_seed: int = 42,
        use_map_init: bool = True,
    ) -> None:
        self.features = features or BAYESIAN_FEATURES
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.use_map_init = use_map_init

        self._idata: Optional[az.InferenceData] = None
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    # -----------------------------------------------------------------------
    # Feature preparation
    # -----------------------------------------------------------------------

    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Select available features, drop NaN rows, and standardize.

        Standardization is fit on training data only (fit=True) and
        applied to new data using stored mean/std (fit=False).
        """
        available = [f for f in self.features if f in df.columns]
        missing = set(self.features) - set(available)
        if missing:
            logger.warning(
                "Bayesian model features missing — using available subset",
                extra={"missing_count": len(missing)},
            )

        X_raw = df[available].values.astype(np.float64)

        if fit:
            self._feature_means = np.nanmean(X_raw, axis=0)
            self._feature_stds = np.nanstd(X_raw, axis=0)
            self._feature_stds = np.where(
                self._feature_stds < 1e-8, 1.0, self._feature_stds
            )
            self._fitted_features = available

        means = self._feature_means
        stds = self._feature_stds

        # Replace NaN with column mean before standardizing
        nan_mask = np.isnan(X_raw)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = means[j]

        X_scaled = (X_raw - means) / stds
        return X_scaled, available

    # -----------------------------------------------------------------------
    # Model construction
    # -----------------------------------------------------------------------

    def _build_pymc_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sigma_garch: np.ndarray,
        regimes: np.ndarray,
        n_regimes: int = 3,
    ) -> pm.Model:
        """
        Construct the PyMC model graph.

        The PyMC model is a context manager that stores the graph in-memory.
        We return it so that fit() can call pm.sample() within the context.
        """
        n_obs, n_features = X.shape

        with pm.Model() as model:
            # ----------------------------------------------------------------
            # Data containers (allow out-of-sample prediction via set_data)
            # ----------------------------------------------------------------
            X_data = pm.Data("X", X, mutable=True)
            sigma_data = pm.Data("sigma_garch", sigma_garch, mutable=True)
            regime_data = pm.Data("regimes", regimes.astype(int), mutable=True)

            # ----------------------------------------------------------------
            # Priors
            # ----------------------------------------------------------------

            # Global intercept — weakly informative
            alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)

            # Hierarchical shrinkage prior on regression coefficients
            # σ_β ~ HalfNormal(1) acts as a regularization hyperparameter
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)
            beta = pm.Normal("beta", mu=0.0, sigma=sigma_beta, shape=n_features)

            # Regime-specific intercept shifts
            # δ_k captures the regime's systematic DART bias above the fundamental
            delta = pm.Normal("delta", mu=0.0, sigma=2.0, shape=n_regimes)

            # ----------------------------------------------------------------
            # Expected DART spread
            # ----------------------------------------------------------------
            mu = alpha + pm.math.dot(X_data, beta) + delta[regime_data]

            # ----------------------------------------------------------------
            # Likelihood
            # ----------------------------------------------------------------
            # σ_GARCH is treated as fixed data (not a latent variable) to
            # maintain sampling efficiency under the gate-closure time constraint.
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_data, observed=y)  # noqa: F841

        return model

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(
        self,
        feature_matrix: pd.DataFrame,
        sigma_garch: np.ndarray,
        regimes: np.ndarray,
        target_col: str = "dart_spread",
    ) -> az.InferenceData:
        """
        Sample from the posterior distribution of model parameters.

        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Training feature matrix from Phase 1 FeatureEngineer.
        sigma_garch : np.ndarray shape (T,)
            Per-observation conditional volatility from MS-GARCH.
        regimes : np.ndarray shape (T,)
            Viterbi-decoded regime labels from HMM.
        target_col : str
            Column name for the DART spread target variable.

        Returns
        -------
        idata : az.InferenceData
            ArviZ InferenceData object containing posterior samples,
            posterior predictive samples, and sampling diagnostics.
        """
        feature_matrix = feature_matrix.dropna(subset=[target_col]).copy()
        y = feature_matrix[target_col].values.astype(np.float64)

        X, available_features = self._prepare_features(feature_matrix, fit=True)
        n_regimes = len(np.unique(regimes))

        logger.info(
            "Starting MCMC sampling",
            extra={
                "n_obs": len(y),
                "n_features": X.shape[1],
                "draws": self.draws,
                "chains": self.chains,
                "target_accept": self.target_accept,
            },
        )

        pymc_model = self._build_pymc_model(X, y, sigma_garch, regimes, n_regimes)

        with pymc_model:
            # Optional MAP initialization — warm-starts NUTS, reduces divergences
            initvals = None
            if self.use_map_init:
                try:
                    map_est = pm.find_MAP(progressbar=False)
                    initvals = map_est
                    logger.info("MAP initialization found")
                except Exception as e:
                    logger.warning("MAP init failed — using default init", extra={"error": str(e)})

            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                initvals=initvals,
                progressbar=False,
                return_inferencedata=True,
            )

            # Sample posterior predictive for in-sample diagnostics
            pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)

        self._idata = idata
        self._is_fitted = True
        self._pymc_model = pymc_model

        # Log sampling diagnostics
        divergences = int(idata.sample_stats.diverging.sum().item())
        r_hat_max = float(az.rhat(idata).max().to_array().max().item())
        logger.info(
            "MCMC sampling complete",
            extra={
                "divergences": divergences,
                "r_hat_max": round(r_hat_max, 4),
                "converged": r_hat_max < 1.05 and divergences == 0,
            },
        )

        return idata

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        feature_matrix: pd.DataFrame,
        sigma_garch: np.ndarray,
        regimes: np.ndarray,
        hdi_prob: float = 0.95,
        n_predictive_draws: int = 2000,
    ) -> PosteriorForecast:
        """
        Generate the posterior predictive distribution for new observations.

        Uses the fitted posterior (idata) to draw samples of μ and σ,
        then constructs the full predictive distribution for the DART spread.

        Parameters
        ----------
        feature_matrix : DataFrame of new observations (pre-gate features).
        sigma_garch : MS-GARCH conditional vol for the new observations.
        regimes : HMM Viterbi regime labels for the new observations.
        hdi_prob : Highest Density Interval probability (default 95%).
        n_predictive_draws : Number of posterior predictive samples to draw.

        Returns
        -------
        PosteriorForecast with full distribution statistics.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X_new, _ = self._prepare_features(feature_matrix, fit=False)
        T = len(X_new)

        # Extract posterior samples
        post = self._idata.posterior
        alpha_samples = post["alpha"].values.reshape(-1)            # (S,)
        beta_samples = post["beta"].values.reshape(-1, X_new.shape[1])  # (S, d)
        delta_samples = post["delta"].values.reshape(-1, post["delta"].shape[-1])  # (S, K)

        # Subsample to n_predictive_draws
        S_total = len(alpha_samples)
        idx = np.random.default_rng(self.random_seed).choice(
            S_total, size=min(n_predictive_draws, S_total), replace=False
        )
        alpha_s = alpha_samples[idx]         # (S,)
        beta_s = beta_samples[idx]           # (S, d)
        delta_s = delta_samples[idx]         # (S, K)

        # Compute posterior μ for each observation and sample
        # mu_matrix shape: (S, T)
        mu_matrix = (
            alpha_s[:, None]
            + beta_s @ X_new.T
            + delta_s[:, regimes.astype(int)]
        )

        # Posterior predictive: ỹ ~ Normal(μ_s, σ_GARCH)
        rng = np.random.default_rng(self.random_seed + 1)
        noise = rng.standard_normal(size=(len(idx), T))
        predictive_samples = mu_matrix + noise * sigma_garch[None, :]

        # Summary statistics
        mu_mean = mu_matrix.mean(axis=0)                            # (T,)
        sigma_mean = predictive_samples.std(axis=0)                 # (T,)

        # HDI via ArviZ (on predictive samples transposed to (T, S) then back)
        hdi_bounds = az.hdi(predictive_samples.T[np.newaxis], hdi_prob=hdi_prob)
        # hdi_bounds shape depends on arviz version — handle both
        if hasattr(hdi_bounds, "values"):
            hdi_arr = hdi_bounds.values  # xarray DataArray
        else:
            hdi_arr = np.array(hdi_bounds)

        if hdi_arr.ndim == 3:
            hdi_lower = hdi_arr[0, :, 0]
            hdi_upper = hdi_arr[0, :, 1]
        else:
            hdi_lower = hdi_arr[:, 0]
            hdi_upper = hdi_arr[:, 1]

        # P(DART > 0)
        prob_positive = (predictive_samples > 0).mean(axis=0)      # (T,)

        timestamps = feature_matrix["timestamp"] if "timestamp" in feature_matrix.columns \
            else pd.Series(np.arange(T))
        nodes = feature_matrix["node"] if "node" in feature_matrix.columns \
            else pd.Series(["unknown"] * T)

        logger.info(
            "Posterior predictive complete",
            extra={
                "n_obs": T,
                "mean_mu": round(float(mu_mean.mean()), 4),
                "mean_sigma": round(float(sigma_mean.mean()), 4),
                "mean_prob_positive": round(float(prob_positive.mean()), 4),
            },
        )

        return PosteriorForecast(
            mu_posterior=mu_matrix,
            sigma_posterior=np.tile(sigma_garch, (len(idx), 1)),
            predictive_samples=predictive_samples,
            mu_mean=mu_mean,
            sigma_mean=sigma_mean,
            hdi_lower=hdi_lower,
            hdi_upper=hdi_upper,
            prob_positive=prob_positive,
            timestamps=timestamps.reset_index(drop=True),
            nodes=nodes.reset_index(drop=True),
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def convergence_summary(self) -> pd.DataFrame:
        """
        Return R-hat and ESS diagnostics for all sampled parameters.

        R-hat < 1.05 and ESS > 400 indicate adequate convergence.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        rhat = az.rhat(self._idata)
        ess = az.ess(self._idata)

        rows = []
        for var in rhat.data_vars:
            rhat_vals = rhat[var].values.flatten()
            ess_vals = ess[var].values.flatten()
            for i, (r, e) in enumerate(zip(rhat_vals, ess_vals)):
                rows.append({
                    "parameter": f"{var}[{i}]" if len(rhat_vals) > 1 else var,
                    "r_hat": round(float(r), 4),
                    "ess": round(float(e), 1),
                    "converged": float(r) < 1.05 and float(e) > 400,
                })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._idata is not None:
            self._idata.to_netcdf(str(path / "idata.nc"))
        meta = {
            "features": self.features,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "fitted_features": getattr(self, "_fitted_features", self.features),
            "draws": self.draws,
            "chains": self.chains,
        }
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("BayesianDARTModel saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> "BayesianDARTModel":
        path = Path(path)
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(features=meta["features"], draws=meta["draws"], chains=meta["chains"])
        obj._feature_means = meta["feature_means"]
        obj._feature_stds = meta["feature_stds"]
        obj._fitted_features = meta.get("fitted_features", meta["features"])
        idata_path = path / "idata.nc"
        if idata_path.exists():
            obj._idata = az.from_netcdf(str(idata_path))
            obj._is_fitted = True
        logger.info("BayesianDARTModel loaded", extra={"path": str(path)})
        return obj
