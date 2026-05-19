"""
Markov-Switching GARCH (MS-GARCH) Volatility Model.

Standard GARCH(1,1) assumes a single volatility regime. In ERCOT, volatility
is structurally different across market regimes:
  - Normal:              Low ω, moderate α and β — slow, mean-reverting vol
  - Scarcity:            High ω, high α — vol spikes fast (ORDC adder events)
  - Negative Congestion: Moderate ω, high β — vol is persistent (sustained surplus)

The MS-GARCH model fits independent GARCH(1,1) parameters (ω, α, β) for each
regime and switches between them using the Viterbi-decoded regime sequence.

Mathematical Formulation
------------------------
For regime k at time t, the conditional variance evolves as:

    σ²_t = ω_k + α_k · ε²_{t-1} + β_k · σ²_{t-1}

where ε_t = r_t - μ_k  (residual from regime mean)
and the constraint ω_k > 0, α_k ≥ 0, β_k ≥ 0, α_k + β_k < 1 ensures stationarity.

Parameter estimation uses scipy.optimize.minimize with the negative
log-likelihood of the regime-conditional Gaussian:

    L_k = -½ Σ_t [ log(2π) + log(σ²_t) + ε²_t / σ²_t ]

over the subset of observations assigned to regime k.

The resulting per-observation σ_t array is passed directly into the
PyMC likelihood as a fixed (data-level) scale parameter, properly propagating
regime-driven uncertainty into the posterior distribution.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ercot_dart.models.hmm import REGIME_LABELS, RegimeForecast
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class GARCHParams:
    """
    GARCH(1,1) parameters for a single regime.

    omega: long-run variance floor
    alpha: ARCH coefficient (shock sensitivity)
    beta:  GARCH coefficient (variance persistence)
    mu:    regime mean of the return series
    """
    omega: float
    alpha: float
    beta: float
    mu: float = 0.0

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        denom = 1 - self.persistence
        return self.omega / max(denom, 1e-8)

    @property
    def long_run_vol(self) -> float:
        return np.sqrt(max(self.long_run_variance, 0.0))

    def is_stationary(self) -> bool:
        return (
            self.omega > 0
            and self.alpha >= 0
            and self.beta >= 0
            and self.persistence < 1.0
        )


@dataclass
class MSGARCHResult:
    """
    Output of MSGARCHVolatility.fit() and .predict().

    Attributes
    ----------
    params : dict regime_id → GARCHParams
    conditional_vol : np.ndarray shape (T,)
        Per-observation conditional standard deviation σ_t.
        Passed to the PyMC model as the likelihood scale.
    conditional_var : np.ndarray shape (T,)
        σ²_t per observation.
    regimes : np.ndarray shape (T,)
        Integer regime label per observation (from HMM Viterbi decoding).
    """
    params: dict[int, GARCHParams]
    conditional_vol: np.ndarray
    conditional_var: np.ndarray
    regimes: np.ndarray
    timestamps: pd.Series = field(default_factory=pd.Series)

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "regime": self.regimes,
            "conditional_vol": self.conditional_vol,
            "conditional_var": self.conditional_var,
        })
        df["regime_name"] = df["regime"].map(REGIME_LABELS)
        return df

    def regime_vol_summary(self) -> pd.DataFrame:
        rows = []
        for regime_id, p in self.params.items():
            rows.append({
                "regime": REGIME_LABELS.get(regime_id, str(regime_id)),
                "omega": round(p.omega, 6),
                "alpha": round(p.alpha, 4),
                "beta": round(p.beta, 4),
                "persistence": round(p.persistence, 4),
                "long_run_vol": round(p.long_run_vol, 4),
                "stationary": p.is_stationary(),
            })
        return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# GARCH(1,1) log-likelihood
# ---------------------------------------------------------------------------

def _garch_nll(
    params: np.ndarray,
    returns: np.ndarray,
    sigma2_init: float,
) -> float:
    """
    Negative log-likelihood of a GARCH(1,1) model.

    params = [mu, omega, alpha, beta]
    Penalizes non-stationary solutions via a large additive constant.
    """
    mu, omega, alpha, beta = params

    # Stationarity and positivity constraints via penalty
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10

    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = sigma2_init

    for t in range(1, T):
        eps2 = (returns[t - 1] - mu) ** 2
        sigma2[t] = omega + alpha * eps2 + beta * sigma2[t - 1]
        # Clip to avoid numerical underflow
        sigma2[t] = max(sigma2[t], 1e-8)

    nll = 0.5 * np.sum(np.log(sigma2) + (returns - mu) ** 2 / sigma2)
    return nll


# ---------------------------------------------------------------------------
# MS-GARCH Model
# ---------------------------------------------------------------------------

class MSGARCHVolatility:
    """
    Markov-Switching GARCH(1,1) volatility model.

    Fits independent GARCH parameters for each market regime detected
    by the HMM, then constructs a per-observation conditional volatility
    series by switching regimes according to the Viterbi-decoded state sequence.

    Usage
    -----
        garch = MSGARCHVolatility()
        result = garch.fit(dart_spread_series, regime_forecast)
        sigma_series = result.conditional_vol  # feed into PyMC
    """

    # Starting values for (mu, omega, alpha, beta) per regime
    _INIT_PARAMS: dict[int, tuple[float, float, float, float]] = {
        0: (0.0,  1.0,  0.05, 0.90),   # Normal: persistent, low baseline
        1: (2.0,  5.0,  0.15, 0.80),   # Scarcity: higher shock sensitivity
        2: (-2.0, 2.0,  0.08, 0.85),   # NegCong: moderate persistence
    }

    def __init__(self, max_iter: int = 500, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._params: dict[int, GARCHParams] = {}
        self._is_fitted: bool = False

    def _fit_single_regime(
        self,
        returns: np.ndarray,
        regime_id: int,
    ) -> GARCHParams:
        """
        Estimate GARCH(1,1) parameters for the observations belonging
        to a single regime via numerical MLE.
        """
        if len(returns) < 10:
            logger.warning(
                "Insufficient observations for GARCH fit — using defaults",
                extra={"regime": REGIME_LABELS.get(regime_id), "n_obs": len(returns)},
            )
            mu0, omega0, alpha0, beta0 = self._INIT_PARAMS.get(regime_id, (0, 1, 0.05, 0.9))
            return GARCHParams(omega=omega0, alpha=alpha0, beta=beta0, mu=mu0)

        sigma2_init = float(np.var(returns))
        x0 = np.array(self._INIT_PARAMS.get(regime_id, (0.0, 1.0, 0.05, 0.90)))

        result = minimize(
            _garch_nll,
            x0=x0,
            args=(returns, sigma2_init),
            method="Nelder-Mead",
            options={"maxiter": self.max_iter, "xatol": self.tol, "fatol": self.tol},
        )

        if not result.success:
            logger.warning(
                "GARCH optimization did not converge — using best found",
                extra={"regime": REGIME_LABELS.get(regime_id), "msg": result.message},
            )

        mu, omega, alpha, beta = result.x
        # Project to feasible region
        omega = max(omega, 1e-6)
        alpha = max(alpha, 0.0)
        beta = max(beta, 0.0)
        # Enforce stationarity
        if alpha + beta >= 1.0:
            scale = 0.98 / (alpha + beta)
            alpha *= scale
            beta *= scale

        params = GARCHParams(omega=omega, alpha=alpha, beta=beta, mu=mu)
        logger.info(
            "GARCH fit complete",
            extra={
                "regime": REGIME_LABELS.get(regime_id),
                "omega": round(omega, 6),
                "alpha": round(alpha, 4),
                "beta": round(beta, 4),
                "persistence": round(params.persistence, 4),
                "long_run_vol": round(params.long_run_vol, 4),
            },
        )
        return params

    def fit(
        self,
        returns: pd.Series,
        regime_forecast: RegimeForecast,
    ) -> MSGARCHResult:
        """
        Fit regime-conditional GARCH parameters and compute the full
        per-observation conditional volatility series.

        Parameters
        ----------
        returns : pd.Series
            The DART spread time series aligned to regime_forecast.timestamps.
        regime_forecast : RegimeForecast
            Output of RegimeDetector.predict() — provides the Viterbi regime sequence.

        Returns
        -------
        MSGARCHResult with per-observation conditional_vol for PyMC input.
        """
        regimes = regime_forecast.regime
        timestamps = regime_forecast.timestamps
        ret_arr = returns.values.astype(np.float64)

        if len(ret_arr) != len(regimes):
            raise ValueError(
                f"Length mismatch: returns ({len(ret_arr)}) != regimes ({len(regimes)})"
            )

        # Fit per-regime GARCH parameters
        for regime_id in range(regime_forecast.transition_matrix.shape[0]):
            mask = regimes == regime_id
            regime_returns = ret_arr[mask]
            self._params[regime_id] = self._fit_single_regime(regime_returns, regime_id)

        # Compute full conditional variance series using regime-switching
        # σ²_t switches GARCH parameter set based on S_t
        T = len(ret_arr)
        sigma2 = np.empty(T)
        sigma2[0] = float(np.var(ret_arr))

        for t in range(1, T):
            p = self._params[regimes[t]]
            eps2 = (ret_arr[t - 1] - p.mu) ** 2
            sigma2[t] = p.omega + p.alpha * eps2 + p.beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-8)

        conditional_vol = np.sqrt(sigma2)
        self._is_fitted = True

        return MSGARCHResult(
            params=self._params,
            conditional_vol=conditional_vol,
            conditional_var=sigma2,
            regimes=regimes,
            timestamps=timestamps,
        )

    def predict_vol(
        self,
        last_return: float,
        last_sigma2: float,
        regime_id: int,
        n_ahead: int = 1,
    ) -> np.ndarray:
        """
        Multi-step ahead conditional variance forecast for a given regime.

        For h > 1, the GARCH(1,1) h-step forecast reverts toward the
        regime's long-run variance:
            σ²_{t+h} = σ²_lr + (α+β)^{h-1} · (σ²_{t+1} - σ²_lr)

        Used in Phase 3 to size positions over the delivery horizon.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_vol().")

        p = self._params[regime_id]
        sigma2_lr = p.long_run_variance
        forecasts = np.empty(n_ahead)

        # 1-step ahead
        eps2 = (last_return - p.mu) ** 2
        sigma2_next = p.omega + p.alpha * eps2 + p.beta * last_sigma2
        forecasts[0] = sigma2_next

        # h-step ahead (analytic formula)
        for h in range(1, n_ahead):
            persistence_h = p.persistence ** h
            forecasts[h] = sigma2_lr + persistence_h * (sigma2_next - sigma2_lr)

        return np.sqrt(np.maximum(forecasts, 1e-8))

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({"params": self._params, "fitted": self._is_fitted}, f)
        logger.info("MSGARCHVolatility saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> "MSGARCHVolatility":
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._params = state["params"]
        obj._is_fitted = state["fitted"]
        logger.info("MSGARCHVolatility loaded", extra={"path": str(path)})
        return obj
