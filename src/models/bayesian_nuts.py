"""Bayesian DART spread forecaster — M5.

Linear regression with weakly-informative priors, sampled via PyMC + nutpie NUTS:
    y_h ~ Normal(X_h @ β, σ)
    β_j ~ Normal(0, 1)           (standardized features, so unit-scale priors are appropriate)
    σ   ~ HalfNormal(1)

Sampling: 4 chains, 2000 draws, 1000 tune. nutpie sampler used if available;
falls back to default PyMC NUTS if nutpie is not installed.

Output per hour:
    q10, q50, q90   — posterior predictive percentiles of DART spread ($/MWh)
    p_positive      — P(spread > 0)  [INC signal]
    p_negative      — P(spread < 0)  [DEC signal]

Walk-forward safety:
    fit() gates feature_matrix to as_of_timestamp before training.
    forecast() does NOT gate (features are already walk-forward compliant from
    the feature engineering step), but raises WalkForwardViolation on a naive
    as_of_timestamp.

Usage:
    model = DARTBayesianForecaster(feature_cols=["dart_lag_24h", "thermal_share", ...])
    model.fit(feature_matrix, target_col="dart_spread_usd", as_of_timestamp=...)
    forecasts = model.forecast(tomorrow_features, as_of_timestamp=...)
    model.save_trace(Path("output/trace"))
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import structlog

from src.ingest.exceptions import MissingDataError, WalkForwardViolation

try:
    import pymc as pm
except ImportError:
    pm = None  # type: ignore[assignment]

try:
    import arviz as az
except ImportError:
    az = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)
UTC = timezone.utc


def _gate(df: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    if as_of.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware; got naive {as_of!r}"
        )
    return df.filter(pl.col("interval_start_utc") <= as_of)


def _require_utc(ts: datetime) -> None:
    if ts.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware; got naive {ts!r}"
        )


class DARTBayesianForecaster:
    """PyMC Bayesian linear forecaster for the DART spread.

    Walk-forward safety:
        fit() gates feature_matrix to as_of_timestamp before model compilation
        and sampling. forecast() requires timezone-aware as_of_timestamp but
        does NOT re-gate (features come pre-gated from build_feature_matrix).
        Raises WalkForwardViolation on naive datetimes.
    """

    def __init__(
        self,
        feature_cols: list[str],
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        nuts_sampler: str = "nutpie",
        random_seed: int = 42,
    ) -> None:
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty")
        self.feature_cols = list(feature_cols)
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.nuts_sampler = nuts_sampler
        self.random_seed = random_seed

        self._trace = None
        self._model = None
        # Z-score scaler params (fitted from training data)
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None

    # ── public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        feature_matrix: pl.DataFrame,
        target_col: str,
        as_of_timestamp: datetime,
    ) -> "DARTBayesianForecaster":
        """Compile and sample the PyMC model on walk-forward gated training data.

        Walk-forward safety:
            feature_matrix is gated to as_of_timestamp before model build.

        Args:
            feature_matrix: Output of build_feature_matrix() with target_col present.
            target_col: Column to predict — typically 'dart_spread_usd'.
            as_of_timestamp: Walk-forward gate. Must be timezone-aware.

        Returns:
            self, for chaining.
        """
        if pm is None:
            raise ImportError("pymc is required: pip install pymc")

        gated = _gate(feature_matrix, as_of_timestamp)
        if len(gated) == 0:
            raise MissingDataError("No data available at or before as_of_timestamp")

        missing_feats = [c for c in self.feature_cols if c not in gated.columns]
        if missing_feats:
            raise MissingDataError(f"Feature columns missing: {missing_feats}")
        if target_col not in gated.columns:
            raise MissingDataError(f"Target column '{target_col}' not in feature_matrix")

        # Drop rows where any required column is null
        required = self.feature_cols + [target_col]
        gated = gated.drop_nulls(subset=required)
        if len(gated) == 0:
            raise MissingDataError("All rows null after dropping nulls on required columns")

        X_raw = gated.select(self.feature_cols).to_numpy().astype(np.float64)
        y_raw = gated[target_col].to_numpy().astype(np.float64)

        # Standardize features and target
        self._feat_mean = X_raw.mean(axis=0)
        self._feat_std = X_raw.std(axis=0) + 1e-8
        self._target_mean = float(y_raw.mean())
        self._target_std = float(y_raw.std() + 1e-8)

        X = (X_raw - self._feat_mean) / self._feat_std
        y = (y_raw - self._target_mean) / self._target_std

        n_features = X.shape[1]
        log = logger.bind(n_obs=len(y), n_features=n_features)
        log.info("bayes_fit_start", draws=self.draws, tune=self.tune, chains=self.chains)

        sampler = self._resolve_sampler()

        with pm.Model() as self._model:
            X_data = pm.Data("X", X, mutable=True)
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            mu = pm.Deterministic("mu", X_data @ beta)
            _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

            self._trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                nuts_sampler=sampler,
                random_seed=self.random_seed,
                progressbar=False,
            )

        log.info("bayes_fit_complete")
        return self

    def forecast(
        self,
        feature_matrix: pl.DataFrame,
        as_of_timestamp: datetime,
    ) -> pl.DataFrame:
        """Generate posterior predictive distribution for each row.

        Walk-forward safety:
            as_of_timestamp must be timezone-aware (backstop check).
            Features are expected to already be walk-forward compliant from
            build_feature_matrix(), but naive as_of raises WalkForwardViolation.

        Args:
            feature_matrix: DataFrame with self.feature_cols and interval_start_utc.
            as_of_timestamp: Must be timezone-aware.

        Returns:
            Polars DataFrame with columns:
                interval_start_utc, q10, q50, q90, p_positive, p_negative
        """
        _require_utc(as_of_timestamp)

        if pm is None:
            raise ImportError("pymc is required: pip install pymc")

        self._check_fitted()

        missing_feats = [c for c in self.feature_cols if c not in feature_matrix.columns]
        if missing_feats:
            raise MissingDataError(f"Feature columns missing from input: {missing_feats}")

        X_raw = feature_matrix.select(self.feature_cols).to_numpy().astype(np.float64)
        X = (X_raw - self._feat_mean) / self._feat_std

        with self._model:
            pm.set_data({"X": X})
            ppc = pm.sample_posterior_predictive(
                self._trace,
                var_names=["obs"],
                random_seed=self.random_seed,
                progressbar=False,
            )

        # ppc["obs"] shape: (chains * draws, n_hours) — in arviz format it's (chain, draw, obs)
        obs_samples = ppc.posterior_predictive["obs"].values
        # Flatten chains and draws
        flat_samples = obs_samples.reshape(-1, obs_samples.shape[-1])

        # Rescale from standardized back to original units
        flat_samples_orig = flat_samples * self._target_std + self._target_mean

        q10 = np.percentile(flat_samples_orig, 10, axis=0).tolist()
        q50 = np.percentile(flat_samples_orig, 50, axis=0).tolist()
        q90 = np.percentile(flat_samples_orig, 90, axis=0).tolist()
        p_positive = (flat_samples_orig > 0).mean(axis=0).tolist()
        p_negative = (flat_samples_orig < 0).mean(axis=0).tolist()

        return feature_matrix.select("interval_start_utc").with_columns([
            pl.Series("q10", q10),
            pl.Series("q50", q50),
            pl.Series("q90", q90),
            pl.Series("p_positive", p_positive),
            pl.Series("p_negative", p_negative),
        ])

    def save_trace(self, path: Path) -> None:
        """Persist trace (arviz InferenceData) and scaler params."""
        if az is None:
            raise ImportError("arviz is required: pip install arviz")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        az.to_netcdf(self._trace, path / "trace.nc")

        meta = {
            "feat_mean": self._feat_mean.tolist(),
            "feat_std": self._feat_std.tolist(),
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "feature_cols": self.feature_cols,
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
        }
        with (path / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        logger.info("bayes_trace_saved", path=str(path))

    def load_trace(self, path: Path) -> "DARTBayesianForecaster":
        """Load a previously saved trace and scaler params."""
        if az is None:
            raise ImportError("arviz is required: pip install arviz")

        path = Path(path)
        self._trace = az.from_netcdf(path / "trace.nc")

        with (path / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)

        self._feat_mean = np.array(meta["feat_mean"])
        self._feat_std = np.array(meta["feat_std"])
        self._target_mean = float(meta["target_mean"])
        self._target_std = float(meta["target_std"])
        self.feature_cols = meta["feature_cols"]

        logger.info("bayes_trace_loaded", path=str(path))
        return self

    # ── internal helpers ────────────────────────────────────────────────────

    def _resolve_sampler(self) -> str:
        """Use nutpie if available; fall back to default NUTS."""
        if self.nuts_sampler == "nutpie":
            try:
                import nutpie  # noqa: F401
                return "nutpie"
            except ImportError:
                logger.warning("nutpie_not_available", fallback="pymc")
                return "pymc"
        return self.nuts_sampler

    def _check_fitted(self) -> None:
        if self._trace is None or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
