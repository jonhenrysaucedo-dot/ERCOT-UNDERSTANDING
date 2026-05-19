"""Volatility model — M4.

GARCH(1,1) on the DART spread, with Markov-Switching variance scaling:
    σ²_adj(t) = σ²_GARCH(t) × Σ_k P(regime=k | t) × regime_var_scale[k]

Where regime_var_scale[k] is estimated from training residuals per regime:
    regime_var_scale[k] = mean(ε²[regime==k]) / mean(ε²)

This provides regime-conditional variance without a full MS-GARCH implementation,
matching the PRD's intent: "variance conditionally scaled by HMM state probability."

Walk-forward safety:
    fit() gates dart_spread to as_of_timestamp before fitting.
    forecast_variance() takes an optional horizon argument and does not look
    at future data — it is a pure forward projection from the fitted model.

Usage:
    model = DARTVolatilityModel()
    model.fit(dart_df, as_of_timestamp=datetime(2025, 1, 1, tzinfo=UTC))
    var_df = model.forecast_variance(horizon=24, regime_probs=probs_df)
    model.save(Path("output/garch.pkl"))
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
    from arch import arch_model
except ImportError:
    arch_model = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)
UTC = timezone.utc

# Default regime variance scaling factors (overridden by fit when regime labels provided)
DEFAULT_REGIME_SCALES = {
    "NORMAL": 1.0,
    "SCARCITY": 1.5,
    "NEGATIVE_CONGESTION": 2.0,
}


def _gate(df: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    if as_of.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware; got naive {as_of!r}"
        )
    return df.filter(pl.col("interval_start_utc") <= as_of)


class DARTVolatilityModel:
    """GARCH(1,1) volatility model with regime-conditional variance scaling.

    Walk-forward safety:
        fit() gates dart_spread to as_of_timestamp.
        forecast_variance() is a pure forward projection — no future data.
        Raises WalkForwardViolation if as_of_timestamp is naive.
    """

    def __init__(self, p: int = 1, q: int = 1, random_state: int = 42) -> None:
        self.p = p
        self.q = q
        self.random_state = random_state

        self._result = None
        self._regime_var_scales: dict[str, float] = dict(DEFAULT_REGIME_SCALES)
        self._last_return: Optional[float] = None
        self._last_variance: Optional[float] = None

    # ── public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        dart_spread: pl.DataFrame,
        as_of_timestamp: datetime,
        regime_labels: Optional[pl.DataFrame] = None,
    ) -> "DARTVolatilityModel":
        """Fit GARCH(1,1) on walk-forward gated DART spread.

        Walk-forward safety:
            dart_spread is gated to as_of_timestamp before fitting.

        Args:
            dart_spread: DataFrame with [interval_start_utc, dart_spread_usd].
            as_of_timestamp: Walk-forward gate. Must be timezone-aware.
            regime_labels: Optional DataFrame from DARTRegimeModel.decode_states()
                with [interval_start_utc, regime]. If provided, regime variance
                scales are estimated from residuals per regime.

        Returns:
            self, for chaining.
        """
        if arch_model is None:
            raise ImportError("arch package is required: pip install arch")

        gated = _gate(dart_spread, as_of_timestamp)
        if len(gated) < 10:
            raise MissingDataError(
                f"Too few observations after gating: {len(gated)}. Need ≥10."
            )
        if "dart_spread_usd" not in gated.columns:
            raise MissingDataError("dart_spread must contain 'dart_spread_usd' column")

        series = gated["dart_spread_usd"].to_numpy().astype(np.float64)

        log = logger.bind(n_obs=len(series), p=self.p, q=self.q)
        log.info("garch_fit_start")

        am = arch_model(series, mean="Constant", vol="GARCH", p=self.p, q=self.q)
        self._result = am.fit(disp="off")

        # Store last observation for forecasting continuity
        self._last_return = float(series[-1])
        resid = self._result.resid
        self._last_variance = float(self._result.conditional_volatility[-1] ** 2)

        if regime_labels is not None:
            self._estimate_regime_scales(gated, regime_labels, resid)

        log.info(
            "garch_fit_complete",
            aic=round(float(self._result.aic), 2),
            bic=round(float(self._result.bic), 2),
        )
        return self

    def forecast_variance(
        self,
        horizon: int,
        regime_probs: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Forecast conditional variance for next `horizon` hours.

        Args:
            horizon: Number of hours to forecast (1–24 typical).
            regime_probs: Optional DataFrame with columns [p_normal, p_scarcity,
                p_negative_congestion] — one row per forecast hour (or one row
                applied uniformly). If None, regime scaling is not applied.

        Returns:
            Polars DataFrame with columns:
                hour (0-indexed), sigma2 (raw GARCH variance),
                sigma2_regime_weighted (regime-adjusted variance if regime_probs provided)
        """
        self._check_fitted()
        if horizon < 1:
            raise ValueError(f"horizon must be ≥1, got {horizon}")

        forecasts = self._result.forecast(horizon=horizon, reindex=False)
        # arch forecast returns variance array shape (1, horizon)
        raw_var = forecasts.variance.values[-1, :horizon].astype(float)

        hours = list(range(horizon))
        sigma2 = raw_var.tolist()

        if regime_probs is not None:
            sigma2_adj = self._apply_regime_scaling(raw_var, regime_probs, horizon)
        else:
            sigma2_adj = sigma2

        return pl.DataFrame({
            "hour": hours,
            "sigma2": sigma2,
            "sigma2_regime_weighted": sigma2_adj,
        })

    def save(self, path: Path) -> None:
        """Persist the fitted model to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("garch_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "DARTVolatilityModel":
        """Load a previously saved model."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected DARTVolatilityModel, got {type(obj)}")
        logger.info("garch_loaded", path=str(path))
        return obj

    # ── internal helpers ────────────────────────────────────────────────────

    def _estimate_regime_scales(
        self,
        gated: pl.DataFrame,
        regime_labels: pl.DataFrame,
        residuals: np.ndarray,
    ) -> None:
        """Compute per-regime variance scaling factors from training residuals."""
        joined = (
            gated.select("interval_start_utc")
            .with_columns(pl.Series("resid2", residuals ** 2))
            .join(regime_labels.select(["interval_start_utc", "regime"]), on="interval_start_utc", how="inner")
        )
        overall_mean_sq = float(joined["resid2"].mean() or 1.0)

        for regime_name in DEFAULT_REGIME_SCALES:
            sub = joined.filter(pl.col("regime") == regime_name)
            if len(sub) > 0:
                regime_mean_sq = float(sub["resid2"].mean())
                self._regime_var_scales[regime_name] = regime_mean_sq / (overall_mean_sq + 1e-10)

        logger.info("garch_regime_scales_estimated", scales=self._regime_var_scales)

    def _apply_regime_scaling(
        self,
        raw_var: np.ndarray,
        regime_probs: pl.DataFrame,
        horizon: int,
    ) -> list[float]:
        """Apply regime-weighted variance scaling."""
        n_regime_rows = len(regime_probs)
        result = []
        for h in range(horizon):
            row_idx = min(h, n_regime_rows - 1)
            row = regime_probs.row(row_idx, named=True)
            p_normal = row.get("p_normal", 1.0 / 3)
            p_scarcity = row.get("p_scarcity", 1.0 / 3)
            p_neg = row.get("p_negative_congestion", 1.0 / 3)
            scale = (
                p_normal * self._regime_var_scales["NORMAL"]
                + p_scarcity * self._regime_var_scales["SCARCITY"]
                + p_neg * self._regime_var_scales["NEGATIVE_CONGESTION"]
            )
            result.append(float(raw_var[h] * scale))
        return result

    def _check_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
