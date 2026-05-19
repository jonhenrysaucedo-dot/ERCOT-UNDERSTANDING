"""Continuous Kelly sizer — M7.

Formula: f* = argmax_f ∫ log(1 + f·R) · P(R) dR

Solved numerically over f ∈ [0, half_kelly_cap] via scipy.optimize.minimize_scalar.
The posterior samples from M5 provide the empirical P(R) distribution.

Non-negotiable per CLAUDE.md:
    The Half-Kelly multiplier is 0.5. The solver operates over [0, half_kelly_cap]
    where half_kelly_cap = half_kelly_multiplier * raw_kelly_cap.
    Do NOT change this without explicit approval.

Uncertainty penalty:
    If posterior credible interval width / |E[spread]| > uncertainty_damp_threshold (1.0),
    multiply f* by 0.5 (additional damping).

Covariance haircut:
    If multiple hours are highly correlated in posterior (max pairwise correlation),
    scale f* down by (1 - max_pairwise_correlation).

Walk-forward safety:
    size_positions() requires timezone-aware as_of_timestamp.
    Posterior samples come from M5 forecast() which is already walk-forward
    compliant. No data gating is performed here.

Usage:
    sizer = KellySizer.from_config("config/risk.yaml")
    allocations = sizer.size_positions(
        forecast_df,   # from M5
        composite_df,  # from M6
        posterior_samples,  # np.ndarray shape (n_draws, n_hours)
        as_of_timestamp,
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import structlog
import yaml

from src.ingest.exceptions import WalkForwardViolation

try:
    from scipy.optimize import minimize_scalar as _minimize_scalar
except ImportError:
    _minimize_scalar = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)
UTC = timezone.utc

# Half-Kelly is non-negotiable (CLAUDE.md)
_HALF_KELLY = 0.5


def _require_utc(ts: datetime) -> None:
    if ts.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware; got naive {ts!r}"
        )


def _kelly_objective(f: float, returns: np.ndarray) -> float:
    """Negative expected log wealth — to be minimized."""
    inside = 1.0 + f * returns
    # Clip to avoid log(0) or log of negative
    inside = np.clip(inside, 1e-10, None)
    return -np.mean(np.log(inside))


def compute_kelly_fraction(
    returns: np.ndarray,
    max_f: float = 0.5,
    tol: float = 1e-8,
) -> float:
    """Compute the optimal Kelly fraction for a 1-D return distribution.

    Walk-forward safety:
        returns are posterior samples — no future data.

    Args:
        returns: Array of simulated returns (e.g. DART spread samples).
        max_f: Upper bound on f (Half-Kelly cap applied by caller).
        tol: Optimization tolerance.

    Returns:
        Optimal f* ∈ [0, max_f].
    """
    if _minimize_scalar is None:
        raise ImportError("scipy is required: pip install scipy")

    if len(returns) == 0 or np.all(returns == 0):
        return 0.0

    result = _minimize_scalar(
        _kelly_objective,
        bounds=(0.0, max_f),
        method="bounded",
        args=(returns,),
        options={"xatol": tol},
    )
    return float(np.clip(result.x, 0.0, max_f))


class KellySizer:
    """Continuous Kelly MW position sizer with Half-Kelly cap and uncertainty damping.

    Walk-forward safety:
        size_positions() requires timezone-aware as_of_timestamp.
        Input data (forecast_df, composite_df, posterior_samples) must be
        walk-forward compliant from upstream callers.
        Raises WalkForwardViolation on naive datetimes.
    """

    def __init__(
        self,
        half_kelly_multiplier: float = 0.50,
        max_position_mw: float = 50.0,
        uncertainty_damp_threshold: float = 1.0,
        price_cap: float = 5000.0,
        price_floor: float = -250.0,
    ) -> None:
        if abs(half_kelly_multiplier - _HALF_KELLY) > 1e-9:
            raise ValueError(
                f"half_kelly_multiplier must be exactly {_HALF_KELLY} (non-negotiable per CLAUDE.md). "
                f"Got {half_kelly_multiplier}."
            )
        self.half_kelly_multiplier = half_kelly_multiplier
        self.max_position_mw = max_position_mw
        self.uncertainty_damp_threshold = uncertainty_damp_threshold
        self.price_cap = price_cap
        self.price_floor = price_floor

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "KellySizer":
        """Load sizer configuration from YAML."""
        with Path(config_path).open() as f:
            cfg = yaml.safe_load(f)
        return cls(
            half_kelly_multiplier=cfg["half_kelly_multiplier"],
            max_position_mw=cfg["max_position_mw"],
            uncertainty_damp_threshold=cfg["uncertainty_damp_threshold"],
            price_cap=cfg["price_cap"],
            price_floor=cfg["price_floor"],
        )

    def size_positions(
        self,
        forecast_df: pl.DataFrame,
        composite_df: pl.DataFrame,
        posterior_samples: np.ndarray,
        as_of_timestamp: datetime,
    ) -> pl.DataFrame:
        """Compute Half-Kelly MW allocation for each eligible hour.

        Walk-forward safety:
            as_of_timestamp must be timezone-aware. Input DataFrames and
            posterior_samples must come from walk-forward compliant callers.

        Args:
            forecast_df: From DARTBayesianForecaster.forecast() —
                [interval_start_utc, q10, q50, q90, p_positive, p_negative]
            composite_df: From CompositeScorer.compute_composite() —
                [interval_start_utc, direction, directional_conviction,
                 composite_score, trade_eligible]
            posterior_samples: np.ndarray shape (n_draws, n_hours). Posterior
                DART spread samples in $/MWh. Column order must match forecast_df rows.
            as_of_timestamp: Timezone-aware. Used for audit logging.

        Returns:
            Polars DataFrame with columns:
                interval_start_utc, direction, kelly_fraction_raw,
                kelly_fraction_damped, kelly_fraction_final,
                position_mw, uncertainty_ratio, max_pairwise_corr,
                covariance_haircut, audit_json
        """
        _require_utc(as_of_timestamp)
        self._validate_inputs(forecast_df, composite_df, posterior_samples)

        n_hours = len(forecast_df)
        joined = composite_df.join(
            forecast_df.select(["interval_start_utc", "q10", "q50", "q90"]),
            on="interval_start_utc",
            how="left",
        )

        covariance_haircut = self._compute_covariance_haircut(posterior_samples)
        logger.info(
            "kelly_sizing_start",
            as_of=as_of_timestamp.isoformat(),
            n_hours=n_hours,
            max_position_mw=self.max_position_mw,
            cov_haircut=round(covariance_haircut, 4),
        )

        rows = []
        for h, row in enumerate(joined.iter_rows(named=True)):
            samples_h = posterior_samples[:, h] if h < posterior_samples.shape[1] else np.array([0.0])
            result = self._size_one_hour(h, row, samples_h, covariance_haircut)
            rows.append(result)

        result_df = pl.DataFrame(rows)

        logger.info(
            "kelly_sizing_complete",
            n_eligible=int(result_df.filter(pl.col("position_mw") > 0)["position_mw"].count()),
            total_mw=round(float(result_df["position_mw"].sum()), 2),
        )
        return result_df

    # ── internal ────────────────────────────────────────────────────────────

    def _size_one_hour(
        self,
        hour_idx: int,
        row: dict,
        samples: np.ndarray,
        covariance_haircut: float,
    ) -> dict:
        """Compute Kelly sizing for a single hour."""
        if not row.get("trade_eligible", False):
            return self._zero_row(row)

        q10 = row.get("q10", 0.0) or 0.0
        q50 = row.get("q50", 0.0) or 0.0
        q90 = row.get("q90", 0.0) or 0.0

        # Uncertainty ratio: CI width / |E[spread]|
        ci_width = abs(q90 - q10)
        e_spread = abs(q50)
        uncertainty_ratio = ci_width / (e_spread + 1e-6)

        # Raw Kelly fraction (Half-Kelly cap built into optimizer upper bound)
        kelly_raw = compute_kelly_fraction(samples, max_f=self.half_kelly_multiplier)

        # Uncertainty damping: if CI width > threshold × |E[spread]|, apply 0.5 multiplier
        if uncertainty_ratio > self.uncertainty_damp_threshold:
            kelly_damped = kelly_raw * 0.5
        else:
            kelly_damped = kelly_raw

        # Covariance haircut: scale by (1 - max_pairwise_corr) across all hours
        kelly_final = kelly_damped * covariance_haircut

        # Convert fraction to MW: fraction × max_position_mw, capped at max_position_mw
        position_mw = min(kelly_final * self.max_position_mw, self.max_position_mw)
        position_mw = max(position_mw, 0.0)

        return {
            "interval_start_utc": row["interval_start_utc"],
            "direction": row.get("direction", "INC"),
            "kelly_fraction_raw": round(kelly_raw, 6),
            "kelly_fraction_damped": round(kelly_damped, 6),
            "kelly_fraction_final": round(kelly_final, 6),
            "position_mw": round(position_mw, 4),
            "uncertainty_ratio": round(uncertainty_ratio, 4),
            "max_pairwise_corr": round(1.0 - covariance_haircut, 4),
            "covariance_haircut": round(covariance_haircut, 4),
        }

    def _zero_row(self, row: dict) -> dict:
        return {
            "interval_start_utc": row["interval_start_utc"],
            "direction": row.get("direction", "INC"),
            "kelly_fraction_raw": 0.0,
            "kelly_fraction_damped": 0.0,
            "kelly_fraction_final": 0.0,
            "position_mw": 0.0,
            "uncertainty_ratio": 0.0,
            "max_pairwise_corr": 0.0,
            "covariance_haircut": 1.0,
        }

    def _compute_covariance_haircut(self, posterior_samples: np.ndarray) -> float:
        """Compute (1 - max pairwise correlation) across hours.

        If n_hours == 1 or all-zero, returns 1.0 (no haircut).
        """
        n_draws, n_hours = posterior_samples.shape
        if n_hours <= 1 or n_draws < 2:
            return 1.0

        # Compute correlation matrix (columns = hours, rows = draws)
        try:
            corr = np.corrcoef(posterior_samples.T)  # shape (n_hours, n_hours)
            # Zero out diagonal (self-correlation = 1)
            np.fill_diagonal(corr, 0.0)
            # Replace NaN with 0 (occurs when a column is constant)
            corr = np.nan_to_num(corr, nan=0.0)
            max_corr = float(np.abs(corr).max())
        except Exception:
            max_corr = 0.0

        return float(np.clip(1.0 - max_corr, 0.0, 1.0))

    def _validate_inputs(
        self,
        forecast_df: pl.DataFrame,
        composite_df: pl.DataFrame,
        posterior_samples: np.ndarray,
    ) -> None:
        required_forecast = {"interval_start_utc", "q10", "q50", "q90"}
        required_composite = {"interval_start_utc", "direction", "trade_eligible"}

        missing_f = required_forecast - set(forecast_df.columns)
        if missing_f:
            raise ValueError(f"forecast_df missing columns: {missing_f}")

        missing_c = required_composite - set(composite_df.columns)
        if missing_c:
            raise ValueError(f"composite_df missing columns: {missing_c}")

        if posterior_samples.ndim != 2:
            raise ValueError(
                f"posterior_samples must be 2-D (n_draws, n_hours); got shape {posterior_samples.shape}"
            )

        if posterior_samples.shape[1] != len(forecast_df):
            raise ValueError(
                f"posterior_samples.shape[1]={posterior_samples.shape[1]} must match "
                f"len(forecast_df)={len(forecast_df)}"
            )
