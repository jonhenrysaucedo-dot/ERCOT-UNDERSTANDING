"""Composite score — M6.

For each operating hour h, computes:
    directional_conviction_h = P(spread > 0) for INC, P(spread < 0) for DEC
    spread_magnitude_h       = |E[spread]| / σ_historical (normalized)
    fundamental_alignment_h  = ∈ [0, 1] (binary in v1: 1 = no major outage)
    composite_h = w1 * directional_conviction + w2 * spread_magnitude_norm + w3 * fundamental_alignment

Weights come from config/scoring.yaml:
    w1: 0.50  (directional conviction — most important)
    w2: 0.30  (spread magnitude)
    w3: 0.20  (fundamental alignment)

Walk-forward safety:
    compute_composite() takes as_of_timestamp for audit/logging. The posterior
    forecast and sigma_historical are already walk-forward compliant from the
    callers (M5 forecaster and feature engineering). Naive as_of raises
    WalkForwardViolation.

Composite score gate criteria (from config/scoring.yaml):
    min_composite_score:        0.30 — hours below this threshold are skipped
    min_directional_conviction: 0.55 — hours below this are skipped regardless of composite

Usage:
    scorer = CompositeScorer.from_config("config/scoring.yaml")
    scores = scorer.compute_composite(
        forecast_df, as_of_timestamp,
        sigma_historical=sigma_series, fundamental_alignment=1.0
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

logger = structlog.get_logger(__name__)
UTC = timezone.utc


def _require_utc(ts: datetime) -> None:
    if ts.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware; got naive {ts!r}"
        )


class CompositeScorer:
    """Computes per-hour composite trading scores (M6).

    Walk-forward safety:
        compute_composite() requires timezone-aware as_of_timestamp.
        Input forecast_df is expected to be walk-forward compliant (produced
        by DARTBayesianForecaster.forecast()). No data gating is performed
        here — the caller ensures walk-forward compliance.
    """

    def __init__(
        self,
        w1: float = 0.50,
        w2: float = 0.30,
        w3: float = 0.20,
        min_composite_score: float = 0.30,
        min_directional_conviction: float = 0.55,
    ) -> None:
        if abs(w1 + w2 + w3 - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0; got {w1+w2+w3:.4f}")
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.min_composite_score = min_composite_score
        self.min_directional_conviction = min_directional_conviction

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "CompositeScorer":
        """Load scorer configuration from YAML."""
        with Path(config_path).open() as f:
            cfg = yaml.safe_load(f)
        return cls(
            w1=cfg["w1"],
            w2=cfg["w2"],
            w3=cfg["w3"],
            min_composite_score=cfg["min_composite_score"],
            min_directional_conviction=cfg["min_directional_conviction"],
        )

    def compute_composite(
        self,
        forecast_df: pl.DataFrame,
        as_of_timestamp: datetime,
        sigma_historical: Optional[Union[float, pl.Series]] = None,
        fundamental_alignment: Union[float, pl.Series] = 1.0,
    ) -> pl.DataFrame:
        """Compute composite score for each hour in forecast_df.

        Walk-forward safety:
            as_of_timestamp is timezone-aware (enforced). The forecast_df
            is assumed to come from M5 forecast(), which is already walk-forward
            compliant. as_of is used only for audit logging.

        Args:
            forecast_df: Output of DARTBayesianForecaster.forecast() with columns:
                [interval_start_utc, q10, q50, q90, p_positive, p_negative]
            as_of_timestamp: Timezone-aware. Used for audit logging only.
            sigma_historical: Historical σ(DART) for spread magnitude normalization.
                Scalar or per-hour Series. If None, magnitude component uses |q50|
                directly (unnormalized).
            fundamental_alignment: ∈ [0, 1]. Binary in v1: 1.0 = no major outage.
                Scalar or per-hour Series.

        Returns:
            Polars DataFrame with columns:
                interval_start_utc, direction, directional_conviction,
                spread_magnitude, spread_magnitude_norm, fundamental_alignment,
                composite_score, trade_eligible
        """
        _require_utc(as_of_timestamp)
        self._validate_forecast_schema(forecast_df)

        n = len(forecast_df)
        if n == 0:
            return pl.DataFrame(schema={
                "interval_start_utc": pl.Datetime("us", "UTC"),
                "direction": pl.String,
                "directional_conviction": pl.Float64,
                "spread_magnitude": pl.Float64,
                "spread_magnitude_norm": pl.Float64,
                "fundamental_alignment": pl.Float64,
                "composite_score": pl.Float64,
                "trade_eligible": pl.Boolean,
            })

        q50 = forecast_df["q50"].to_numpy()
        p_pos = forecast_df["p_positive"].to_numpy()
        p_neg = forecast_df["p_negative"].to_numpy()

        # Direction: INC when P(spread > 0) >= P(spread < 0), else DEC
        directions = np.where(p_pos >= p_neg, "INC", "DEC")

        # Directional conviction: P for whichever direction was chosen
        directional_conviction = np.where(p_pos >= p_neg, p_pos, p_neg)

        # Spread magnitude (un-normalized)
        spread_mag = np.abs(q50)

        # Normalize magnitude by historical sigma
        if sigma_historical is None:
            # Fallback: normalize by own max across the horizon
            denom = spread_mag.max() + 1e-6
            spread_mag_norm = spread_mag / denom
        elif isinstance(sigma_historical, (int, float)):
            spread_mag_norm = spread_mag / (float(sigma_historical) + 1e-6)
            spread_mag_norm = np.clip(spread_mag_norm, 0.0, 1.0)
        else:
            sigma_arr = sigma_historical.to_numpy().astype(float)
            spread_mag_norm = spread_mag / (sigma_arr + 1e-6)
            spread_mag_norm = np.clip(spread_mag_norm, 0.0, 1.0)

        # Fundamental alignment
        if isinstance(fundamental_alignment, (int, float)):
            fa_arr = np.full(n, float(fundamental_alignment))
        else:
            fa_arr = fundamental_alignment.to_numpy().astype(float)

        # Composite score
        composite = (
            self.w1 * directional_conviction
            + self.w2 * spread_mag_norm
            + self.w3 * fa_arr
        )

        # Eligibility gates
        eligible = (
            (composite >= self.min_composite_score)
            & (directional_conviction >= self.min_directional_conviction)
        )

        logger.info(
            "composite_scores_computed",
            as_of=as_of_timestamp.isoformat(),
            n_hours=n,
            n_eligible=int(eligible.sum()),
        )

        return forecast_df.select("interval_start_utc").with_columns([
            pl.Series("direction", directions.tolist()),
            pl.Series("directional_conviction", directional_conviction.tolist()),
            pl.Series("spread_magnitude", spread_mag.tolist()),
            pl.Series("spread_magnitude_norm", spread_mag_norm.tolist()),
            pl.Series("fundamental_alignment", fa_arr.tolist()),
            pl.Series("composite_score", composite.tolist()),
            pl.Series("trade_eligible", eligible.tolist()),
        ])

    # ── internal ────────────────────────────────────────────────────────────

    def _validate_forecast_schema(self, df: pl.DataFrame) -> None:
        required = {"interval_start_utc", "q50", "p_positive", "p_negative"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"forecast_df missing required columns: {missing}. "
                "Expected output from DARTBayesianForecaster.forecast()."
            )
