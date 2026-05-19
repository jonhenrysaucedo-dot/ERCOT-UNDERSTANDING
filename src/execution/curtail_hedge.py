"""Solar curtailment hedge overlay — M11.

When the posterior P(RT_LMP < REC_floor) > hedge_trigger_prob for an hour,
adds an INC virtual bid (sell DAM / buy RT) to offset the physical curtailment
loss. The hedge profits when DA > RT — exactly the condition that triggers
physical curtailment at the RN_QTUM_SLR solar node.

WARNING — direction is always INC:
    The whitepaper §2 stated DEC. That is an error. A DEC bid amplifies the loss
    when RT goes deeply negative. This implementation uses INC and logs
    hedge_direction_override=true in the output for audit compliance.
    See CLAUDE.md §M11 and PRD §M11 for full explanation.

Sizing:
    hedge_mw = min(PV_forecast_mw × coverage_ratio, installed_mw, max_position_mw)

Config (config/risk.yaml and config/nodes.yaml):
    hedge_trigger_prob:  0.30  — P(RT < REC_floor) threshold
    hedge_coverage_ratio: 0.80 — fraction of forecast PV to hedge
    rec_floor_usd:       -5.0  — REC value floor in $/MWh
    installed_mw:        160.0 — nameplate capacity (RN_QTUM_SLR)
    max_position_mw:     50.0  — risk limit cap

Walk-forward safety:
    compute_hedge() requires timezone-aware as_of_timestamp.
    All inputs (forecast_df, pv_forecast_mw) are from walk-forward compliant callers.
    Raises WalkForwardViolation on naive datetimes.

Usage:
    hedge = CurtailmentHedge.from_configs("config/risk.yaml", "config/nodes.yaml")
    overlay = hedge.compute_hedge(forecast_df, pv_forecast_mw, as_of_timestamp)
"""

from __future__ import annotations

import json
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


class CurtailmentHedge:
    """Solar curtailment INC hedge overlay for RN_QTUM_SLR.

    Walk-forward safety:
        compute_hedge() requires timezone-aware as_of_timestamp.
        forecast_df must come from DARTBayesianForecaster.forecast() which
        is walk-forward compliant.
        Raises WalkForwardViolation on naive datetimes.
    """

    def __init__(
        self,
        trigger_prob: float = 0.30,
        coverage_ratio: float = 0.80,
        rec_floor_usd: float = -5.0,
        installed_mw: float = 160.0,
        max_position_mw: float = 50.0,
    ) -> None:
        if not 0.0 < trigger_prob < 1.0:
            raise ValueError(f"trigger_prob must be in (0, 1); got {trigger_prob}")
        if not 0.0 < coverage_ratio <= 1.0:
            raise ValueError(f"coverage_ratio must be in (0, 1]; got {coverage_ratio}")
        if installed_mw <= 0:
            raise ValueError(f"installed_mw must be positive; got {installed_mw}")

        self.trigger_prob = trigger_prob
        self.coverage_ratio = coverage_ratio
        self.rec_floor_usd = rec_floor_usd
        self.installed_mw = installed_mw
        self.max_position_mw = max_position_mw

    @classmethod
    def from_configs(
        cls,
        risk_config_path: Union[str, Path],
        nodes_config_path: Union[str, Path],
    ) -> "CurtailmentHedge":
        """Load hedge configuration from risk.yaml and nodes.yaml."""
        with Path(risk_config_path).open() as f:
            risk = yaml.safe_load(f)
        with Path(nodes_config_path).open() as f:
            nodes = yaml.safe_load(f)

        curtail = nodes.get("curtailment_hedge", {})
        return cls(
            trigger_prob=curtail.get("trigger_prob", 0.30),
            coverage_ratio=curtail.get("coverage_ratio", 0.80),
            rec_floor_usd=curtail.get("rec_floor_usd", -5.0),
            installed_mw=nodes.get("installed_mw", 160.0),
            max_position_mw=risk.get("max_position_mw", 50.0),
        )

    def compute_hedge(
        self,
        forecast_df: pl.DataFrame,
        pv_forecast_mw: Union[float, pl.Series],
        as_of_timestamp: datetime,
        posterior_samples: Optional[np.ndarray] = None,
    ) -> pl.DataFrame:
        """Compute the INC hedge overlay for each hour.

        Walk-forward safety:
            as_of_timestamp must be timezone-aware.
            forecast_df must come from a walk-forward compliant M5 caller.

        Args:
            forecast_df: From DARTBayesianForecaster.forecast() —
                [interval_start_utc, q10, q50, q90, p_positive, p_negative].
                p_negative is used as a proxy for P(RT < REC_floor) when
                posterior_samples is not provided.
            pv_forecast_mw: Forecast PV output for each hour (MW). Scalar
                (uniform) or per-hour Series.
            as_of_timestamp: Timezone-aware.
            posterior_samples: Optional np.ndarray shape (n_draws, n_hours).
                If provided, P(RT < REC_floor) is estimated directly from
                the fraction of samples below rec_floor_usd. If None, uses
                p_negative from forecast_df as an approximation.

        Returns:
            Polars DataFrame with columns:
                interval_start_utc, hedge_triggered, hedge_mw,
                direction, p_curtail, pv_forecast_mw,
                hedge_direction_override, audit_json
        """
        _require_utc(as_of_timestamp)
        self._validate_forecast_schema(forecast_df)

        n = len(forecast_df)
        if n == 0:
            return self._empty_schema()

        # P(RT < REC_floor) per hour
        if posterior_samples is not None:
            p_curtail = (posterior_samples < self.rec_floor_usd).mean(axis=0)
        else:
            # p_negative is P(spread < 0), a conservative proxy
            p_curtail = forecast_df["p_negative"].to_numpy()

        # PV forecast array
        if isinstance(pv_forecast_mw, (int, float)):
            pv_arr = np.full(n, float(pv_forecast_mw))
        else:
            pv_arr = pv_forecast_mw.to_numpy().astype(float)

        triggered = p_curtail > self.trigger_prob

        # Sizing: min(PV_forecast × ratio, installed, max_position)
        hedge_mw = np.minimum(
            pv_arr * self.coverage_ratio,
            self.installed_mw,
        )
        hedge_mw = np.minimum(hedge_mw, self.max_position_mw)
        hedge_mw = np.where(triggered, hedge_mw, 0.0)

        # Always INC — whitepaper DEC direction is WRONG (see module docstring)
        directions = np.where(triggered, "INC", "NONE")

        n_triggered = int(triggered.sum())
        logger.info(
            "hedge_overlay_computed",
            as_of=as_of_timestamp.isoformat(),
            n_hours=n,
            n_triggered=n_triggered,
            trigger_prob=self.trigger_prob,
            hedge_direction_override=True,
        )

        audit_jsons = [
            json.dumps({
                "p_curtail": round(float(p_curtail[h]), 4),
                "trigger_prob": self.trigger_prob,
                "hedge_triggered": bool(triggered[h]),
                "pv_forecast_mw": round(float(pv_arr[h]), 2),
                "hedge_mw": round(float(hedge_mw[h]), 4),
                "direction": directions[h],
                "hedge_direction_override": True,
                "whitepaper_error_note": "PRD §M11: DEC is incorrect; INC implemented",
            })
            for h in range(n)
        ]

        return forecast_df.select("interval_start_utc").with_columns([
            pl.Series("hedge_triggered", triggered.tolist()),
            pl.Series("hedge_mw", hedge_mw.tolist()),
            pl.Series("direction", directions.tolist()),
            pl.Series("p_curtail", p_curtail.tolist()),
            pl.Series("pv_forecast_mw", pv_arr.tolist()),
            pl.lit(True).alias("hedge_direction_override"),
            pl.Series("audit_json", audit_jsons),
        ])

    def merge_with_kelly(
        self,
        kelly_allocations: pl.DataFrame,
        hedge_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge Kelly allocations with hedge overlay.

        For hours where hedge is triggered and Kelly direction conflicts:
          - INC hedge overrides Kelly direction for the hedge_mw portion
          - Remaining capacity (max_position - hedge_mw) runs the Kelly direction

        For hours where hedge is not triggered, Kelly allocations are unchanged.

        Returns:
            Polars DataFrame matching kelly_allocations schema with additional
            columns: hedge_mw, hedge_triggered, final_direction.
        """
        merged = kelly_allocations.join(
            hedge_df.select([
                "interval_start_utc", "hedge_triggered", "hedge_mw", "p_curtail"
            ]),
            on="interval_start_utc",
            how="left",
        )

        # Apply override: where hedge_triggered=True and Kelly direction != INC,
        # set direction to INC and increase/set position to hedge_mw
        def _final_direction(row: dict) -> str:
            if row.get("hedge_triggered") and row["direction"] != "INC":
                return "INC"
            return row["direction"]

        def _final_mw(row: dict) -> float:
            if row.get("hedge_triggered"):
                hedge = row.get("hedge_mw", 0.0) or 0.0
                kelly = row.get("position_mw", 0.0) or 0.0
                return max(hedge, kelly)
            return row.get("position_mw", 0.0) or 0.0

        final_directions = [_final_direction(r) for r in merged.iter_rows(named=True)]
        final_mw = [_final_mw(r) for r in merged.iter_rows(named=True)]

        return merged.with_columns([
            pl.Series("final_direction", final_directions),
            pl.Series("final_position_mw", final_mw),
        ])

    # ── internal ────────────────────────────────────────────────────────────

    def _validate_forecast_schema(self, df: pl.DataFrame) -> None:
        required = {"interval_start_utc", "p_negative"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"forecast_df missing columns: {missing}. "
                "Expected output from DARTBayesianForecaster.forecast()."
            )

    def _empty_schema(self) -> pl.DataFrame:
        return pl.DataFrame(schema={
            "interval_start_utc": pl.Datetime("us", "UTC"),
            "hedge_triggered": pl.Boolean,
            "hedge_mw": pl.Float64,
            "direction": pl.String,
            "p_curtail": pl.Float64,
            "pv_forecast_mw": pl.Float64,
            "hedge_direction_override": pl.Boolean,
            "audit_json": pl.String,
        })
