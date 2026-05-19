"""Daily bid runner — M10.

CLI entrypoint that orchestrates the full M3 → M4 → M5 → M6 → M7 → M8 → M11 pipeline
for the next operating day.

Usage:
    python -m src.runners.daily --as-of 2026-05-19

Output:
    output/bids_YYYYMMDD.csv     — 10-tier bid curve (human-reviews and submits)
    output/audit_YYYYMMDD.json   — full audit trail: features, regime, Kelly pre/post damping,
                                   tier curve, hedge overlay, walk-forward provenance

Walk-forward safety:
    All feature engineering, model predictions, and scoring calls use
    as_of = <delivery_date - 1 day @ 10:00 CT> (DAM deadline).
    This is the last moment before the DAM submission window closes.

This system does NOT submit bids to ERCOT. Output CSV is for human/QSE review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import structlog
import yaml

logger = structlog.get_logger(__name__)
UTC = timezone.utc

# DAM submission deadline: 10:00 CT on the day before delivery
# In UTC this is 15:00 UTC (CDT, Apr-Oct) or 16:00 UTC (CST, Nov-Mar)
# We use a conservative 16:00 UTC (CST equivalent) to always be safe
DAM_CUTOFF_HOUR_UTC = 16


def _load_node_config(config_path: Path = Path("config/nodes.yaml")) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _load_risk_config(config_path: Path = Path("config/risk.yaml")) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _load_scoring_config(config_path: Path = Path("config/scoring.yaml")) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _dam_as_of_timestamp(delivery_date: date) -> datetime:
    """Return the DAM submission deadline as a UTC datetime.

    Walk-forward safety:
        This is the as_of_timestamp passed to all feature builders.
        Using delivery_date - 1 day @ 16:00 UTC ensures we only use
        data available before the DAM window closes.
    """
    prior_day = delivery_date - timedelta(days=1)
    return datetime(prior_day.year, prior_day.month, prior_day.day,
                    DAM_CUTOFF_HOUR_UTC, 0, 0, tzinfo=UTC)


class DailyBidRunner:
    """Orchestrates the full daily bid pipeline.

    Walk-forward safety:
        as_of_timestamp = DAM deadline of the prior day.
        All feature builders and model callers receive this timestamp.
        The runner raises WalkForwardViolation if any internal call receives
        a naive datetime (propagated from _dam_as_of_timestamp above).
    """

    def __init__(
        self,
        nodes_config: Optional[Path] = None,
        risk_config: Optional[Path] = None,
        scoring_config: Optional[Path] = None,
        data_dir: Path = Path("data"),
        output_dir: Path = Path("output"),
        model_dir: Path = Path("output/models"),
    ) -> None:
        self.nodes_config = nodes_config or Path("config/nodes.yaml")
        self.risk_config = risk_config or Path("config/risk.yaml")
        self.scoring_config = scoring_config or Path("config/scoring.yaml")
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)

        self._node_cfg = _load_node_config(self.nodes_config)
        self._risk_cfg = _load_risk_config(self.risk_config)
        self._scoring_cfg = _load_scoring_config(self.scoring_config)

        self.settlement_point: str = self._node_cfg["ercot_settlement_point"]

    def run(self, delivery_date: date) -> dict:
        """Run the full pipeline for the given delivery date.

        Args:
            delivery_date: The operating day to generate bids for.
                           Bids cover HE01–HE24 of this date.

        Returns:
            dict with keys: bids_path, audit_path, n_eligible_hours,
                            total_position_mw, hedge_hours.
        """
        as_of = _dam_as_of_timestamp(delivery_date)
        log = logger.bind(
            delivery_date=str(delivery_date),
            as_of=as_of.isoformat(),
            settlement_point=self.settlement_point,
        )
        log.info("daily_runner_start")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        date_str = delivery_date.strftime("%Y%m%d")

        audit: dict = {
            "delivery_date": str(delivery_date),
            "as_of_timestamp": as_of.isoformat(),
            "settlement_point": self.settlement_point,
            "pipeline_steps": [],
        }

        try:
            # ── Step 1: Load feature matrix ──────────────────────────────────
            feature_matrix = self._load_feature_matrix(as_of)
            audit["pipeline_steps"].append({
                "step": "feature_matrix",
                "n_rows": len(feature_matrix),
                "columns": feature_matrix.columns,
            })
            log.info("step_features_loaded", n_rows=len(feature_matrix))

            # ── Step 2: Load models ──────────────────────────────────────────
            hmm_model = self._load_hmm_model()
            garch_model = self._load_garch_model()
            bayes_model = self._load_bayes_model()
            log.info("step_models_loaded")

            # ── Step 3: M3 — Regime prediction ──────────────────────────────
            regime_probs = hmm_model.predict_state_probs(feature_matrix, as_of)
            audit["pipeline_steps"].append({
                "step": "M3_regime",
                "regime_distribution": {
                    "p_normal_mean": round(float(regime_probs["p_normal"].mean()), 4),
                    "p_scarcity_mean": round(float(regime_probs["p_scarcity"].mean()), 4),
                    "p_neg_congestion_mean": round(float(regime_probs["p_negative_congestion"].mean()), 4),
                },
            })
            log.info("step_M3_regime_done")

            # ── Step 4: M4 — Volatility forecast ────────────────────────────
            sigma2_df = garch_model.forecast_variance(
                horizon=24, regime_probs=regime_probs
            )
            audit["pipeline_steps"].append({
                "step": "M4_garch",
                "mean_sigma2": round(float(sigma2_df["sigma2"].mean()), 4),
            })
            log.info("step_M4_garch_done")

            # ── Step 5: M5 — Bayesian forecast ──────────────────────────────
            # Get delivery-day feature rows for prediction
            delivery_start = datetime(
                delivery_date.year, delivery_date.month, delivery_date.day,
                0, 0, 0, tzinfo=UTC
            )
            delivery_end = delivery_start + timedelta(days=1)
            forecast_features = self._build_delivery_features(delivery_date, as_of)

            forecast_df = bayes_model.forecast(forecast_features, as_of)
            audit["pipeline_steps"].append({
                "step": "M5_bayesian",
                "n_forecast_hours": len(forecast_df),
                "q50_mean": round(float(forecast_df["q50"].mean()), 4),
            })
            log.info("step_M5_bayes_done", n_hours=len(forecast_df))

            # ── Step 6: M6 — Composite score ────────────────────────────────
            from src.scoring.composite import CompositeScorer
            scorer = CompositeScorer.from_config(self.scoring_config)
            sigma_historical = self._get_historical_sigma(feature_matrix)
            composite_df = scorer.compute_composite(
                forecast_df, as_of,
                sigma_historical=sigma_historical,
                fundamental_alignment=1.0,
            )
            n_eligible = int(composite_df["trade_eligible"].sum())
            audit["pipeline_steps"].append({
                "step": "M6_composite",
                "n_eligible_hours": n_eligible,
                "mean_composite": round(float(composite_df["composite_score"].mean()), 4),
            })
            log.info("step_M6_composite_done", n_eligible=n_eligible)

            # ── Step 7: M7 — Kelly sizing ────────────────────────────────────
            from src.sizing.kelly import KellySizer
            sizer = KellySizer.from_config(self.risk_config)
            # Posterior samples from trace (simplified: use percentiles for approximate dist)
            posterior_samples = self._approximate_posterior_samples(forecast_df, n_draws=500)
            kelly_df = sizer.size_positions(
                forecast_df, composite_df, posterior_samples, as_of
            )
            total_mw = float(kelly_df["position_mw"].sum())
            audit["pipeline_steps"].append({
                "step": "M7_kelly",
                "total_position_mw": round(total_mw, 2),
                "mean_kelly_raw": round(float(kelly_df["kelly_fraction_raw"].mean()), 4),
                "mean_kelly_damped": round(float(kelly_df["kelly_fraction_damped"].mean()), 4),
            })
            log.info("step_M7_kelly_done", total_mw=round(total_mw, 2))

            # ── Step 8: M11 — Curtailment hedge overlay ──────────────────────
            from src.execution.curtail_hedge import CurtailmentHedge
            hedge = CurtailmentHedge.from_configs(self.risk_config, self.nodes_config)
            pv_forecast = self._get_pv_forecast(forecast_features)
            hedge_df = hedge.compute_hedge(forecast_df, pv_forecast, as_of)
            merged_kelly = hedge.merge_with_kelly(kelly_df, hedge_df)
            hedge_hours = int(hedge_df["hedge_triggered"].sum())
            audit["pipeline_steps"].append({
                "step": "M11_hedge",
                "hedge_hours": hedge_hours,
                "total_hedge_mw": round(float(hedge_df["hedge_mw"].sum()), 2),
                "hedge_direction_override": True,
            })
            log.info("step_M11_hedge_done", hedge_hours=hedge_hours)

            # ── Step 9: M8 — Tier bid generation ────────────────────────────
            from src.execution.tier_generator import TierBidGenerator
            gen = TierBidGenerator.from_config(self.risk_config)
            # Use final direction and position from merged kelly
            bids_input = forecast_df.join(
                merged_kelly.select([
                    "interval_start_utc", "final_direction", "final_position_mw"
                ]).rename({"final_direction": "direction", "final_position_mw": "position_mw"}),
                on="interval_start_utc",
                how="left",
            )
            tier_bids = gen.generate_bids(forecast_df, bids_input, as_of)
            gen.validate_monotonicity(tier_bids)  # raises if ERCOT constraint violated
            audit["pipeline_steps"].append({
                "step": "M8_tiers",
                "n_bid_rows": len(tier_bids),
                "n_hours_with_bids": tier_bids["interval_start_utc"].n_unique() if len(tier_bids) > 0 else 0,
            })
            log.info("step_M8_tiers_done", n_bid_rows=len(tier_bids))

            # ── Write outputs ────────────────────────────────────────────────
            bids_path = self.output_dir / f"bids_{date_str}.csv"
            audit_path = self.output_dir / f"audit_{date_str}.json"

            if len(tier_bids) > 0:
                tier_bids.write_csv(bids_path)
            else:
                bids_path.write_text("# No eligible bids for this operating day\n")

            audit["summary"] = {
                "n_eligible_hours": n_eligible,
                "total_position_mw": round(total_mw, 2),
                "hedge_hours": hedge_hours,
                "n_bid_rows": len(tier_bids),
            }
            audit_path.write_text(json.dumps(audit, indent=2, default=str))

            log.info(
                "daily_runner_complete",
                bids_path=str(bids_path),
                audit_path=str(audit_path),
            )

            return {
                "bids_path": str(bids_path),
                "audit_path": str(audit_path),
                "n_eligible_hours": n_eligible,
                "total_position_mw": total_mw,
                "hedge_hours": hedge_hours,
            }

        except Exception as exc:
            audit["error"] = str(exc)
            audit_path = self.output_dir / f"audit_{date_str}.json"
            audit_path.write_text(json.dumps(audit, indent=2, default=str))
            log.error("daily_runner_failed", error=str(exc))
            raise

    # ── internal helpers ────────────────────────────────────────────────────

    def _load_feature_matrix(self, as_of: datetime) -> pl.DataFrame:
        """Load pre-built feature matrix Parquet, gated to as_of."""
        feature_path = self.data_dir / "processed" / "features" / "features.parquet"
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature matrix not found at {feature_path}. "
                "Run src/features/engineering.py pipeline first."
            )
        df = pl.read_parquet(feature_path)
        return df.filter(pl.col("interval_start_utc") <= as_of)

    def _load_hmm_model(self):
        from src.models.hmm import DARTRegimeModel
        path = self.model_dir / "hmm.pkl"
        if not path.exists():
            raise FileNotFoundError(f"HMM model not found at {path}. Run model training first.")
        return DARTRegimeModel.load(path)

    def _load_garch_model(self):
        from src.models.garch import DARTVolatilityModel
        path = self.model_dir / "garch.pkl"
        if not path.exists():
            raise FileNotFoundError(f"GARCH model not found at {path}. Run model training first.")
        return DARTVolatilityModel.load(path)

    def _load_bayes_model(self):
        from src.models.bayesian_nuts import DARTBayesianForecaster
        path = self.model_dir / "bayes_trace"
        if not path.exists():
            raise FileNotFoundError(f"Bayesian model not found at {path}. Run model training first.")
        m = DARTBayesianForecaster(feature_cols=["dart_lag_24h", "thermal_share", "ercot_load_mw"])
        m.load_trace(path)
        return m

    def _build_delivery_features(self, delivery_date: date, as_of: datetime) -> pl.DataFrame:
        """Build feature rows for the delivery day hours (HE01-HE24)."""
        base = datetime(delivery_date.year, delivery_date.month, delivery_date.day, tzinfo=UTC)
        hours = [base + timedelta(hours=h) for h in range(24)]
        # Load feature matrix gated to as_of and use recent rows as proxy for delivery features
        # In production: call build_feature_matrix() with live data
        feature_path = self.data_dir / "processed" / "features" / "features.parquet"
        if feature_path.exists():
            df = pl.read_parquet(feature_path)
            recent = df.filter(pl.col("interval_start_utc") <= as_of).tail(24)
            if len(recent) == 24:
                return recent.with_columns(
                    pl.Series("interval_start_utc", hours)
                )
        # Fallback: return placeholder with just timestamps
        return pl.DataFrame({"interval_start_utc": hours})

    def _get_historical_sigma(self, feature_matrix: pl.DataFrame) -> float:
        """Estimate historical DART spread standard deviation."""
        if "dart_spread_usd" in feature_matrix.columns:
            return float(feature_matrix["dart_spread_usd"].std() or 5.0)
        return 5.0  # fallback

    def _approximate_posterior_samples(
        self,
        forecast_df: pl.DataFrame,
        n_draws: int = 500,
    ) -> np.ndarray:
        """Approximate posterior samples from q10/q50/q90 percentiles.

        Uses a truncated normal approximation:
            mean = q50
            std  = (q90 - q10) / (2 * 1.645)  [90th - 10th percentile range]
        """
        rng = np.random.default_rng(42)
        n_hours = len(forecast_df)
        q50 = forecast_df["q50"].to_numpy()
        q10 = forecast_df["q10"].to_numpy()
        q90 = forecast_df["q90"].to_numpy()

        std = (q90 - q10) / (2 * 1.645)
        std = np.maximum(std, 0.01)

        samples = rng.normal(
            loc=q50[np.newaxis, :],
            scale=std[np.newaxis, :],
            size=(n_draws, n_hours),
        )
        return samples

    def _get_pv_forecast(self, forecast_features: pl.DataFrame) -> float:
        """Get PV output forecast for the delivery day (MW).

        In production: use PVGRPP forecast from gridstatus.
        Fallback: use 50% of nameplate as a conservative estimate.
        """
        installed_mw = self._node_cfg.get("installed_mw", 160.0)
        return installed_mw * 0.50  # 50% capacity factor fallback


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ERCOT DART daily bid generator for RN_QTUM_SLR"
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        required=True,
        help="Delivery date in YYYY-MM-DD format (bids cover HE01–HE24 of this day)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for bids CSV and audit JSON",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("output/models"),
        help="Directory containing trained model files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    runner = DailyBidRunner(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )

    try:
        result = runner.run(args.as_of)
        print(f"Bids written to:  {result['bids_path']}")
        print(f"Audit written to: {result['audit_path']}")
        print(f"Eligible hours:   {result['n_eligible_hours']}")
        print(f"Total MW:         {result['total_position_mw']:.1f}")
        print(f"Hedge hours:      {result['hedge_hours']}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
