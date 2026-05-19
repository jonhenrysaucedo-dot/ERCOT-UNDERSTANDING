"""Tests for DailyBidRunner (M10).

Full pipeline is integration-tested with mocked models and minimal data.
DAM as_of timestamp, output file generation, and audit JSON are highest priority.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.runners.daily import DailyBidRunner, _dam_as_of_timestamp, main

UTC = timezone.utc


# ── DAM timestamp logic ───────────────────────────────────────────────────────

class TestDAMTimestamp:
    def test_is_prior_day(self):
        """as_of timestamp must be on the day BEFORE delivery."""
        delivery = date(2026, 5, 20)
        as_of = _dam_as_of_timestamp(delivery)
        assert as_of.date() == date(2026, 5, 19)

    def test_is_utc(self):
        as_of = _dam_as_of_timestamp(date(2026, 5, 20))
        assert as_of.tzinfo == UTC

    def test_is_timezone_aware(self):
        as_of = _dam_as_of_timestamp(date(2026, 5, 20))
        assert as_of.tzinfo is not None

    def test_hour_is_16_utc(self):
        """Deadline is 16:00 UTC (conservative CST equivalent)."""
        as_of = _dam_as_of_timestamp(date(2026, 5, 20))
        assert as_of.hour == 16


# ── DailyBidRunner — initialization ──────────────────────────────────────────

class TestInit:
    def test_reads_node_from_config(self, tmp_path):
        """settlement_point must be read from nodes.yaml, never hard-coded."""
        nodes_yaml = tmp_path / "nodes.yaml"
        risk_yaml = tmp_path / "risk.yaml"
        scoring_yaml = tmp_path / "scoring.yaml"

        nodes_yaml.write_text(
            "ercot_settlement_point: RN_QTUM_SLR\n"
            "installed_mw: 160.0\n"
            "curtailment_hedge:\n"
            "  trigger_prob: 0.30\n"
            "  coverage_ratio: 0.80\n"
            "  rec_floor_usd: -5.0\n"
        )
        risk_yaml.write_text(
            "half_kelly_multiplier: 0.50\n"
            "max_position_mw: 50.0\n"
            "uncertainty_damp_threshold: 1.0\n"
            "price_cap: 5000.0\n"
            "price_floor: -250.0\n"
            "num_offer_tiers: 10\n"
        )
        scoring_yaml.write_text(
            "w1: 0.50\nw2: 0.30\nw3: 0.20\n"
            "min_composite_score: 0.30\nmin_directional_conviction: 0.55\n"
        )

        runner = DailyBidRunner(
            nodes_config=nodes_yaml,
            risk_config=risk_yaml,
            scoring_config=scoring_yaml,
        )
        assert runner.settlement_point == "RN_QTUM_SLR"

    def test_settlement_point_not_hardcoded(self, tmp_path):
        """Different node name in YAML → different settlement_point."""
        nodes_yaml = tmp_path / "nodes.yaml"
        risk_yaml = tmp_path / "risk.yaml"
        scoring_yaml = tmp_path / "scoring.yaml"

        nodes_yaml.write_text(
            "ercot_settlement_point: HB_WEST\n"
            "installed_mw: 0.0\n"
            "curtailment_hedge:\n"
            "  trigger_prob: 0.30\n"
            "  coverage_ratio: 0.80\n"
            "  rec_floor_usd: -5.0\n"
        )
        risk_yaml.write_text(
            "half_kelly_multiplier: 0.50\nmax_position_mw: 50.0\n"
            "uncertainty_damp_threshold: 1.0\nprice_cap: 5000.0\n"
            "price_floor: -250.0\nnum_offer_tiers: 10\n"
        )
        scoring_yaml.write_text(
            "w1: 0.50\nw2: 0.30\nw3: 0.20\n"
            "min_composite_score: 0.30\nmin_directional_conviction: 0.55\n"
        )

        runner = DailyBidRunner(
            nodes_config=nodes_yaml, risk_config=risk_yaml, scoring_config=scoring_yaml,
        )
        assert runner.settlement_point == "HB_WEST"


# ── Output file generation ────────────────────────────────────────────────────

class _RunnerFixture:
    """Shared fixture: config files + minimal mocked runner."""

    def make_runner(self, tmp_path: Path) -> DailyBidRunner:
        nodes_yaml = tmp_path / "nodes.yaml"
        risk_yaml = tmp_path / "risk.yaml"
        scoring_yaml = tmp_path / "scoring.yaml"

        nodes_yaml.write_text(
            "ercot_settlement_point: RN_QTUM_SLR\n"
            "installed_mw: 160.0\n"
            "curtailment_hedge:\n"
            "  trigger_prob: 0.30\n"
            "  coverage_ratio: 0.80\n"
            "  rec_floor_usd: -5.0\n"
        )
        risk_yaml.write_text(
            "half_kelly_multiplier: 0.50\nmax_position_mw: 50.0\n"
            "uncertainty_damp_threshold: 1.0\nprice_cap: 5000.0\n"
            "price_floor: -250.0\nnum_offer_tiers: 10\n"
        )
        scoring_yaml.write_text(
            "w1: 0.50\nw2: 0.30\nw3: 0.20\n"
            "min_composite_score: 0.30\nmin_directional_conviction: 0.55\n"
        )
        return DailyBidRunner(
            nodes_config=nodes_yaml,
            risk_config=risk_yaml,
            scoring_config=scoring_yaml,
            output_dir=tmp_path / "output",
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
        )


_FAKE_MINIMIZE = type("R", (), {"x": 0.20})()


def _fake_minimize_scalar(*a, **kw):
    return _FAKE_MINIMIZE


class TestOutputFiles(_RunnerFixture):
    def _mock_runner_internals(self, runner: DailyBidRunner) -> None:
        """Patch all I/O-heavy methods to return minimal stubs."""
        base = datetime(2026, 5, 19, tzinfo=UTC)
        n = 24
        hours = [base + timedelta(hours=h) for h in range(n)]

        feature_matrix = pl.DataFrame({
            "interval_start_utc": hours,
            "dart_spread_usd": [2.0] * n,
            "thermal_share": [0.6] * n,
            "ercot_load_mw": [40000.0] * n,
            "dart_lag_24h": [1.5] * n,
        })

        # Regime probs stub
        regime_probs = pl.DataFrame({
            "interval_start_utc": hours,
            "p_normal": [0.7] * n,
            "p_scarcity": [0.2] * n,
            "p_negative_congestion": [0.1] * n,
            "regime": ["NORMAL"] * n,
        })

        # GARCH stub
        sigma2_df = pl.DataFrame({
            "hour": list(range(n)),
            "sigma2": [4.0] * n,
            "sigma2_regime_weighted": [4.5] * n,
        })

        # Forecast stub (M5 output schema)
        forecast_df = pl.DataFrame({
            "interval_start_utc": hours,
            "q10": [-2.0] * n,
            "q50": [3.0] * n,
            "q90": [8.0] * n,
            "p_positive": [0.75] * n,
            "p_negative": [0.25] * n,
        })

        mock_hmm = MagicMock()
        mock_hmm.predict_state_probs.return_value = regime_probs

        mock_garch = MagicMock()
        mock_garch.forecast_variance.return_value = sigma2_df

        mock_bayes = MagicMock()
        mock_bayes.forecast.return_value = forecast_df

        runner._load_feature_matrix = MagicMock(return_value=feature_matrix)
        runner._load_hmm_model = MagicMock(return_value=mock_hmm)
        runner._load_garch_model = MagicMock(return_value=mock_garch)
        runner._load_bayes_model = MagicMock(return_value=mock_bayes)
        runner._build_delivery_features = MagicMock(return_value=forecast_df)

    def _run_with_scipy_mock(self, runner: DailyBidRunner, delivery_date: date) -> dict:
        with patch("src.sizing.kelly._minimize_scalar", _fake_minimize_scalar):
            return runner.run(delivery_date)

    def test_creates_bids_csv(self, tmp_path):
        runner = self.make_runner(tmp_path)
        self._mock_runner_internals(runner)
        result = self._run_with_scipy_mock(runner, date(2026, 5, 20))
        bids_path = Path(result["bids_path"])
        assert bids_path.exists()

    def test_creates_audit_json(self, tmp_path):
        runner = self.make_runner(tmp_path)
        self._mock_runner_internals(runner)
        result = self._run_with_scipy_mock(runner, date(2026, 5, 20))
        audit_path = Path(result["audit_path"])
        assert audit_path.exists()

    def test_audit_json_parseable(self, tmp_path):
        import json
        runner = self.make_runner(tmp_path)
        self._mock_runner_internals(runner)
        result = self._run_with_scipy_mock(runner, date(2026, 5, 20))
        audit = json.loads(Path(result["audit_path"]).read_text())
        assert "delivery_date" in audit
        assert "as_of_timestamp" in audit
        assert "pipeline_steps" in audit

    def test_audit_records_as_of_timestamp(self, tmp_path):
        import json
        runner = self.make_runner(tmp_path)
        self._mock_runner_internals(runner)
        result = self._run_with_scipy_mock(runner, date(2026, 5, 20))
        audit = json.loads(Path(result["audit_path"]).read_text())
        # as_of should be delivery_date - 1 day @ 16:00 UTC
        assert "2026-05-19" in audit["as_of_timestamp"]

    def test_result_contains_expected_keys(self, tmp_path):
        runner = self.make_runner(tmp_path)
        self._mock_runner_internals(runner)
        result = self._run_with_scipy_mock(runner, date(2026, 5, 20))
        assert set(result.keys()) == {
            "bids_path", "audit_path", "n_eligible_hours",
            "total_position_mw", "hedge_hours",
        }

    def test_bids_filename_includes_date(self, tmp_path):
        runner = self.make_runner(tmp_path)
        self._mock_runner_internals(runner)
        result = self._run_with_scipy_mock(runner, date(2026, 5, 20))
        assert "20260520" in result["bids_path"]


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCLI(_RunnerFixture):
    def test_main_returns_zero_on_success(self, tmp_path):
        """main() exits 0 on success."""
        nodes_yaml = tmp_path / "nodes.yaml"
        risk_yaml = tmp_path / "risk.yaml"
        scoring_yaml = tmp_path / "scoring.yaml"

        nodes_yaml.write_text(
            "ercot_settlement_point: RN_QTUM_SLR\n"
            "installed_mw: 160.0\n"
            "curtailment_hedge:\n  trigger_prob: 0.30\n  coverage_ratio: 0.80\n  rec_floor_usd: -5.0\n"
        )
        risk_yaml.write_text(
            "half_kelly_multiplier: 0.50\nmax_position_mw: 50.0\n"
            "uncertainty_damp_threshold: 1.0\nprice_cap: 5000.0\nprice_floor: -250.0\nnum_offer_tiers: 10\n"
        )
        scoring_yaml.write_text(
            "w1: 0.50\nw2: 0.30\nw3: 0.20\nmin_composite_score: 0.30\nmin_directional_conviction: 0.55\n"
        )

        base = datetime(2026, 5, 19, tzinfo=UTC)
        n = 24
        hours = [base + timedelta(hours=h) for h in range(n)]
        feature_df = pl.DataFrame({
            "interval_start_utc": hours,
            "dart_spread_usd": [2.0] * n,
            "thermal_share": [0.6] * n,
            "ercot_load_mw": [40000.0] * n,
            "dart_lag_24h": [1.5] * n,
        })
        forecast_df = pl.DataFrame({
            "interval_start_utc": hours,
            "q10": [-2.0] * n, "q50": [3.0] * n, "q90": [8.0] * n,
            "p_positive": [0.75] * n, "p_negative": [0.25] * n,
        })
        regime_probs = pl.DataFrame({
            "interval_start_utc": hours,
            "p_normal": [0.7] * n, "p_scarcity": [0.2] * n,
            "p_negative_congestion": [0.1] * n, "regime": ["NORMAL"] * n,
        })
        sigma2_df = pl.DataFrame({
            "hour": list(range(n)), "sigma2": [4.0] * n, "sigma2_regime_weighted": [4.5] * n,
        })

        mock_hmm = MagicMock()
        mock_hmm.predict_state_probs.return_value = regime_probs
        mock_garch = MagicMock()
        mock_garch.forecast_variance.return_value = sigma2_df
        mock_bayes = MagicMock()
        mock_bayes.forecast.return_value = forecast_df

        with patch("src.runners.daily.DailyBidRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run.return_value = {
                "bids_path": str(tmp_path / "bids_20260520.csv"),
                "audit_path": str(tmp_path / "audit_20260520.json"),
                "n_eligible_hours": 5,
                "total_position_mw": 25.0,
                "hedge_hours": 2,
            }
            exit_code = main(["--as-of", "2026-05-20",
                              "--output-dir", str(tmp_path / "output")])

        assert exit_code == 0
