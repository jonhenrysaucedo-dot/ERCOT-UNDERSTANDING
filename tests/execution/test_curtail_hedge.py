"""Tests for CurtailmentHedge (M11).

INC direction and trigger/sizing logic are highest priority.
The whitepaper DEC error is documented and tested.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.ingest.exceptions import WalkForwardViolation
from src.execution.curtail_hedge import CurtailmentHedge

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_forecast(n: int = 6, p_neg: float = 0.20) -> pl.DataFrame:
    base = datetime(2025, 6, 1, tzinfo=UTC)
    return pl.DataFrame({
        "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
        "q10": [-5.0] * n,
        "q50": [2.0] * n,
        "q90": [8.0] * n,
        "p_positive": [1.0 - p_neg] * n,
        "p_negative": [p_neg] * n,
    })


def _default_hedge() -> CurtailmentHedge:
    return CurtailmentHedge(
        trigger_prob=0.30,
        coverage_ratio=0.80,
        rec_floor_usd=-5.0,
        installed_mw=160.0,
        max_position_mw=50.0,
    )


# ── Initialization ────────────────────────────────────────────────────────────

class TestInit:
    def test_trigger_prob_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CurtailmentHedge(trigger_prob=1.5)

    def test_trigger_prob_zero_raises(self):
        with pytest.raises(ValueError):
            CurtailmentHedge(trigger_prob=0.0)

    def test_coverage_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CurtailmentHedge(coverage_ratio=1.5)

    def test_installed_mw_must_be_positive(self):
        with pytest.raises(ValueError):
            CurtailmentHedge(installed_mw=0.0)

    def test_valid_init(self):
        h = _default_hedge()
        assert h.trigger_prob == 0.30
        assert h.installed_mw == 160.0


# ── Walk-forward gate ─────────────────────────────────────────────────────────

class TestWalkForwardGate:
    def test_naive_datetime_raises(self):
        h = _default_hedge()
        with pytest.raises(WalkForwardViolation):
            h.compute_hedge(_make_forecast(), 100.0, datetime(2025, 6, 2))  # naive

    def test_utc_datetime_accepted(self):
        h = _default_hedge()
        result = h.compute_hedge(_make_forecast(), 100.0, datetime(2025, 6, 2, tzinfo=UTC))
        assert len(result) == 6


# ── Output schema ─────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_required_columns(self):
        h = _default_hedge()
        result = h.compute_hedge(_make_forecast(), 100.0, datetime(2025, 6, 2, tzinfo=UTC))
        expected = {
            "interval_start_utc", "hedge_triggered", "hedge_mw",
            "direction", "p_curtail", "pv_forecast_mw",
            "hedge_direction_override", "audit_json",
        }
        assert set(result.columns) == expected

    def test_empty_forecast_returns_empty(self):
        h = _default_hedge()
        empty = pl.DataFrame(schema={
            "interval_start_utc": pl.Datetime("us", "UTC"),
            "q10": pl.Float64, "q50": pl.Float64, "q90": pl.Float64,
            "p_positive": pl.Float64, "p_negative": pl.Float64,
        })
        result = h.compute_hedge(empty, 100.0, datetime(2025, 6, 2, tzinfo=UTC))
        assert len(result) == 0

    def test_hedge_direction_override_always_true(self):
        """audit trail must show hedge_direction_override=True for all rows."""
        h = _default_hedge()
        result = h.compute_hedge(_make_forecast(), 100.0, datetime(2025, 6, 2, tzinfo=UTC))
        assert result["hedge_direction_override"].all()


# ── INC direction enforcement ─────────────────────────────────────────────────

class TestINCDirection:
    def test_triggered_hours_always_inc(self):
        """When hedge is triggered, direction must be INC (never DEC)."""
        h = CurtailmentHedge(trigger_prob=0.10)  # low threshold → always triggered
        result = h.compute_hedge(
            _make_forecast(n=6, p_neg=0.50),  # p_neg=0.50 > trigger_prob=0.10
            100.0,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        triggered = result.filter(pl.col("hedge_triggered"))
        assert (triggered["direction"] == "INC").all()

    def test_untriggered_hours_are_none(self):
        h = CurtailmentHedge(trigger_prob=0.90)  # high threshold → never triggered
        result = h.compute_hedge(
            _make_forecast(n=6, p_neg=0.10),  # p_neg << trigger_prob
            100.0,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert (result["direction"] == "NONE").all()
        assert not result["hedge_triggered"].any()

    def test_untriggered_hours_have_zero_hedge_mw(self):
        h = CurtailmentHedge(trigger_prob=0.90)
        result = h.compute_hedge(
            _make_forecast(n=6, p_neg=0.05),
            100.0,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert (result["hedge_mw"] == 0.0).all()


# ── Trigger logic ─────────────────────────────────────────────────────────────

class TestTriggerLogic:
    def test_high_p_neg_triggers_hedge(self):
        """p_neg=0.70 > trigger_prob=0.30 → triggered."""
        h = _default_hedge()
        result = h.compute_hedge(
            _make_forecast(n=3, p_neg=0.70),
            100.0,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert result["hedge_triggered"].all()

    def test_low_p_neg_no_trigger(self):
        h = _default_hedge()
        result = h.compute_hedge(
            _make_forecast(n=3, p_neg=0.05),
            100.0,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert not result["hedge_triggered"].any()

    def test_posterior_samples_override_p_negative(self):
        """When posterior_samples provided, p_curtail is computed directly."""
        h = CurtailmentHedge(trigger_prob=0.30, rec_floor_usd=-5.0)
        # Samples where 60% are below -5.0 → should trigger
        rng = np.random.default_rng(0)
        samples_low = np.full((100, 3), -10.0)  # all below -5.0 → p_curtail=1.0
        result = h.compute_hedge(
            _make_forecast(n=3, p_neg=0.05),  # p_neg alone wouldn't trigger
            100.0,
            datetime(2025, 6, 2, tzinfo=UTC),
            posterior_samples=samples_low,
        )
        assert result["hedge_triggered"].all()
        # p_curtail should be 1.0 (all samples below rec_floor)
        assert abs(result["p_curtail"][0] - 1.0) < 1e-9


# ── Sizing ────────────────────────────────────────────────────────────────────

class TestSizing:
    def test_hedge_mw_capped_at_max_position(self):
        """hedge_mw must never exceed max_position_mw."""
        h = CurtailmentHedge(
            trigger_prob=0.10,
            coverage_ratio=0.80,
            installed_mw=160.0,
            max_position_mw=30.0,  # cap
        )
        # PV forecast = 200 MW × 0.8 = 160, then min(160, 160, 30) = 30
        result = h.compute_hedge(
            _make_forecast(n=3, p_neg=0.80),
            200.0,  # high PV → coverage_ratio * pv > max_position
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert (result["hedge_mw"] <= 30.0 + 1e-6).all()

    def test_hedge_mw_capped_at_installed(self):
        """hedge_mw must never exceed installed_mw nameplate capacity."""
        h = CurtailmentHedge(
            trigger_prob=0.10,
            coverage_ratio=0.80,
            installed_mw=50.0,
            max_position_mw=200.0,
        )
        # PV forecast = 200 MW × 0.8 = 160, then min(160, 50, 200) = 50
        result = h.compute_hedge(
            _make_forecast(n=3, p_neg=0.80),
            200.0,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert (result["hedge_mw"] <= 50.0 + 1e-6).all()

    def test_hedge_mw_scales_with_pv_forecast(self):
        """Smaller PV forecast → smaller hedge_mw."""
        h = CurtailmentHedge(trigger_prob=0.10, coverage_ratio=0.80, max_position_mw=200.0)
        r_small = h.compute_hedge(_make_forecast(n=1, p_neg=0.80), 50.0, datetime(2025, 6, 2, tzinfo=UTC))
        r_large = h.compute_hedge(_make_forecast(n=1, p_neg=0.80), 150.0, datetime(2025, 6, 2, tzinfo=UTC))
        assert r_small["hedge_mw"][0] < r_large["hedge_mw"][0]

    def test_per_hour_pv_forecast_series(self):
        """Per-hour Series for pv_forecast_mw should be respected."""
        h = _default_hedge()
        pv_series = pl.Series([50.0, 100.0, 150.0, 100.0, 50.0, 0.0])
        result = h.compute_hedge(
            _make_forecast(n=6, p_neg=0.80),
            pv_series,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        # Hour with pv=0 → hedge_mw=0 (min(0*0.8, ...) = 0)
        assert result["hedge_mw"][5] == 0.0


# ── Audit JSON ────────────────────────────────────────────────────────────────

class TestAuditJSON:
    def test_audit_json_parseable(self):
        import json
        h = _default_hedge()
        result = h.compute_hedge(_make_forecast(), 100.0, datetime(2025, 6, 2, tzinfo=UTC))
        for j in result["audit_json"].to_list():
            parsed = json.loads(j)
            assert "hedge_direction_override" in parsed
            assert parsed["hedge_direction_override"] is True
            assert "whitepaper_error_note" in parsed  # documents the DEC→INC correction

    def test_audit_records_trigger_prob(self):
        import json
        h = _default_hedge()
        result = h.compute_hedge(_make_forecast(), 100.0, datetime(2025, 6, 2, tzinfo=UTC))
        for j in result["audit_json"].to_list():
            parsed = json.loads(j)
            assert parsed["trigger_prob"] == 0.30


# ── merge_with_kelly ──────────────────────────────────────────────────────────

class TestMergeWithKelly:
    def _kelly_alloc(self, n: int = 6) -> pl.DataFrame:
        base = datetime(2025, 6, 1, tzinfo=UTC)
        return pl.DataFrame({
            "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
            "direction": ["DEC"] * n,
            "position_mw": [15.0] * n,
            "kelly_fraction_raw": [0.20] * n,
            "kelly_fraction_damped": [0.20] * n,
            "kelly_fraction_final": [0.18] * n,
        })

    def test_hedge_overrides_dec_to_inc(self):
        """Triggered hedge switches direction from DEC to INC."""
        h = CurtailmentHedge(trigger_prob=0.10)
        hedge_df = h.compute_hedge(
            _make_forecast(n=6, p_neg=0.80),
            100.0,
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        merged = h.merge_with_kelly(self._kelly_alloc(6), hedge_df)
        # Triggered hours should show INC in final_direction
        triggered = merged.filter(pl.col("hedge_triggered"))
        assert (triggered["final_direction"] == "INC").all()

    def test_final_mw_is_max_of_hedge_and_kelly(self):
        h = CurtailmentHedge(trigger_prob=0.10, max_position_mw=50.0)
        hedge_df = h.compute_hedge(
            _make_forecast(n=1, p_neg=0.80),
            100.0,  # hedge_mw ≈ 50.0
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        kelly = pl.DataFrame({
            "interval_start_utc": [datetime(2025, 6, 1, tzinfo=UTC)],
            "direction": ["DEC"],
            "position_mw": [10.0],  # Kelly < hedge
            "kelly_fraction_raw": [0.10],
            "kelly_fraction_damped": [0.10],
            "kelly_fraction_final": [0.09],
        })
        merged = h.merge_with_kelly(kelly, hedge_df)
        # final_position_mw should be max(hedge_mw, kelly_mw)
        assert merged["final_position_mw"][0] >= 10.0
