"""Tests for CompositeScorer (M6).

No heavy dependencies needed — pure numpy/polars arithmetic.
Walk-forward gate, score formula correctness, and eligibility filtering are
highest priority.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from src.ingest.exceptions import WalkForwardViolation
from src.scoring.composite import CompositeScorer

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_forecast(n: int = 6, p_pos: float = 0.70) -> pl.DataFrame:
    """Minimal forecast DataFrame matching M5 output schema."""
    from datetime import timedelta
    base = datetime(2025, 6, 1, tzinfo=UTC)
    return pl.DataFrame({
        "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
        "q10": [-3.0] * n,
        "q50": [4.0] * n,
        "q90": [9.0] * n,
        "p_positive": [p_pos] * n,
        "p_negative": [1.0 - p_pos] * n,
    })


def _default_scorer() -> CompositeScorer:
    return CompositeScorer(w1=0.50, w2=0.30, w3=0.20)


# ── Initialization ────────────────────────────────────────────────────────────

class TestInit:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            CompositeScorer(w1=0.5, w2=0.4, w3=0.2)

    def test_valid_init(self):
        s = CompositeScorer(w1=0.5, w2=0.3, w3=0.2)
        assert s.w1 == 0.5

    def test_defaults(self):
        s = CompositeScorer()
        assert abs(s.w1 + s.w2 + s.w3 - 1.0) < 1e-9


# ── Walk-forward gate ─────────────────────────────────────────────────────────

class TestWalkForwardGate:
    def test_raises_on_naive_datetime(self):
        s = _default_scorer()
        f = _make_forecast()
        with pytest.raises(WalkForwardViolation):
            s.compute_composite(f, datetime(2025, 6, 2))  # naive

    def test_accepts_utc_datetime(self):
        s = _default_scorer()
        f = _make_forecast()
        result = s.compute_composite(f, datetime(2025, 6, 2, tzinfo=UTC))
        assert len(result) == 6


# ── Schema ────────────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_required_columns(self):
        s = _default_scorer()
        result = s.compute_composite(_make_forecast(), datetime(2025, 6, 2, tzinfo=UTC))
        expected = {
            "interval_start_utc", "direction", "directional_conviction",
            "spread_magnitude", "spread_magnitude_norm", "fundamental_alignment",
            "composite_score", "trade_eligible",
        }
        assert set(result.columns) == expected

    def test_empty_forecast_returns_empty(self):
        s = _default_scorer()
        empty = pl.DataFrame(schema={
            "interval_start_utc": pl.Datetime("us", "UTC"),
            "q10": pl.Float64, "q50": pl.Float64, "q90": pl.Float64,
            "p_positive": pl.Float64, "p_negative": pl.Float64,
        })
        result = s.compute_composite(empty, datetime(2025, 6, 2, tzinfo=UTC))
        assert len(result) == 0

    def test_missing_column_raises(self):
        s = _default_scorer()
        bad = _make_forecast().drop("q50")
        with pytest.raises(ValueError, match="missing required columns"):
            s.compute_composite(bad, datetime(2025, 6, 2, tzinfo=UTC))


# ── Direction logic ───────────────────────────────────────────────────────────

class TestDirectionLogic:
    def test_inc_when_p_positive_dominates(self):
        s = _default_scorer()
        f = _make_forecast(n=3, p_pos=0.80)
        result = s.compute_composite(f, datetime(2025, 6, 2, tzinfo=UTC))
        assert (result["direction"] == "INC").all()

    def test_dec_when_p_negative_dominates(self):
        s = _default_scorer()
        f = _make_forecast(n=3, p_pos=0.20)
        result = s.compute_composite(f, datetime(2025, 6, 2, tzinfo=UTC))
        assert (result["direction"] == "DEC").all()

    def test_inc_on_exact_tie(self):
        """P(pos) == P(neg) == 0.5 → INC (>=)."""
        s = _default_scorer()
        f = _make_forecast(n=2, p_pos=0.50)
        result = s.compute_composite(f, datetime(2025, 6, 2, tzinfo=UTC))
        assert (result["direction"] == "INC").all()


# ── Score formula ─────────────────────────────────────────────────────────────

class TestScoreFormula:
    def test_composite_in_zero_one(self):
        s = _default_scorer()
        result = s.compute_composite(
            _make_forecast(), datetime(2025, 6, 2, tzinfo=UTC), sigma_historical=5.0
        )
        assert (result["composite_score"] >= 0).all()
        assert (result["composite_score"] <= 1.01).all()  # slight tolerance for float

    def test_formula_manual(self):
        """Verify w1·conviction + w2·mag_norm + w3·fa = composite."""
        s = CompositeScorer(w1=0.5, w2=0.3, w3=0.2)
        f = _make_forecast(n=1, p_pos=0.80)
        result = s.compute_composite(
            f, datetime(2025, 6, 2, tzinfo=UTC), sigma_historical=5.0, fundamental_alignment=1.0
        )
        conviction = result["directional_conviction"][0]
        mag_norm = result["spread_magnitude_norm"][0]
        fa = result["fundamental_alignment"][0]
        expected = 0.5 * conviction + 0.3 * mag_norm + 0.2 * fa
        assert abs(result["composite_score"][0] - expected) < 1e-9

    def test_sigma_none_normalizes_by_max(self):
        """When sigma_historical=None, magnitude is normalized by max across horizon."""
        s = _default_scorer()
        from datetime import timedelta
        base = datetime(2025, 6, 1, tzinfo=UTC)
        f = pl.DataFrame({
            "interval_start_utc": [base + timedelta(hours=h) for h in range(3)],
            "q10": [-2.0, -3.0, -1.0],
            "q50": [2.0, 6.0, 3.0],  # max |q50| = 6
            "q90": [8.0, 12.0, 7.0],
            "p_positive": [0.7, 0.8, 0.75],
            "p_negative": [0.3, 0.2, 0.25],
        })
        result = s.compute_composite(f, datetime(2025, 6, 2, tzinfo=UTC))
        # Max mag_norm should be ≈1.0
        assert result["spread_magnitude_norm"].max() <= 1.0 + 1e-9

    def test_fundamental_alignment_zero_reduces_score(self):
        """fa=0 should lower composite vs fa=1."""
        s = _default_scorer()
        gate = datetime(2025, 6, 2, tzinfo=UTC)
        r1 = s.compute_composite(_make_forecast(n=1), gate, fundamental_alignment=1.0)
        r0 = s.compute_composite(_make_forecast(n=1), gate, fundamental_alignment=0.0)
        assert r1["composite_score"][0] > r0["composite_score"][0]


# ── Eligibility gates ─────────────────────────────────────────────────────────

class TestEligibilityGates:
    def test_high_conviction_eligible(self):
        s = CompositeScorer(
            w1=0.5, w2=0.3, w3=0.2,
            min_composite_score=0.30,
            min_directional_conviction=0.55,
        )
        f = _make_forecast(n=3, p_pos=0.80)
        result = s.compute_composite(
            f, datetime(2025, 6, 2, tzinfo=UTC), sigma_historical=5.0
        )
        assert result["trade_eligible"].all()

    def test_low_conviction_ineligible(self):
        s = CompositeScorer(
            w1=0.5, w2=0.3, w3=0.2,
            min_directional_conviction=0.55,
        )
        f = _make_forecast(n=3, p_pos=0.51)  # conviction = 0.51, below threshold
        result = s.compute_composite(
            f, datetime(2025, 6, 2, tzinfo=UTC), sigma_historical=5.0
        )
        assert not result["trade_eligible"].any()

    def test_low_composite_ineligible(self):
        """Even with high conviction, composite below threshold → ineligible."""
        s = CompositeScorer(
            w1=0.5, w2=0.3, w3=0.2,
            min_composite_score=0.99,  # near impossible threshold
        )
        f = _make_forecast(n=3, p_pos=0.80)
        result = s.compute_composite(
            f, datetime(2025, 6, 2, tzinfo=UTC), sigma_historical=5.0
        )
        assert not result["trade_eligible"].any()

    def test_directional_conviction_correct(self):
        """For INC direction, conviction should be p_positive."""
        s = _default_scorer()
        f = _make_forecast(n=4, p_pos=0.75)
        result = s.compute_composite(f, datetime(2025, 6, 2, tzinfo=UTC))
        # All INC (p_pos=0.75 > 0.25)
        assert (result["directional_conviction"] - 0.75).abs().max() < 1e-9

    def test_per_hour_fundamental_alignment(self):
        """Per-hour fundamental_alignment Series is respected."""
        import polars as pl
        s = _default_scorer()
        gate = datetime(2025, 6, 2, tzinfo=UTC)
        fa_series = pl.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        f = _make_forecast(n=6, p_pos=0.80)
        result = s.compute_composite(f, gate, fundamental_alignment=fa_series)
        # Odd-indexed rows should have fa=0, lower composite
        assert result["fundamental_alignment"][0] == 1.0
        assert result["fundamental_alignment"][1] == 0.0
        assert result["composite_score"][0] > result["composite_score"][1]
