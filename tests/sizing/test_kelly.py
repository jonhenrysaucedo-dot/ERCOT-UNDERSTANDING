"""Tests for KellySizer (M7).

scipy.optimize.minimize_scalar is mocked for the Kelly fraction tests.
The KellySizer orchestration logic (damping, haircut, MW cap) is tested with
a mock compute_kelly_fraction to keep tests fast.

Half-Kelly non-negotiability and covariance haircut are the highest-priority tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from src.ingest.exceptions import WalkForwardViolation
from src.sizing.kelly import (
    KellySizer,
    _kelly_objective,
    compute_kelly_fraction,
)

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_forecast(n: int = 6) -> pl.DataFrame:
    base = datetime(2025, 6, 1, tzinfo=UTC)
    return pl.DataFrame({
        "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
        "q10": [-3.0] * n,
        "q50": [4.0] * n,
        "q90": [9.0] * n,
        "p_positive": [0.75] * n,
        "p_negative": [0.25] * n,
    })


def _make_composite(n: int = 6, eligible: bool = True) -> pl.DataFrame:
    base = datetime(2025, 6, 1, tzinfo=UTC)
    return pl.DataFrame({
        "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
        "direction": ["INC"] * n,
        "directional_conviction": [0.75] * n,
        "composite_score": [0.60] * n,
        "trade_eligible": [eligible] * n,
    })


def _make_samples(n_draws: int = 200, n_hours: int = 6, mean: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(mean, 1.0, (n_draws, n_hours))


def _default_sizer() -> KellySizer:
    return KellySizer(
        half_kelly_multiplier=0.50,
        max_position_mw=50.0,
        uncertainty_damp_threshold=1.0,
    )


# ── Kelly objective function ──────────────────────────────────────────────────

class TestKellyObjective:
    def test_positive_returns_finite(self):
        r = np.array([1.0, 2.0, 3.0])
        val = _kelly_objective(0.1, r)
        assert np.isfinite(val)
        assert val < 0  # negative expected log wealth (to be minimized)

    def test_zero_fraction_is_zero_growth(self):
        """f=0 → log(1 + 0) = 0 for all returns → objective = 0."""
        r = np.array([1.0, -0.5, 2.0])
        val = _kelly_objective(0.0, r)
        assert abs(val) < 1e-9

    def test_clips_near_ruin(self):
        """Very large f with large negative returns should not raise."""
        r = np.array([-0.99, -0.99, -0.99])
        val = _kelly_objective(1.0, r)
        assert np.isfinite(val)


# ── compute_kelly_fraction ────────────────────────────────────────────────────

class TestComputeKellyFraction:
    def _mock_minimize(self, optimal_f: float = 0.3):
        mock_result = type("R", (), {"x": optimal_f})()

        def fake_minimize(fn, bounds, method, args, options):
            return mock_result

        return fake_minimize

    def test_returns_value_in_range(self):
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize(0.3)):
            f = compute_kelly_fraction(np.array([1.0, 2.0, -0.5]), max_f=0.5)
        assert 0.0 <= f <= 0.5

    def test_empty_returns_zero(self):
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize()):
            f = compute_kelly_fraction(np.array([]), max_f=0.5)
        assert f == 0.0

    def test_all_zero_returns_zero(self):
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize()):
            f = compute_kelly_fraction(np.zeros(10), max_f=0.5)
        assert f == 0.0

    def test_clipped_to_max_f(self):
        """Even if optimizer returns > max_f, result is clipped."""
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize(0.99)):
            f = compute_kelly_fraction(np.array([1.0]), max_f=0.5)
        assert f <= 0.5

    def test_scipy_missing_raises(self):
        with patch("src.sizing.kelly._minimize_scalar", None):
            with pytest.raises(ImportError):
                compute_kelly_fraction(np.array([1.0, 2.0]))


# ── KellySizer initialization ─────────────────────────────────────────────────

class TestKellySizerInit:
    def test_half_kelly_enforced(self):
        """Any multiplier != 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="non-negotiable"):
            KellySizer(half_kelly_multiplier=0.4)

    def test_half_kelly_accepted(self):
        s = KellySizer(half_kelly_multiplier=0.50)
        assert s.half_kelly_multiplier == 0.50

    def test_max_position_stored(self):
        s = KellySizer(half_kelly_multiplier=0.50, max_position_mw=30.0)
        assert s.max_position_mw == 30.0


# ── Walk-forward gate ─────────────────────────────────────────────────────────

class TestWalkForwardGate:
    def test_naive_datetime_raises(self):
        s = _default_sizer()
        with pytest.raises(WalkForwardViolation):
            s.size_positions(
                _make_forecast(), _make_composite(), _make_samples(),
                datetime(2025, 6, 2),  # naive
            )

    def test_utc_datetime_accepted(self):
        mock_result = type("R", (), {"x": 0.2})()

        def fake_minimize(*a, **kw):
            return mock_result

        with patch("src.sizing.kelly._minimize_scalar", fake_minimize):
            s = _default_sizer()
            result = s.size_positions(
                _make_forecast(), _make_composite(), _make_samples(),
                datetime(2025, 6, 2, tzinfo=UTC),
            )
        assert len(result) == 6


# ── size_positions output ─────────────────────────────────────────────────────

class TestSizePositions:
    def _mock_minimize(self, f: float = 0.25):
        mock_result = type("R", (), {"x": f})()

        def fake_min(*a, **kw):
            return mock_result

        return fake_min

    def test_output_schema(self):
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize()):
            result = _default_sizer().size_positions(
                _make_forecast(), _make_composite(), _make_samples(),
                datetime(2025, 6, 2, tzinfo=UTC),
            )
        expected_cols = {
            "interval_start_utc", "direction", "kelly_fraction_raw",
            "kelly_fraction_damped", "kelly_fraction_final",
            "position_mw", "uncertainty_ratio", "max_pairwise_corr",
            "covariance_haircut",
        }
        assert set(result.columns) == expected_cols

    def test_position_capped_at_max(self):
        """position_mw must never exceed max_position_mw."""
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize(0.5)):
            s = KellySizer(half_kelly_multiplier=0.50, max_position_mw=30.0)
            result = s.size_positions(
                _make_forecast(), _make_composite(), _make_samples(),
                datetime(2025, 6, 2, tzinfo=UTC),
            )
        assert (result["position_mw"] <= 30.0 + 1e-6).all()

    def test_ineligible_hours_zero_position(self):
        """Ineligible hours get zero position."""
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize(0.3)):
            result = _default_sizer().size_positions(
                _make_forecast(), _make_composite(eligible=False), _make_samples(),
                datetime(2025, 6, 2, tzinfo=UTC),
            )
        assert (result["position_mw"] == 0.0).all()

    def test_uncertainty_damping_reduces_fraction(self):
        """High CI width → kelly_damped < kelly_raw."""
        # Make samples with very wide CI: q10=-100, q50=1, q90=100
        base = datetime(2025, 6, 1, tzinfo=UTC)
        f_df = pl.DataFrame({
            "interval_start_utc": [base],
            "q10": [-100.0],
            "q50": [1.0],
            "q90": [100.0],  # CI width = 200, |q50| = 1 → ratio = 200 >> threshold 1.0
            "p_positive": [0.75],
            "p_negative": [0.25],
        })
        c_df = pl.DataFrame({
            "interval_start_utc": [base],
            "direction": ["INC"],
            "directional_conviction": [0.75],
            "composite_score": [0.60],
            "trade_eligible": [True],
        })
        samples = np.array([[1.0]] * 100)

        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize(0.3)):
            result = _default_sizer().size_positions(
                f_df, c_df, samples, datetime(2025, 6, 2, tzinfo=UTC)
            )
        assert result["kelly_fraction_damped"][0] < result["kelly_fraction_raw"][0]

    def test_position_non_negative(self):
        with patch("src.sizing.kelly._minimize_scalar", self._mock_minimize(0.2)):
            result = _default_sizer().size_positions(
                _make_forecast(), _make_composite(), _make_samples(),
                datetime(2025, 6, 2, tzinfo=UTC),
            )
        assert (result["position_mw"] >= 0.0).all()


# ── Covariance haircut ────────────────────────────────────────────────────────

class TestCovarianceHaircut:
    def test_perfectly_correlated_hours_max_haircut(self):
        """Samples identical across hours → max correlation → haircut near 0."""
        s = _default_sizer()
        # All hours same signal → correlation = 1.0 → haircut = 0.0
        samples = np.tile(np.arange(100), (6, 1)).T  # shape (100, 6)
        haircut = s._compute_covariance_haircut(samples)
        assert haircut < 0.1

    def test_independent_hours_no_haircut(self):
        """Truly independent samples → low correlation → haircut near 1.0."""
        rng = np.random.default_rng(0)
        samples = rng.normal(0, 1, (10000, 6))
        s = _default_sizer()
        haircut = s._compute_covariance_haircut(samples)
        assert haircut > 0.7  # should be close to 1.0

    def test_single_hour_no_haircut(self):
        samples = np.ones((100, 1)) * 2.0
        s = _default_sizer()
        haircut = s._compute_covariance_haircut(samples)
        assert haircut == 1.0

    def test_haircut_in_unit_interval(self):
        rng = np.random.default_rng(99)
        samples = rng.normal(0, 1, (50, 4))
        s = _default_sizer()
        haircut = s._compute_covariance_haircut(samples)
        assert 0.0 <= haircut <= 1.0


# ── Input validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    def test_wrong_samples_shape_raises(self):
        s = _default_sizer()
        bad_samples = np.ones((6,))  # 1-D, not 2-D
        with pytest.raises(ValueError, match="2-D"):
            s.size_positions(
                _make_forecast(), _make_composite(), bad_samples,
                datetime(2025, 6, 2, tzinfo=UTC),
            )

    def test_samples_columns_mismatch_raises(self):
        s = _default_sizer()
        bad_samples = np.ones((100, 3))  # 3 hours but forecast has 6
        with pytest.raises(ValueError, match="shape\\[1\\]"):
            s.size_positions(
                _make_forecast(6), _make_composite(6), bad_samples,
                datetime(2025, 6, 2, tzinfo=UTC),
            )
