"""Tests for DARTVolatilityModel (M4).

arch package is mocked — no real GARCH fitting in CI.
Walk-forward gate, output schema, and regime scaling are highest priority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.ingest.exceptions import MissingDataError, WalkForwardViolation
from src.models.garch import DARTVolatilityModel

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_dart(n: int = 100) -> pl.DataFrame:
    from datetime import timedelta
    base = datetime(2025, 1, 1, tzinfo=UTC)
    hours = [base + timedelta(hours=h) for h in range(n)]
    rng = np.random.default_rng(7)
    return pl.DataFrame({
        "interval_start_utc": hours,
        "dart_spread_usd": rng.normal(0, 3, n).tolist(),
        "data_tag": ["REAL"] * n,
    })


def _mock_arch_result(horizon: int = 24) -> MagicMock:
    result = MagicMock()
    result.aic = -500.0
    result.bic = -490.0
    result.resid = np.random.default_rng(0).normal(0, 1, 100)
    result.conditional_volatility = np.abs(result.resid) + 0.1
    # forecast().variance.values shape (1, horizon)
    forecast_obj = MagicMock()
    forecast_obj.variance.values = np.full((1, horizon), 4.0)
    result.forecast.return_value = forecast_obj
    return result


def _mock_arch_model(horizon: int = 24):
    mock_am = MagicMock()
    mock_am.fit.return_value = _mock_arch_result(horizon)
    return mock_am


# ── Walk-forward gate ─────────────────────────────────────────────────────────

class TestWalkForwardGate:
    def test_fit_raises_on_naive_datetime(self):
        m = DARTVolatilityModel()
        dart = _make_dart()
        with pytest.raises(WalkForwardViolation):
            with patch("src.models.garch.arch_model", return_value=_mock_arch_model()):
                m.fit(dart, datetime(2025, 1, 5))  # naive

    def test_fit_gates_future_rows(self):
        """Only rows up to as_of should be used in fitting."""
        from datetime import timedelta
        dart = _make_dart(n=100)  # sequential hours 0..99
        # Gate at hour 9: rows 0–9 inclusive = 10 rows, well above the 10-row minimum
        gate = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=9)
        captured_series: list = []

        def capture(series, **kwargs):
            captured_series.append(series)
            return _mock_arch_model()

        with patch("src.models.garch.arch_model", side_effect=capture):
            m = DARTVolatilityModel()
            m.fit(dart, gate)

        assert len(captured_series[0]) == 10  # rows 0–9 inclusive

    def test_fit_raises_when_too_few_rows(self):
        """Fewer than 10 rows after gate → MissingDataError."""
        from datetime import timedelta
        dart = _make_dart(n=100)
        # Gate before 10th row: only 9 rows (hours 0–8)
        gate = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=8)
        m = DARTVolatilityModel()
        with pytest.raises(MissingDataError, match="Too few"):
            with patch("src.models.garch.arch_model", return_value=_mock_arch_model()):
                m.fit(dart, gate)

    def test_fit_raises_missing_column(self):
        dart = _make_dart().drop("dart_spread_usd")
        m = DARTVolatilityModel()
        with pytest.raises(MissingDataError):
            with patch("src.models.garch.arch_model", return_value=_mock_arch_model()):
                m.fit(dart, datetime(2025, 1, 5, tzinfo=UTC))


# ── fit() ─────────────────────────────────────────────────────────────────────

class TestFit:
    def _fitted(self, n: int = 100, horizon: int = 24):
        dart = _make_dart(n)
        gate = datetime(2025, 1, 5, tzinfo=UTC)
        with patch("src.models.garch.arch_model", return_value=_mock_arch_model(horizon)):
            m = DARTVolatilityModel()
            m.fit(dart, gate)
        return m

    def test_fit_returns_self(self):
        dart = _make_dart()
        gate = datetime(2025, 1, 5, tzinfo=UTC)
        with patch("src.models.garch.arch_model", return_value=_mock_arch_model()):
            m = DARTVolatilityModel()
            result = m.fit(dart, gate)
        assert result is m

    def test_result_stored(self):
        m = self._fitted()
        assert m._result is not None

    def test_last_variance_set(self):
        m = self._fitted()
        assert m._last_variance is not None
        assert m._last_variance > 0


# ── forecast_variance() ───────────────────────────────────────────────────────

class TestForecastVariance:
    def _fitted(self, horizon: int = 24):
        dart = _make_dart()
        gate = datetime(2025, 1, 5, tzinfo=UTC)
        with patch("src.models.garch.arch_model", return_value=_mock_arch_model(horizon)):
            m = DARTVolatilityModel()
            m.fit(dart, gate)
        return m

    def test_output_schema(self):
        m = self._fitted()
        result = m.forecast_variance(horizon=24)
        assert set(result.columns) == {"hour", "sigma2", "sigma2_regime_weighted"}

    def test_output_length_matches_horizon(self):
        m = self._fitted(horizon=24)
        result = m.forecast_variance(horizon=24)
        assert len(result) == 24

    def test_sigma2_positive(self):
        m = self._fitted()
        result = m.forecast_variance(horizon=24)
        assert (result["sigma2"] > 0).all()

    def test_no_regime_probs_equals_base_variance(self):
        """Without regime_probs, sigma2_regime_weighted == sigma2."""
        m = self._fitted()
        result = m.forecast_variance(horizon=24)
        delta = (result["sigma2"] - result["sigma2_regime_weighted"]).abs().max()
        assert delta < 1e-9

    def test_scarcity_regime_inflates_variance(self):
        """When P(SCARCITY)=1, regime_weighted > base variance (scale > 1)."""
        m = self._fitted(horizon=6)
        probs = pl.DataFrame({
            "p_normal": [0.0] * 6,
            "p_scarcity": [1.0] * 6,
            "p_negative_congestion": [0.0] * 6,
        })
        result = m.forecast_variance(horizon=6, regime_probs=probs)
        assert (result["sigma2_regime_weighted"] > result["sigma2"]).all()

    def test_invalid_horizon_raises(self):
        m = self._fitted()
        with pytest.raises(ValueError):
            m.forecast_variance(horizon=0)

    def test_unfitted_raises(self):
        m = DARTVolatilityModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.forecast_variance(horizon=24)


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        dart = _make_dart()
        gate = datetime(2025, 1, 5, tzinfo=UTC)
        with patch("src.models.garch.arch_model", return_value=_mock_arch_model()):
            m = DARTVolatilityModel()
            m.fit(dart, gate)

        # Replace MagicMock (not picklable) with a sentinel
        result_sentinel = {"fitted": True, "aic": -500.0}
        m._result = result_sentinel

        path = tmp_path / "garch.pkl"
        m.save(path)
        m2 = DARTVolatilityModel.load(path)
        assert m2._result is not None
        assert m2.p == m.p
        assert m2.q == m.q
        assert m2._regime_var_scales == m._regime_var_scales
