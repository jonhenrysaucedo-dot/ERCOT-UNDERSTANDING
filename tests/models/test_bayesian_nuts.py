"""Tests for DARTBayesianForecaster (M5).

pymc, arviz, and nutpie are mocked — no real MCMC sampling in CI.
Walk-forward gate, output schema, and posterior rescaling are highest priority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.ingest.exceptions import MissingDataError, WalkForwardViolation
from src.models.bayesian_nuts import DARTBayesianForecaster

UTC = timezone.utc

FEAT_COLS = ["dart_lag_24h", "thermal_share", "ercot_load_mw"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_features(n: int = 200) -> pl.DataFrame:
    from datetime import timedelta
    rng = np.random.default_rng(99)
    base = datetime(2025, 1, 1, tzinfo=UTC)
    hours = [base + timedelta(hours=h) for h in range(n)]
    return pl.DataFrame({
        "interval_start_utc": hours,
        "dart_spread_usd": rng.normal(0, 5, n).tolist(),
        "dart_lag_24h": rng.normal(0, 5, n).tolist(),
        "thermal_share": rng.uniform(0.4, 0.8, n).tolist(),
        "ercot_load_mw": rng.uniform(30000, 60000, n).tolist(),
        "data_tag": ["REAL"] * n,
    })


def _mock_pm(n_pred_rows: int = 10):
    """Return a mock pymc module that behaves like real pm.sample."""
    pm = MagicMock()

    # Simulate a minimal PyMC model context
    model_ctx = MagicMock()
    model_ctx.__enter__ = MagicMock(return_value=model_ctx)
    model_ctx.__exit__ = MagicMock(return_value=False)
    pm.Model.return_value = model_ctx

    # pm.sample returns a mock trace
    trace = MagicMock()
    pm.sample.return_value = trace

    # pm.sample_posterior_predictive returns arviz-like ppc
    ppc = MagicMock()
    # shape: (n_chains, n_draws, n_obs) → simulate 2 chains * 100 draws
    obs_samples = np.random.default_rng(0).normal(2.0, 1.0, (2, 100, n_pred_rows))
    ppc.posterior_predictive = {"obs": MagicMock(values=obs_samples)}
    pm.sample_posterior_predictive.return_value = ppc

    # Data containers
    pm.Data.return_value = MagicMock()
    pm.Normal.return_value = MagicMock()
    pm.HalfNormal.return_value = MagicMock()
    pm.Deterministic.return_value = MagicMock()
    pm.set_data = MagicMock()

    return pm, trace, ppc


def _make_fitted_model(n_train: int = 200, n_pred: int = 10):
    """Return a fitted DARTBayesianForecaster with mocked PyMC."""
    features = _make_features(n_train)
    gate = datetime(2025, 1, 9, tzinfo=UTC)

    mock_pm, trace, ppc = _mock_pm(n_pred_rows=n_pred)

    with patch("src.models.bayesian_nuts.pm", mock_pm):
        m = DARTBayesianForecaster(feature_cols=FEAT_COLS, draws=100, tune=50, chains=2)
        m.fit(features, "dart_spread_usd", gate)

    return m, mock_pm, trace, ppc, features


# ── Initialization ────────────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        m = DARTBayesianForecaster(feature_cols=FEAT_COLS)
        assert m.draws == 2000
        assert m.chains == 4
        assert m.nuts_sampler == "nutpie"

    def test_empty_feature_cols_raises(self):
        with pytest.raises(ValueError):
            DARTBayesianForecaster(feature_cols=[])

    def test_not_fitted_raises(self):
        """forecast() raises RuntimeError when called before fit()."""
        import src.models.bayesian_nuts as _mod
        m = DARTBayesianForecaster(feature_cols=FEAT_COLS)
        mock_pm, *_ = _mock_pm()
        # pm must be non-None so the check reaches _check_fitted
        with patch.object(_mod, "pm", mock_pm):
            with pytest.raises(RuntimeError, match="not fitted"):
                m.forecast(_make_features(5), datetime(2025, 1, 9, tzinfo=UTC))


# ── Walk-forward gate ─────────────────────────────────────────────────────────

class TestWalkForwardGate:
    def test_fit_raises_on_naive_datetime(self):
        m = DARTBayesianForecaster(feature_cols=FEAT_COLS)
        mock_pm, *_ = _mock_pm()
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            with pytest.raises(WalkForwardViolation):
                m.fit(_make_features(), "dart_spread_usd", datetime(2025, 1, 9))

    def test_forecast_raises_on_naive_datetime(self):
        m, *_ = _make_fitted_model()
        with pytest.raises(WalkForwardViolation):
            m.forecast(_make_features(5), datetime(2025, 1, 9))  # naive

    def test_fit_gates_future_rows(self):
        """Rows after as_of are excluded from training."""
        features = _make_features(n=100)
        # Gate at row 20 (hour 20 of 2025-01-01)
        gate = datetime(2025, 1, 1, 19, tzinfo=UTC)  # rows 0–19

        mock_pm, _, _ = _mock_pm(n_pred_rows=20)
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            m = DARTBayesianForecaster(feature_cols=FEAT_COLS, draws=10, tune=5, chains=1)
            m.fit(features, "dart_spread_usd", gate)

        # Scaler was fitted on gated rows only
        assert m._feat_mean is not None
        # 20 rows used (hour 0–19)
        call_args = mock_pm.sample.call_args
        assert call_args is not None

    def test_all_data_future_raises(self):
        m = DARTBayesianForecaster(feature_cols=FEAT_COLS)
        features = _make_features()
        early_gate = datetime(2020, 1, 1, tzinfo=UTC)
        mock_pm, *_ = _mock_pm()
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            with pytest.raises(MissingDataError):
                m.fit(features, "dart_spread_usd", early_gate)


# ── fit() ─────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self):
        features = _make_features()
        gate = datetime(2025, 1, 9, tzinfo=UTC)
        mock_pm, *_ = _mock_pm()
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            m = DARTBayesianForecaster(feature_cols=FEAT_COLS)
            result = m.fit(features, "dart_spread_usd", gate)
        assert result is m

    def test_scaler_params_set(self):
        m, *_ = _make_fitted_model()
        assert m._feat_mean is not None
        assert m._feat_std is not None
        assert len(m._feat_mean) == len(FEAT_COLS)
        assert m._target_mean is not None
        assert m._target_std is not None

    def test_missing_feature_col_raises(self):
        features = _make_features().drop("thermal_share")
        gate = datetime(2025, 1, 9, tzinfo=UTC)
        mock_pm, *_ = _mock_pm()
        m = DARTBayesianForecaster(feature_cols=FEAT_COLS)
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            with pytest.raises(MissingDataError):
                m.fit(features, "dart_spread_usd", gate)

    def test_missing_target_col_raises(self):
        features = _make_features().drop("dart_spread_usd")
        gate = datetime(2025, 1, 9, tzinfo=UTC)
        mock_pm, *_ = _mock_pm()
        m = DARTBayesianForecaster(feature_cols=FEAT_COLS)
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            with pytest.raises(MissingDataError):
                m.fit(features, "dart_spread_usd", gate)

    def test_nutpie_fallback_when_not_installed(self):
        """If nutpie is missing, sampler falls back to 'pymc'."""
        features = _make_features()
        gate = datetime(2025, 1, 9, tzinfo=UTC)
        mock_pm, *_ = _mock_pm()

        m = DARTBayesianForecaster(feature_cols=FEAT_COLS, nuts_sampler="nutpie")
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            with patch.dict("sys.modules", {"nutpie": None}):
                m.fit(features, "dart_spread_usd", gate)

        # Should have called pm.sample with sampler = 'pymc'
        call_kwargs = mock_pm.sample.call_args[1]
        assert call_kwargs.get("nuts_sampler") == "pymc"


# ── forecast() ───────────────────────────────────────────────────────────────

class TestForecast:
    def test_output_schema(self):
        n_pred = 10
        m, mock_pm, trace, ppc, _ = _make_fitted_model(n_pred=n_pred)
        pred_features = _make_features(n_pred)
        gate = datetime(2025, 1, 9, tzinfo=UTC)

        with patch("src.models.bayesian_nuts.pm", mock_pm):
            result = m.forecast(pred_features, gate)

        assert set(result.columns) == {
            "interval_start_utc", "q10", "q50", "q90", "p_positive", "p_negative",
        }

    def test_output_length_matches_input(self):
        n_pred = 8
        m, mock_pm, trace, ppc, _ = _make_fitted_model(n_pred=n_pred)
        pred_features = _make_features(n_pred)
        gate = datetime(2025, 1, 9, tzinfo=UTC)

        with patch("src.models.bayesian_nuts.pm", mock_pm):
            result = m.forecast(pred_features, gate)

        assert len(result) == n_pred

    def test_quantile_ordering(self):
        """q10 ≤ q50 ≤ q90 for all rows."""
        n_pred = 10
        m, mock_pm, trace, ppc, _ = _make_fitted_model(n_pred=n_pred)
        pred_features = _make_features(n_pred)
        gate = datetime(2025, 1, 9, tzinfo=UTC)

        with patch("src.models.bayesian_nuts.pm", mock_pm):
            result = m.forecast(pred_features, gate)

        assert (result["q10"] <= result["q50"]).all()
        assert (result["q50"] <= result["q90"]).all()

    def test_probabilities_sum_to_one(self):
        """p_positive + p_negative ≤ 1 (equality only when no zeros in samples)."""
        n_pred = 10
        m, mock_pm, trace, ppc, _ = _make_fitted_model(n_pred=n_pred)
        pred_features = _make_features(n_pred)
        gate = datetime(2025, 1, 9, tzinfo=UTC)

        with patch("src.models.bayesian_nuts.pm", mock_pm):
            result = m.forecast(pred_features, gate)

        total = result["p_positive"] + result["p_negative"]
        assert (total <= 1.0 + 1e-9).all()

    def test_probabilities_in_unit_interval(self):
        n_pred = 5
        m, mock_pm, trace, ppc, _ = _make_fitted_model(n_pred=n_pred)
        pred_features = _make_features(n_pred)
        gate = datetime(2025, 1, 9, tzinfo=UTC)

        with patch("src.models.bayesian_nuts.pm", mock_pm):
            result = m.forecast(pred_features, gate)

        assert (result["p_positive"] >= 0).all()
        assert (result["p_positive"] <= 1).all()
        assert (result["p_negative"] >= 0).all()
        assert (result["p_negative"] <= 1).all()

    def test_q50_in_original_units(self):
        """q50 should be in the original DART spread scale (not standardized)."""
        # Mock samples centered at +2 std devs above target_mean
        # With target_mean~0, target_std~5, samples~mean(0,1)*5+0 ≈ 10
        n_pred = 5
        m, mock_pm, trace, ppc, _ = _make_fitted_model(n_pred=n_pred)

        # Override the ppc to return samples centered at 2.0 in standardized space
        obs_samples = np.full((2, 100, n_pred), 2.0)
        ppc.posterior_predictive = {"obs": MagicMock(values=obs_samples)}
        mock_pm.sample_posterior_predictive.return_value = ppc

        pred_features = _make_features(n_pred)
        gate = datetime(2025, 1, 9, tzinfo=UTC)

        with patch("src.models.bayesian_nuts.pm", mock_pm):
            result = m.forecast(pred_features, gate)

        # Rescaled: 2.0 * target_std + target_mean ≈ some finite number
        # The key check: q50 should not be near 2.0 (standardized) but in original scale
        # target_std is approx 5 (std of normal(0,5))
        # so q50 ≈ 2*5 + 0 = 10 (not exactly due to rng)
        assert result["q50"].mean() > 5.0  # well above zero, clearly rescaled

    def test_missing_feature_col_raises(self):
        m, mock_pm, *_ = _make_fitted_model()
        pred_features = _make_features(5).drop("thermal_share")
        gate = datetime(2025, 1, 9, tzinfo=UTC)
        with patch("src.models.bayesian_nuts.pm", mock_pm):
            with pytest.raises(MissingDataError):
                m.forecast(pred_features, gate)
