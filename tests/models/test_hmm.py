"""Tests for DARTRegimeModel (M3).

hmmlearn is mocked — no real training required in CI.
Walk-forward gate and output schema are the highest-priority tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.ingest.exceptions import MissingDataError, WalkForwardViolation
from src.models.hmm import (
    REGIME_LABELS,
    REGIME_NEGATIVE_CONGESTION,
    REGIME_NORMAL,
    REGIME_SCARCITY,
    DARTRegimeModel,
)

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_features(n: int = 50, start_hour: int = 0) -> pl.DataFrame:
    """Minimal feature matrix with required HMM input columns."""
    from datetime import timedelta
    base = datetime(2025, 1, 1, tzinfo=UTC)
    hours = [base + timedelta(hours=start_hour + h) for h in range(n)]
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "interval_start_utc": hours,
        "dart_spread_usd": rng.normal(0, 5, n).tolist(),
        "thermal_share": rng.uniform(0.4, 0.8, n).tolist(),
        "ercot_load_mw": rng.uniform(30000, 60000, n).tolist(),
        "data_tag": ["REAL"] * n,
    })


def _mock_hmm(n_states: int = 3, n_rows: int = 50):
    """Return a mock GaussianHMM that dynamically sizes output to input length."""
    mock = MagicMock()
    mock.monitor_ = MagicMock(converged=True)

    def _predict(X, *a, **kw):
        n = len(X)
        return np.tile(np.arange(n_states), n // n_states + 1)[:n]

    def _predict_proba(X, *a, **kw):
        n = len(X)
        return np.ones((n, n_states)) / n_states

    mock.predict.side_effect = _predict
    mock.predict_proba.side_effect = _predict_proba
    return mock


# ── Initialization ────────────────────────────────────────────────────────────

class TestInit:
    def test_default_params(self):
        m = DARTRegimeModel()
        assert m.n_states == 3
        assert m.n_iter == 100
        assert m.covariance_type == "full"

    def test_custom_params(self):
        m = DARTRegimeModel(n_states=2, n_iter=50)
        assert m.n_states == 2

    def test_not_fitted_initially(self):
        m = DARTRegimeModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_state_probs(_make_features(), datetime(2025, 1, 2, tzinfo=UTC))


# ── Walk-forward gate ─────────────────────────────────────────────────────────

class TestWalkForwardGate:
    def test_fit_raises_on_naive_datetime(self):
        m = DARTRegimeModel()
        with pytest.raises(WalkForwardViolation):
            with patch("src.models.hmm.GaussianHMM", return_value=_mock_hmm()):
                m.fit(_make_features(), datetime(2025, 1, 3))  # naive

    def test_predict_raises_on_naive_datetime(self):
        m = DARTRegimeModel()
        features = _make_features()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        with patch("src.models.hmm.GaussianHMM", return_value=_mock_hmm()):
            m.fit(features, gate)
        with pytest.raises(WalkForwardViolation):
            m.predict_state_probs(features, datetime(2025, 1, 3))

    def test_fit_gates_future_rows(self):
        """Rows after as_of should be excluded from training."""
        features = _make_features(n=50)  # sequential hours 0..49
        # Gate at hour 10: only first 11 rows (hour 0–10 inclusive)
        from datetime import timedelta
        gate = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=10)
        captured_X: list = []

        mock_instance = _mock_hmm(n_rows=11)
        original_fit = mock_instance.fit

        def capture_fit(X, *a, **kw):
            captured_X.append(X)
            original_fit(X)
            return mock_instance

        mock_instance.fit = capture_fit

        with patch("src.models.hmm.GaussianHMM", return_value=mock_instance):
            m = DARTRegimeModel()
            m.fit(features, gate)

        assert captured_X[0].shape[0] == 11  # hours 0–10 inclusive

    def test_fit_raises_when_all_data_future(self):
        features = _make_features(n=10)
        gate = datetime(2020, 1, 1, tzinfo=UTC)  # way before all data
        m = DARTRegimeModel()
        with pytest.raises(MissingDataError):
            with patch("src.models.hmm.GaussianHMM", return_value=_mock_hmm()):
                m.fit(features, gate)


# ── fit() ─────────────────────────────────────────────────────────────────────

class TestFit:
    def _fitted_model(self, n: int = 50):
        features = _make_features(n)
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        mock = _mock_hmm(n_rows=n)
        with patch("src.models.hmm.GaussianHMM", return_value=mock):
            m = DARTRegimeModel()
            m.fit(features, gate)
        return m, features

    def test_fit_returns_self(self):
        features = _make_features()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        mock = _mock_hmm()
        with patch("src.models.hmm.GaussianHMM", return_value=mock):
            m = DARTRegimeModel()
            result = m.fit(features, gate)
        assert result is m

    def test_scaler_params_set(self):
        m, _ = self._fitted_model()
        assert m._feat_mean is not None
        assert m._feat_std is not None
        assert len(m._feat_mean) == 3  # 3 FEATURE_COLS

    def test_state_perm_set(self):
        m, _ = self._fitted_model()
        assert m._state_perm is not None
        # Permutation must be a valid mapping of {0,1,2}
        assert set(m._state_perm.tolist()) == {0, 1, 2}


# ── predict_state_probs() ─────────────────────────────────────────────────────

class TestPredictStateProbs:
    def _fitted_model(self, n: int = 50):
        features = _make_features(n)
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        mock = _mock_hmm(n_rows=n)
        with patch("src.models.hmm.GaussianHMM", return_value=mock):
            m = DARTRegimeModel()
            m.fit(features, gate)
        return m, features

    def test_output_schema(self):
        m, features = self._fitted_model()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        result = m.predict_state_probs(features, gate)
        assert set(result.columns) == {
            "interval_start_utc", "p_normal", "p_scarcity",
            "p_negative_congestion", "regime",
        }

    def test_probabilities_sum_to_one(self):
        m, features = self._fitted_model()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        result = m.predict_state_probs(features, gate)
        total = (
            result["p_normal"] + result["p_scarcity"] + result["p_negative_congestion"]
        )
        assert (total - 1.0).abs().max() < 1e-6

    def test_regime_column_is_valid_label(self):
        m, features = self._fitted_model()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        result = m.predict_state_probs(features, gate)
        valid = set(REGIME_LABELS.values())
        assert set(result["regime"].unique().to_list()).issubset(valid)

    def test_missing_feature_col_raises(self):
        m, features = self._fitted_model()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        bad = features.drop("thermal_share")
        with pytest.raises(MissingDataError):
            m.predict_state_probs(bad, gate)

    def test_empty_after_gate_raises(self):
        m, features = self._fitted_model()
        early_gate = datetime(2020, 1, 1, tzinfo=UTC)
        with pytest.raises(MissingDataError):
            m.predict_state_probs(features, early_gate)


# ── decode_states() ───────────────────────────────────────────────────────────

class TestDecodeStates:
    def test_output_schema(self):
        features = _make_features()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        mock = _mock_hmm()
        with patch("src.models.hmm.GaussianHMM", return_value=mock):
            m = DARTRegimeModel()
            m.fit(features, gate)
        result = m.decode_states(features, gate)
        assert set(result.columns) == {"interval_start_utc", "regime"}

    def test_regime_values_valid(self):
        features = _make_features()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        mock = _mock_hmm()
        with patch("src.models.hmm.GaussianHMM", return_value=mock):
            m = DARTRegimeModel()
            m.fit(features, gate)
        result = m.decode_states(features, gate)
        valid = set(REGIME_LABELS.values())
        assert set(result["regime"].unique().to_list()).issubset(valid)


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        features = _make_features()
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        mock = _mock_hmm()
        with patch("src.models.hmm.GaussianHMM", return_value=mock):
            m = DARTRegimeModel()
            m.fit(features, gate)

        # Replace MagicMock (not picklable) with a sentinel so save/load can be tested
        m._model = None

        path = tmp_path / "hmm.pkl"
        m.save(path)
        m2 = DARTRegimeModel.load(path)
        assert m2._feat_mean is not None
        assert m2.n_states == m.n_states
        assert (m2._feat_mean == m._feat_mean).all()
