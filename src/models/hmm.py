"""Regime detection model — M3.

3-state GaussianHMM classifies each hour into:
  NORMAL              — typical market conditions
  SCARCITY            — supply tightness, elevated DART spread
  NEGATIVE_CONGESTION — RT prices deeply negative (solar surplus / congestion)

Walk-forward safety:
    fit() gates feature_matrix to as_of_timestamp before fitting.
    predict_state_probs() requires a timezone-aware as_of_timestamp and gates
    its input. Features passed in must already be walk-forward compliant from
    the feature engineering step, but the gate is a backstop.

Usage:
    model = DARTRegimeModel()
    model.fit(feature_matrix, as_of_timestamp=datetime(2025, 1, 1, tzinfo=UTC))
    probs = model.predict_state_probs(feature_matrix, as_of_timestamp=...)
    model.save(Path("output/hmm.pkl"))
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import structlog

from src.ingest.exceptions import MissingDataError, WalkForwardViolation

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None  # type: ignore[assignment,misc]

logger = structlog.get_logger(__name__)
UTC = timezone.utc

# Semantic regime indices (assigned post-fit based on mean DART spread)
REGIME_NORMAL = 0
REGIME_SCARCITY = 1
REGIME_NEGATIVE_CONGESTION = 2

REGIME_LABELS = {
    REGIME_NORMAL: "NORMAL",
    REGIME_SCARCITY: "SCARCITY",
    REGIME_NEGATIVE_CONGESTION: "NEGATIVE_CONGESTION",
}

# Features the HMM trains on (must all be present in feature_matrix)
FEATURE_COLS = ["dart_spread_usd", "thermal_share", "ercot_load_mw"]


def _gate(df: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    if as_of.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware; got naive {as_of!r}"
        )
    return df.filter(pl.col("interval_start_utc") <= as_of)


class DARTRegimeModel:
    """GaussianHMM-based regime classifier for the DART spread.

    After fitting, internal HMM state indices are remapped to semantic labels
    (NORMAL=0, SCARCITY=1, NEGATIVE_CONGESTION=2) based on each state's mean
    dart_spread_usd emission — lowest mean → NEGATIVE_CONGESTION, highest →
    SCARCITY.

    Walk-forward safety:
        fit() and predict_state_probs() gate their input to as_of_timestamp.
        Raises WalkForwardViolation if as_of_timestamp is naive.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model = None
        # Z-score scaler params (fitted from training data)
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None
        # Permutation: hmm_state_idx → semantic_regime_idx
        self._state_perm: Optional[np.ndarray] = None
        self.feature_cols: list[str] = FEATURE_COLS

    # ── public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        feature_matrix: pl.DataFrame,
        as_of_timestamp: datetime,
        feature_cols: Optional[list[str]] = None,
    ) -> "DARTRegimeModel":
        """Fit the GaussianHMM on walk-forward gated features.

        Walk-forward safety:
            feature_matrix is gated to as_of_timestamp before fitting.

        Args:
            feature_matrix: Output of build_feature_matrix() or compatible.
            as_of_timestamp: Walk-forward gate. Must be timezone-aware.
            feature_cols: Override FEATURE_COLS (for unit tests / ablations).

        Returns:
            self, for chaining.
        """
        if GaussianHMM is None:
            raise ImportError("hmmlearn is required: pip install hmmlearn")

        if feature_cols is not None:
            self.feature_cols = feature_cols

        gated = _gate(feature_matrix, as_of_timestamp)
        if len(gated) == 0:
            raise MissingDataError("No data available at or before as_of_timestamp")

        X = self._extract_features(gated)
        X_scaled = self._fit_scaler(X)

        log = logger.bind(n_rows=len(X_scaled), n_states=self.n_states)
        log.info("hmm_fit_start")

        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(X_scaled)

        self._assign_regime_labels(X_scaled)

        log.info("hmm_fit_complete", converged=self._model.monitor_.converged)
        return self

    def predict_state_probs(
        self,
        feature_matrix: pl.DataFrame,
        as_of_timestamp: datetime,
    ) -> pl.DataFrame:
        """Compute posterior state probability vector per hour.

        Walk-forward safety:
            feature_matrix is gated to as_of_timestamp before prediction.
            The model must have been fitted on data strictly before
            as_of_timestamp (enforced by fit()).

        Returns:
            Polars DataFrame with columns:
                interval_start_utc, p_normal, p_scarcity,
                p_negative_congestion, regime (most-likely semantic label)
        """
        self._check_fitted()
        gated = _gate(feature_matrix, as_of_timestamp)
        if len(gated) == 0:
            raise MissingDataError("No data at or before as_of_timestamp")

        X = self._extract_features(gated)
        X_scaled = self._transform_features(X)

        raw_probs = self._model.predict_proba(X_scaled)  # shape (n, n_states)
        reordered = raw_probs[:, np.argsort(self._state_perm)]

        df = gated.select("interval_start_utc").with_columns([
            pl.Series("p_normal", reordered[:, REGIME_NORMAL].tolist()),
            pl.Series("p_scarcity", reordered[:, REGIME_SCARCITY].tolist()),
            pl.Series("p_negative_congestion", reordered[:, REGIME_NEGATIVE_CONGESTION].tolist()),
        ])

        regime_idx = np.argmax(reordered, axis=1)
        regime_names = [REGIME_LABELS[i] for i in regime_idx]
        return df.with_columns(pl.Series("regime", regime_names))

    def decode_states(
        self,
        feature_matrix: pl.DataFrame,
        as_of_timestamp: datetime,
    ) -> pl.DataFrame:
        """Viterbi-decoded most-likely state sequence.

        Walk-forward safety:
            Same as predict_state_probs.

        Returns:
            Polars DataFrame with columns: [interval_start_utc, regime]
        """
        self._check_fitted()
        gated = _gate(feature_matrix, as_of_timestamp)
        if len(gated) == 0:
            raise MissingDataError("No data at or before as_of_timestamp")

        X = self._extract_features(gated)
        X_scaled = self._transform_features(X)

        raw_states = self._model.predict(X_scaled)
        # Remap internal HMM indices to semantic indices
        semantic = self._state_perm[raw_states]
        regime_names = [REGIME_LABELS[i] for i in semantic]

        return gated.select("interval_start_utc").with_columns(
            pl.Series("regime", regime_names)
        )

    def save(self, path: Path) -> None:
        """Persist the fitted model to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("hmm_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "DARTRegimeModel":
        """Load a previously saved model."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected DARTRegimeModel, got {type(obj)}")
        logger.info("hmm_loaded", path=str(path))
        return obj

    # ── internal helpers ────────────────────────────────────────────────────

    def _extract_features(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise MissingDataError(f"HMM feature columns missing: {missing}")
        return df.select(self.feature_cols).to_numpy().astype(np.float64)

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-8
        return (X - self._feat_mean) / self._feat_std

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feat_mean) / self._feat_std

    def _assign_regime_labels(self, X_scaled: np.ndarray) -> None:
        """Map HMM state indices to semantic labels by mean DART spread."""
        raw_states = self._model.predict(X_scaled)
        # dart_spread_usd is always feature_cols[0]
        dart_col_idx = 0
        state_means = np.array([
            X_scaled[raw_states == k, dart_col_idx].mean() if (raw_states == k).any() else 0.0
            for k in range(self.n_states)
        ])
        # Sort: lowest mean → NEGATIVE_CONGESTION(2), middle → NORMAL(0), highest → SCARCITY(1)
        order = np.argsort(state_means)  # indices of states from lowest to highest dart spread
        # state_perm[hmm_idx] = semantic_idx
        self._state_perm = np.empty(self.n_states, dtype=int)
        semantic_order = [REGIME_NEGATIVE_CONGESTION, REGIME_NORMAL, REGIME_SCARCITY]
        for rank, hmm_idx in enumerate(order):
            self._state_perm[hmm_idx] = semantic_order[rank]

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
