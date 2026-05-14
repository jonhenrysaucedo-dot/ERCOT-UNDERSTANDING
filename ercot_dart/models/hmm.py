"""
Hidden Markov Model for ERCOT Market Regime Detection.

Detects three structural market regimes from the feature matrix:
  - Regime 0 — Normal:             Balanced supply/demand, mean-reverting DART spreads
  - Regime 1 — Scarcity:           Tight reserves, ORDC adder spikes, high volatility
  - Regime 2 — Negative Congestion: Surplus generation, negative LMPs, negative spreads

Mathematical Formulation
------------------------
Let x_t be the d-dimensional observation vector at time t.
The HMM assumes:
  - Hidden state: S_t ∈ {0, 1, 2}
  - Transition: P(S_t | S_{t-1}) = A  (3×3 transition matrix)
  - Emission:   P(x_t | S_t = k) = N(μ_k, Σ_k)  (Gaussian per regime)

Parameters are estimated via the Baum-Welch (EM) algorithm.
The Viterbi algorithm decodes the most probable state sequence.
Regime probabilities (soft assignment) come from the forward-backward algorithm.

Feature Selection for HMM
--------------------------
We use a reduced feature set that captures the regime signal without
overfitting to high-dimensional noise:
  - dart_z_score_168h:    Mean-reversion signal (Normal → near zero)
  - stack_slope_high:     Supply stack steepness (Scarcity → high)
  - peaker_mw:            Peaker capacity on offer (Scarcity → low)
  - shift_factor_proxy:   Congestion signal (Neg. Congestion → strongly negative)
  - implied_heat_rate:    Fuel-cost premium (Scarcity → above 10 MMBtu/MWh)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------

REGIME_LABELS: dict[int, str] = {
    0: "Normal",
    1: "Scarcity",
    2: "NegativeCongestion",
}

# Default HMM observation features — must exist in the feature matrix
DEFAULT_HMM_FEATURES: list[str] = [
    "dart_z_score_168h",
    "stack_slope_high",
    "peaker_mw",
    "shift_factor_proxy",
    "implied_heat_rate",
]

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeForecast:
    """
    Output of RegimeDetector.predict() for a single node/timestamp slice.

    Attributes
    ----------
    regime : int
        Viterbi-decoded (hard) regime label.
    regime_name : str
        Human-readable regime name.
    regime_proba : np.ndarray shape (n_regimes,)
        Soft posterior probability over regimes from forward-backward.
    transition_matrix : np.ndarray shape (n_regimes, n_regimes)
        Fitted HMM transition matrix A — used by the MCMC model as a
        prior on regime persistence.
    """

    regime: np.ndarray                         # shape (T,)
    regime_name: list[str]                     # shape (T,)
    regime_proba: np.ndarray                   # shape (T, n_regimes)
    transition_matrix: np.ndarray              # shape (n_regimes, n_regimes)
    means: np.ndarray                          # shape (n_regimes, n_features)
    covars: np.ndarray                         # shape (n_regimes, n_features, n_features)
    timestamps: pd.Series = field(default_factory=pd.Series)

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "regime": self.regime,
            "regime_name": self.regime_name,
        })
        for k, name in REGIME_LABELS.items():
            df[f"regime_proba_{name.lower()}"] = self.regime_proba[:, k]
        return df


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Fits a 3-state GaussianHMM to ERCOT feature data and assigns market
    regimes via Viterbi decoding + forward-backward soft probabilities.

    The model is trained on a single node's time series (e.g., HB_NORTH)
    and the learned transition matrix is reused when scoring all nodes,
    because regime transitions are system-wide events driven by grid
    conditions, not node-specific.

    Usage
    -----
        detector = RegimeDetector()
        detector.fit(feature_matrix, node="HB_NORTH")
        forecast = detector.predict(feature_matrix)
        df = forecast.as_dataframe()
    """

    def __init__(
        self,
        n_regimes: int = 3,
        hmm_features: Optional[list[str]] = None,
        n_iter: int = 200,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.hmm_features = hmm_features or DEFAULT_HMM_FEATURES
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model: Optional[GaussianHMM] = None
        self._scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False
        self._regime_map: dict[int, int] = {}  # raw HMM state → semantic regime

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _extract_observations(
        self, df: pd.DataFrame, node: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract and scale the HMM observation matrix from the feature DataFrame.

        Returns X (scaled, shape T×d) and lengths ([T]) for hmmlearn's
        multi-sequence API. We pass a single sequence so lengths=[T].
        """
        if node is not None:
            df = df[df["node"] == node].copy()

        df = df.sort_values("timestamp").dropna(subset=self.hmm_features)

        available = [f for f in self.hmm_features if f in df.columns]
        missing = set(self.hmm_features) - set(available)
        if missing:
            logger.warning(
                "HMM features missing from DataFrame — using available subset",
                extra={"missing": list(missing)},
            )

        X_raw = df[available].values.astype(np.float64)
        lengths = [len(X_raw)]
        return X_raw, lengths

    def _semantic_regime_map(self) -> dict[int, int]:
        """
        Map raw HMM states (arbitrary ordering) to semantic regimes by
        inspecting the fitted emission means:
          - Scarcity state has the HIGHEST implied heat rate mean
          - Negative Congestion state has the MOST NEGATIVE shift factor mean
          - Normal state is the remainder

        Returns dict: {raw_state → semantic_regime (0=Normal,1=Scarcity,2=NegCong)}
        """
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")

        means = self._model.means_          # shape (n_regimes, n_features)
        feat_names = [f for f in self.hmm_features if f in self.hmm_features]

        ihr_idx = next(
            (i for i, f in enumerate(feat_names) if "heat_rate" in f), None
        )
        sfp_idx = next(
            (i for i, f in enumerate(feat_names) if "shift_factor" in f), None
        )

        raw_states = list(range(self.n_regimes))
        mapping: dict[int, int] = {}

        # Assign Scarcity (highest IHR)
        if ihr_idx is not None:
            scarcity_raw = int(np.argmax(means[:, ihr_idx]))
            mapping[scarcity_raw] = 1  # Scarcity
        else:
            # Fallback: state with highest DART z-score variance
            scarcity_raw = int(np.argmax(np.diag(self._model.covars_[:, 0, 0])))
            mapping[scarcity_raw] = 1

        # Assign Negative Congestion (most negative shift factor proxy)
        remaining = [s for s in raw_states if s not in mapping]
        if sfp_idx is not None:
            neg_cong_raw = min(remaining, key=lambda s: means[s, sfp_idx])
            mapping[neg_cong_raw] = 2  # NegativeCongestion
        else:
            neg_cong_raw = remaining[0]
            mapping[neg_cong_raw] = 2

        # Normal is the remainder
        for s in raw_states:
            if s not in mapping:
                mapping[s] = 0  # Normal

        logger.info("Regime map", extra={"raw_to_semantic": mapping})
        return mapping

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(
        self,
        feature_matrix: pd.DataFrame,
        node: str = "HB_NORTH",
    ) -> "RegimeDetector":
        """
        Fit the GaussianHMM on a single node's time series.

        The scaler is fit here so that predict() can reuse it on out-of-sample
        data without refitting (prevents data leakage in walk-forward validation).
        """
        X_raw, lengths = self._extract_observations(feature_matrix, node)

        # Fit scaler on training data only
        X_scaled = self._scaler.fit_transform(X_raw)

        logger.info(
            "Fitting GaussianHMM",
            extra={
                "n_regimes": self.n_regimes,
                "n_obs": len(X_scaled),
                "n_features": X_scaled.shape[1],
                "node": node,
            },
        )

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self._model.fit(X_scaled, lengths)

        converged = self._model.monitor_.converged
        log_prob = self._model.monitor_.history[-1]
        logger.info(
            "HMM fitting complete",
            extra={"converged": converged, "log_prob": round(log_prob, 4)},
        )

        self._regime_map = self._semantic_regime_map()
        self._is_fitted = True
        return self

    def predict(
        self,
        feature_matrix: pd.DataFrame,
        node: Optional[str] = None,
    ) -> RegimeForecast:
        """
        Decode regimes for new observations using the fitted HMM.

        Returns both hard (Viterbi) and soft (forward-backward) assignments.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        df = feature_matrix.copy()
        if node is not None:
            df = df[df["node"] == node]
        df = df.sort_values("timestamp").dropna(subset=[
            f for f in self.hmm_features if f in df.columns
        ])

        available = [f for f in self.hmm_features if f in df.columns]
        X_raw = df[available].values.astype(np.float64)
        X_scaled = self._scaler.transform(X_raw)

        # Viterbi decoding — most probable state sequence
        log_prob, raw_states = self._model.decode(X_scaled, algorithm="viterbi")

        # Forward-backward — posterior state probabilities P(S_t | x_{1:T})
        posteriors = self._model.predict_proba(X_scaled)   # shape (T, n_regimes)

        # Remap raw HMM states → semantic regimes
        semantic_states = np.array([self._regime_map[s] for s in raw_states])

        # Reorder posteriors columns to match semantic ordering
        reordered_posteriors = np.zeros_like(posteriors)
        for raw, sem in self._regime_map.items():
            reordered_posteriors[:, sem] = posteriors[:, raw]

        regime_names = [REGIME_LABELS[s] for s in semantic_states]

        logger.info(
            "Regime prediction complete",
            extra={
                "n_obs": len(semantic_states),
                "regime_counts": {
                    REGIME_LABELS[k]: int((semantic_states == k).sum())
                    for k in range(self.n_regimes)
                },
            },
        )

        # Reorder transition matrix to match semantic regime ordering
        A_raw = self._model.transmat_
        A_sem = np.zeros_like(A_raw)
        for raw_i, sem_i in self._regime_map.items():
            for raw_j, sem_j in self._regime_map.items():
                A_sem[sem_i, sem_j] = A_raw[raw_i, raw_j]

        # Reorder emission means/covars
        means_sem = np.zeros_like(self._model.means_)
        covars_sem = np.zeros_like(self._model.covars_)
        for raw, sem in self._regime_map.items():
            means_sem[sem] = self._model.means_[raw]
            covars_sem[sem] = self._model.covars_[raw]

        return RegimeForecast(
            regime=semantic_states,
            regime_name=regime_names,
            regime_proba=reordered_posteriors,
            transition_matrix=A_sem,
            means=means_sem,
            covars=covars_sem,
            timestamps=df["timestamp"].reset_index(drop=True),
        )

    def regime_statistics(
        self, feature_matrix: pd.DataFrame, dart_col: str = "dart_spread"
    ) -> pd.DataFrame:
        """
        Compute per-regime DART spread statistics from the fitted model.

        Useful for validating that the HMM has captured meaningful regimes:
          - Normal:            dart_spread near zero, low vol
          - Scarcity:          dart_spread strongly positive, high vol
          - Negative Congestion: dart_spread negative, moderate vol
        """
        forecast = self.predict(feature_matrix)
        regime_df = forecast.as_dataframe()

        merged = feature_matrix.merge(
            regime_df[["timestamp", "regime", "regime_name"]],
            on="timestamp",
            how="inner",
        )

        stats = (
            merged.groupby("regime_name")[dart_col]
            .agg(["mean", "std", "skew", "count"])
            .rename(columns={"mean": "dart_mean", "std": "dart_std",
                             "skew": "dart_skew", "count": "n_obs"})
        )
        return stats

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler,
                         "regime_map": self._regime_map,
                         "hmm_features": self.hmm_features,
                         "n_regimes": self.n_regimes}, f)
        logger.info("RegimeDetector saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> "RegimeDetector":
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        detector = cls(
            n_regimes=state["n_regimes"],
            hmm_features=state["hmm_features"],
        )
        detector._model = state["model"]
        detector._scaler = state["scaler"]
        detector._regime_map = state["regime_map"]
        detector._is_fitted = True
        logger.info("RegimeDetector loaded", extra={"path": str(path)})
        return detector
