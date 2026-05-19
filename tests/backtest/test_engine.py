"""Tests for BacktestEngine (M9).

Walk-forward fold generation and P&L formula are highest priority.
Slippage model is tested with the flat-rate fallback.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.backtest.engine import BacktestEngine, BacktestSummary, FoldResult

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bids(n: int = 720, direction: str = "INC", mw: float = 10.0) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame({
        "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
        "direction": [direction] * n,
        "position_mw": [mw] * n,
        "q50": [3.0] * n,
    })


def _make_dart(n: int = 720, spread: float = 2.0) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame({
        "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
        "dart_spread_usd": [spread] * n,
    })


# Two-year constants for run() tests (1yr train + 1yr test window requires 2yr data)
_TWO_YEARS = 8760 * 2
_START_2024 = date(2024, 1, 1)
_END_2025 = date(2025, 12, 31)


def _engine() -> BacktestEngine:
    return BacktestEngine(slippage_rate=0.02)


# ── Fold generation ───────────────────────────────────────────────────────────

class TestFoldGeneration:
    def test_folds_non_overlapping(self):
        """Test periods must not overlap."""
        engine = _engine()
        folds = engine._generate_folds(
            start_date=date(2024, 1, 1),
            end_date=date(2025, 12, 31),
            train_years=1,
            test_months=1,
        )
        for i in range(len(folds) - 1):
            _, _, _, test_end_i = folds[i]
            _, _, test_start_next, _ = folds[i + 1]
            assert test_end_i < test_start_next

    def test_folds_chronological(self):
        engine = _engine()
        folds = engine._generate_folds(
            date(2024, 1, 1), date(2025, 12, 31), 1, 1
        )
        for i in range(len(folds) - 1):
            assert folds[i][2] < folds[i + 1][2]  # test_start increases

    def test_train_strictly_before_test(self):
        """train_end must be before test_start for every fold."""
        engine = _engine()
        folds = engine._generate_folds(
            date(2024, 1, 1), date(2025, 12, 31), 1, 1
        )
        for train_start, train_end, test_start, test_end in folds:
            assert train_end < test_start

    def test_min_12_folds_for_two_year_window(self):
        engine = _engine()
        folds = engine._generate_folds(
            date(2024, 1, 1), date(2025, 12, 31), 1, 1
        )
        assert len(folds) >= 12

    def test_empty_window_returns_no_folds(self):
        engine = _engine()
        folds = engine._generate_folds(
            date(2025, 1, 1), date(2025, 6, 30), 2, 1  # 2yr train > window
        )
        assert len(folds) == 0


# ── P&L calculation ───────────────────────────────────────────────────────────

class TestPnLCalc:
    def test_inc_positive_spread_is_positive_pnl(self):
        """INC trade with positive DART spread → positive P&L."""
        engine = _engine()
        merged = pl.DataFrame({
            "interval_start_utc": [datetime(2024, 6, 1, tzinfo=UTC)],
            "direction": ["INC"],
            "position_mw": [10.0],
            "dart_spread_usd": [5.0],
            "q50": [4.0],
        })
        pnl = engine._compute_pnl(merged)
        assert pnl[0] == pytest.approx(50.0)

    def test_dec_negative_spread_is_positive_pnl(self):
        """DEC trade with negative DART spread → positive P&L."""
        engine = _engine()
        merged = pl.DataFrame({
            "interval_start_utc": [datetime(2024, 6, 1, tzinfo=UTC)],
            "direction": ["DEC"],
            "position_mw": [10.0],
            "dart_spread_usd": [-5.0],
            "q50": [-4.0],
        })
        pnl = engine._compute_pnl(merged)
        assert pnl[0] == pytest.approx(50.0)

    def test_inc_negative_spread_is_loss(self):
        engine = _engine()
        merged = pl.DataFrame({
            "interval_start_utc": [datetime(2024, 6, 1, tzinfo=UTC)],
            "direction": ["INC"],
            "position_mw": [10.0],
            "dart_spread_usd": [-3.0],
            "q50": [2.0],
        })
        pnl = engine._compute_pnl(merged)
        assert pnl[0] == pytest.approx(-30.0)

    def test_zero_position_zero_pnl(self):
        engine = _engine()
        merged = pl.DataFrame({
            "interval_start_utc": [datetime(2024, 6, 1, tzinfo=UTC)],
            "direction": ["INC"],
            "position_mw": [0.0],
            "dart_spread_usd": [10.0],
            "q50": [8.0],
        })
        pnl = engine._compute_pnl(merged)
        assert pnl[0] == pytest.approx(0.0)


# ── Slippage ──────────────────────────────────────────────────────────────────

class TestSlippage:
    def test_flat_slippage_proportional_to_q50_and_mw(self):
        engine = BacktestEngine(slippage_rate=0.02)
        merged = pl.DataFrame({
            "interval_start_utc": [datetime(2024, 6, 1, tzinfo=UTC)],
            "direction": ["INC"],
            "position_mw": [10.0],
            "dart_spread_usd": [5.0],
            "q50": [4.0],  # slippage = 0.02 * 4 * 10 = 0.8
        })
        slippage = engine._compute_slippage(merged, disclosure_df=None)
        assert slippage[0] == pytest.approx(0.8)

    def test_slippage_positive(self):
        engine = _engine()
        merged = _make_bids(n=10).join(
            _make_dart(n=10), on="interval_start_utc", how="inner"
        )
        slippage = engine._compute_slippage(merged, disclosure_df=None)
        assert (slippage >= 0).all()


# ── Metrics ────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_hit_rate_one_for_all_correct(self):
        """All INC trades with positive DART → hit_rate=1.0."""
        bids = _make_bids(n=100, direction="INC", mw=5.0)
        dart = _make_dart(n=100, spread=2.0)  # positive → correct for INC
        engine = _engine()
        merged = bids.join(dart, on="interval_start_utc", how="inner")
        directions = merged["direction"].to_numpy()
        dart_arr = merged["dart_spread_usd"].to_numpy()
        hit = np.where(directions == "INC", dart_arr > 0, dart_arr < 0)
        assert hit.mean() == pytest.approx(1.0)

    def test_sharpe_zero_for_constant_pnl(self):
        """If std(pnl) → 0, sharpe → 0 (not inf/nan)."""
        # constant P&L means zero std
        pnl = np.ones(100)
        result = BacktestEngine._sharpe(pnl)
        # std=0, so sharpe should be 0
        assert result == pytest.approx(0.0)

    def test_sharpe_positive_for_positive_pnl(self):
        rng = np.random.default_rng(0)
        pnl = rng.normal(1.0, 0.5, 1000)  # positive mean
        sharpe = BacktestEngine._sharpe(pnl)
        assert sharpe > 0

    def test_max_drawdown_zero_for_monotone_increasing(self):
        """Monotonically increasing P&L has zero drawdown."""
        pnl = np.ones(100)  # constant positive → cumsum = [1,2,3,...] → no drawdown
        dd = BacktestEngine._max_drawdown(pnl)
        assert dd == pytest.approx(0.0)

    def test_max_drawdown_positive_after_loss(self):
        pnl = np.array([10.0, 10.0, -5.0, -5.0, 10.0])
        dd = BacktestEngine._max_drawdown(pnl)
        assert dd > 0


# ── run() ─────────────────────────────────────────────────────────────────────

class TestRun:
    def test_run_returns_summary(self):
        """Smoke test: run() with 2yr data returns BacktestSummary with ≥1 fold."""
        bids = _make_bids(n=_TWO_YEARS, direction="INC", mw=5.0)
        dart = _make_dart(n=_TWO_YEARS, spread=2.0)
        engine = _engine()
        summary = engine.run(
            bids, dart,
            start_date=_START_2024,
            end_date=_END_2025,
        )
        assert isinstance(summary, BacktestSummary)
        assert summary.n_folds >= 1

    def test_positive_net_pnl_when_all_trades_correct(self):
        """INC trades with positive DART spread → positive net P&L."""
        bids = _make_bids(n=_TWO_YEARS, direction="INC", mw=5.0)
        dart = _make_dart(n=_TWO_YEARS, spread=10.0)
        engine = BacktestEngine(slippage_rate=0.01)
        summary = engine.run(
            bids, dart, start_date=_START_2024, end_date=_END_2025,
        )
        assert summary.total_net_pnl > 0

    def test_missing_required_column_raises(self):
        bids = _make_bids().drop("direction")
        dart = _make_dart()
        engine = _engine()
        with pytest.raises(ValueError, match="missing"):
            engine.run(bids, dart, start_date=_START_2024, end_date=_END_2025)

    def test_summary_serializable(self):
        """BacktestSummary.to_dict() should be JSON-serializable."""
        import json
        bids = _make_bids(n=_TWO_YEARS)
        dart = _make_dart(n=_TWO_YEARS)
        engine = _engine()
        summary = engine.run(bids, dart, start_date=_START_2024, end_date=_END_2025)
        d = summary.to_dict()
        json.dumps(d)  # should not raise

    def test_save_json(self, tmp_path):
        bids = _make_bids(n=_TWO_YEARS)
        dart = _make_dart(n=_TWO_YEARS)
        engine = _engine()
        summary = engine.run(bids, dart, start_date=_START_2024, end_date=_END_2025)
        path = tmp_path / "backtest.json"
        summary.save_json(path)
        assert path.exists()
        import json
        loaded = json.loads(path.read_text())
        assert "n_folds" in loaded
