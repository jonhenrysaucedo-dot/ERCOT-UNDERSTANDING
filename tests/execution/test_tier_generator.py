"""Tests for TierBidGenerator (M8).

Monotonicity enforcement is the highest-priority test — ERCOT rejects non-monotone curves.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.ingest.exceptions import WalkForwardViolation
from src.execution.tier_generator import NUM_TIERS, TierBidGenerator

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


def _make_kelly(n: int = 6, direction: str = "INC", mw: float = 20.0) -> pl.DataFrame:
    base = datetime(2025, 6, 1, tzinfo=UTC)
    return pl.DataFrame({
        "interval_start_utc": [base + timedelta(hours=h) for h in range(n)],
        "direction": [direction] * n,
        "position_mw": [mw] * n,
        "kelly_fraction_raw": [0.25] * n,
        "kelly_fraction_damped": [0.25] * n,
        "kelly_fraction_final": [0.22] * n,
    })


def _gen() -> TierBidGenerator:
    return TierBidGenerator(price_cap=5000.0, price_floor=-250.0)


# ── Initialization ────────────────────────────────────────────────────────────

class TestInit:
    def test_wrong_tier_count_raises(self):
        with pytest.raises(ValueError, match="10 tiers"):
            TierBidGenerator(num_tiers=9)

    def test_default_tiers(self):
        g = TierBidGenerator()
        assert g.num_tiers == NUM_TIERS


# ── Walk-forward gate ─────────────────────────────────────────────────────────

class TestWalkForwardGate:
    def test_naive_datetime_raises(self):
        g = _gen()
        with pytest.raises(WalkForwardViolation):
            g.generate_bids(
                _make_forecast(), _make_kelly(), datetime(2025, 6, 2)  # naive
            )

    def test_utc_datetime_accepted(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(), _make_kelly(), datetime(2025, 6, 2, tzinfo=UTC)
        )
        assert len(result) > 0


# ── Output schema ─────────────────────────────────────────────────────────────

class TestOutputSchema:
    def test_required_columns(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(), _make_kelly(), datetime(2025, 6, 2, tzinfo=UTC)
        )
        expected = {
            "interval_start_utc", "hour_ending", "direction",
            "tier", "price_usd_per_mwh", "quantity_mw", "is_tail_tier",
        }
        assert set(result.columns) == expected

    def test_ten_tiers_per_hour(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=3), _make_kelly(n=3), datetime(2025, 6, 2, tzinfo=UTC)
        )
        for ts in result["interval_start_utc"].unique():
            hour_rows = result.filter(pl.col("interval_start_utc") == ts)
            assert len(hour_rows) == NUM_TIERS

    def test_tier_numbers_one_to_ten(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=1), _make_kelly(n=1), datetime(2025, 6, 2, tzinfo=UTC)
        )
        assert set(result["tier"].to_list()) == set(range(1, NUM_TIERS + 1))

    def test_zero_position_hours_excluded(self):
        g = _gen()
        zero_kelly = _make_kelly(n=3, mw=0.0)
        result = g.generate_bids(
            _make_forecast(n=3), zero_kelly, datetime(2025, 6, 2, tzinfo=UTC)
        )
        assert len(result) == 0

    def test_empty_result_has_correct_schema(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=2), _make_kelly(n=2, mw=0.0),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert "tier" in result.columns
        assert len(result) == 0


# ── Monotonicity ──────────────────────────────────────────────────────────────

class TestMonotonicity:
    def test_inc_prices_strictly_increasing(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=6), _make_kelly(n=6, direction="INC"),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        for ts in result["interval_start_utc"].unique():
            prices = (
                result.filter(pl.col("interval_start_utc") == ts)
                .sort("tier")["price_usd_per_mwh"]
                .to_numpy()
            )
            diffs = np.diff(prices)
            assert (diffs > 0).all(), f"INC prices not strictly increasing for {ts}: {prices}"

    def test_dec_prices_strictly_decreasing(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=6), _make_kelly(n=6, direction="DEC"),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        for ts in result["interval_start_utc"].unique():
            prices = (
                result.filter(pl.col("interval_start_utc") == ts)
                .sort("tier")["price_usd_per_mwh"]
                .to_numpy()
            )
            diffs = np.diff(prices)
            assert (diffs < 0).all(), f"DEC prices not strictly decreasing for {ts}: {prices}"

    def test_validate_monotonicity_passes_valid_bids(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=3), _make_kelly(n=3, direction="INC"),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        g.validate_monotonicity(result)  # should not raise

    def test_validate_monotonicity_raises_on_violation(self):
        g = _gen()
        # Manually construct a non-monotone bid
        base = datetime(2025, 6, 1, tzinfo=UTC)
        bad_bids = pl.DataFrame({
            "interval_start_utc": [base] * NUM_TIERS,
            "hour_ending": [2] * NUM_TIERS,
            "direction": ["INC"] * NUM_TIERS,
            "tier": list(range(1, NUM_TIERS + 1)),
            "price_usd_per_mwh": [float(i) for i in range(NUM_TIERS, 0, -1)],  # decreasing!
            "quantity_mw": [1.0] * NUM_TIERS,
            "is_tail_tier": [False] * (NUM_TIERS - 2) + [True, True],
        })
        with pytest.raises(ValueError, match="non-strictly-increasing"):
            g.validate_monotonicity(bad_bids)


# ── Price bounds ──────────────────────────────────────────────────────────────

class TestPriceBounds:
    def test_prices_within_ercot_bounds(self):
        g = TierBidGenerator(price_cap=5000.0, price_floor=-250.0)
        result = g.generate_bids(
            _make_forecast(n=6), _make_kelly(n=6),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert (result["price_usd_per_mwh"] >= -250.0).all()
        assert (result["price_usd_per_mwh"] <= 5000.0).all()

    def test_tail_tiers_marked(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=1), _make_kelly(n=1),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        tail = result.filter(pl.col("is_tail_tier"))
        non_tail = result.filter(~pl.col("is_tail_tier"))
        assert len(tail) == 2  # tiers 9 and 10
        assert len(non_tail) == 8


# ── Quantity allocation ────────────────────────────────────────────────────────

class TestQuantityAllocation:
    def test_total_quantity_equals_position_mw(self):
        g = _gen()
        mw = 20.0
        result = g.generate_bids(
            _make_forecast(n=1), _make_kelly(n=1, mw=mw),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        total = result["quantity_mw"].sum()
        assert abs(total - mw) < 0.01

    def test_quantities_positive(self):
        g = _gen()
        result = g.generate_bids(
            _make_forecast(n=3), _make_kelly(n=3, mw=30.0),
            datetime(2025, 6, 2, tzinfo=UTC),
        )
        assert (result["quantity_mw"] > 0).all()
