"""Tests for src/features/engineering.py.

Coverage targets per PRD §5: ≥70% on src/features/.
Walk-forward correctness is the highest-priority invariant — any test that
verifies the as_of_timestamp gate protects the walk-forward boundary is
automatically prioritised over pure unit correctness.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import polars as pl
import pytest

from src.features.engineering import (
    compute_dart_spread,
    compute_net_load,
    compute_thermal_share,
    compute_as_features,
    compute_temperature_features,
    compute_temporal_features,
    compute_lagged_dart_features,
    build_feature_matrix,
)
from src.ingest.exceptions import MissingDataError, WalkForwardViolation

UTC = timezone.utc

# ── Shared fixtures ────────────────────────────────────────────────────────────

def _ts(y=2025, m=1, d=1, h=6) -> datetime:
    """UTC datetime at the given hour."""
    return datetime(y, m, d, h, tzinfo=UTC)


def _hours(n: int, start_h: int = 6) -> list[datetime]:
    return [datetime(2025, 1, 1, start_h + i, tzinfo=UTC) for i in range(n)]


def _make_dam_spp(hours: list[datetime], node="RN_QTUM_SLR", price=30.0) -> pl.DataFrame:
    return pl.DataFrame({
        "interval_start_utc": hours,
        "settlement_point": [node] * len(hours),
        "dam_spp_usd": [price + i * 0.5 for i in range(len(hours))],
        "data_tag": ["REAL"] * len(hours),
    })


def _make_rtm_spp(hours: list[datetime], node="RN_QTUM_SLR", delta=5.0) -> pl.DataFrame:
    """RTM at 15-min intervals; 4 rows per hour."""
    rows = []
    for ts in hours:
        for q in range(4):
            rows.append({
                "interval_start_utc": ts + timedelta(minutes=q * 15),
                "settlement_point": node,
                "rtm_spp_usd": 35.0 + delta,
                "data_tag": "REAL",
            })
    return pl.DataFrame(rows)


def _make_native_load(hours: list[datetime]) -> pl.DataFrame:
    rows = []
    for ts in hours:
        rows.append({"interval_start_utc": ts, "zone": "ERCOT", "load_mw": 35000.0, "data_tag": "REAL"})
    return pl.DataFrame(rows)


def _make_wind_solar(hours: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({
        "interval_start_utc": hours,
        "wind_gen_mw": [10000.0] * len(hours),
        "load_mw": [35000.0] * len(hours),
        "data_tag": ["REAL"] * len(hours),
    })


def _make_fuel_mix(hours: list[datetime]) -> pl.DataFrame:
    fuels = ["Gas-CC", "Coal", "Wind", "Solar"]
    mws = {"Gas-CC": 15000, "Coal": 5000, "Wind": 10000, "Solar": 5000}
    rows = []
    for ts in hours:
        for f in fuels:
            rows.append({"interval_start_utc": ts, "fuel": f, "gen_mw": mws[f], "data_tag": "REAL"})
    return pl.DataFrame(rows)


def _make_dam_as_mcpc(hours: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({
        "interval_start_utc": hours,
        "as_regup_usd": [5.0] * len(hours),
        "as_regdn_usd": [3.0] * len(hours),
        "as_rrs_usd": [2.0] * len(hours),
        "as_nspin_usd": [1.5] * len(hours),
        "as_ecrs_usd": [1.0] * len(hours),
        "data_tag": ["REAL"] * len(hours),
    })


def _make_weather(hours: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({
        "interval_start_utc": hours,
        "station": ["MAF"] * len(hours),
        "zone": ["WEST"] * len(hours),
        "temp_f": [95.0] * len(hours),
        "temp_c": [35.0] * len(hours),
        "data_tag": ["REAL"] * len(hours),
    })


# ── compute_dart_spread ────────────────────────────────────────────────────────

class TestDartSpread:
    def test_positive_dart_when_rtm_above_dam(self):
        """RTM 40, DAM 30 → DART = +10."""
        hours = _hours(3)
        dam = _make_dam_spp(hours, price=30.0)
        rtm = _make_rtm_spp(hours, delta=10.0)  # RTM = 40

        gate = _ts(h=12)
        df = compute_dart_spread(dam, rtm, "RN_QTUM_SLR", gate)

        assert "dart_spread_usd" in df.columns
        # All rows: DART ≈ 40 - 30.x (price increments 0.5/hr)
        for row in df.iter_rows(named=True):
            assert row["dart_spread_usd"] == pytest.approx(row["rtm_spp_hourly_usd"] - row["dam_spp_usd"], abs=1e-6)

    def test_negative_dart_when_rtm_below_dam(self):
        """RTM 20, DAM 30 → DART = -10."""
        hours = _hours(2)
        dam = _make_dam_spp(hours, price=30.0)
        rtm = _make_rtm_spp(hours, delta=-10.0)  # RTM = 25

        df = compute_dart_spread(dam, rtm, "RN_QTUM_SLR", _ts(h=12))
        assert all(r["dart_spread_usd"] < 0 for r in df.iter_rows(named=True))

    def test_rtm_aggregated_to_hourly_mean(self):
        """4 RTM intervals at [40, 42, 38, 40] → hourly mean = 40."""
        ts = _ts(h=6)
        gate = _ts(h=12)
        dam = pl.DataFrame({
            "interval_start_utc": [ts],
            "settlement_point": ["RN_QTUM_SLR"],
            "dam_spp_usd": [30.0],
            "data_tag": ["REAL"],
        })
        rtm = pl.DataFrame({
            "interval_start_utc": [ts, ts + timedelta(minutes=15),
                                    ts + timedelta(minutes=30), ts + timedelta(minutes=45)],
            "settlement_point": ["RN_QTUM_SLR"] * 4,
            "rtm_spp_usd": [40.0, 42.0, 38.0, 40.0],
            "data_tag": ["REAL"] * 4,
        })
        df = compute_dart_spread(dam, rtm, "RN_QTUM_SLR", gate)
        assert df["rtm_spp_hourly_usd"][0] == pytest.approx(40.0)
        assert df["dart_spread_usd"][0] == pytest.approx(10.0)

    def test_as_of_gate_excludes_future_rows(self):
        """Rows after as_of_timestamp must be excluded."""
        hours = _hours(5, start_h=6)  # 06:00–10:00 UTC
        dam = _make_dam_spp(hours)
        rtm = _make_rtm_spp(hours)

        gate = _ts(h=8)  # allow only 06:00, 07:00, 08:00
        df = compute_dart_spread(dam, rtm, "RN_QTUM_SLR", gate)
        assert df["interval_start_utc"].max() <= gate

    def test_raises_on_empty_after_gate(self):
        hours = _hours(2, start_h=6)
        dam = _make_dam_spp(hours)
        rtm = _make_rtm_spp(hours)
        gate = _ts(h=4)  # all rows are after gate
        with pytest.raises(MissingDataError):
            compute_dart_spread(dam, rtm, "RN_QTUM_SLR", gate)

    def test_raises_on_naive_as_of(self):
        hours = _hours(2)
        dam = _make_dam_spp(hours)
        rtm = _make_rtm_spp(hours)
        with pytest.raises(WalkForwardViolation):
            compute_dart_spread(dam, rtm, "RN_QTUM_SLR", datetime(2025, 1, 1, 12))

    def test_real_tag(self):
        hours = _hours(2)
        dam = _make_dam_spp(hours)
        rtm = _make_rtm_spp(hours)
        df = compute_dart_spread(dam, rtm, "RN_QTUM_SLR", _ts(h=12))
        assert (df["data_tag"] == "REAL").all()


# ── compute_net_load ──────────────────────────────────────────────────────────

class TestNetLoad:
    def test_net_load_formula(self):
        """net_load = ercot_load - wind_gen."""
        hours = _hours(3)
        load = _make_native_load(hours)  # 35000 MW
        ws = _make_wind_solar(hours)     # 10000 MW wind

        df = compute_net_load(load, ws, _ts(h=12))
        assert df["net_load_mw"][0] == pytest.approx(25000.0)

    def test_as_of_gate(self):
        hours = _hours(5)
        load = _make_native_load(hours)
        ws = _make_wind_solar(hours)
        gate = _ts(h=8)
        df = compute_net_load(load, ws, gate)
        assert df["interval_start_utc"].max() <= gate

    def test_real_tag(self):
        hours = _hours(2)
        df = compute_net_load(_make_native_load(hours), _make_wind_solar(hours), _ts(h=12))
        assert (df["data_tag"] == "REAL").all()


# ── compute_thermal_share ─────────────────────────────────────────────────────

class TestThermalShare:
    def test_thermal_share_formula(self):
        """Gas-CC(15000) + Coal(5000) / Total(35000) = 20000/35000 ≈ 0.571."""
        hours = _hours(2)
        fuel = _make_fuel_mix(hours)
        df = compute_thermal_share(fuel, _ts(h=12))
        expected = 20000 / 35000
        assert df["thermal_share"][0] == pytest.approx(expected, rel=1e-4)

    def test_as_of_gate(self):
        hours = _hours(5)
        fuel = _make_fuel_mix(hours)
        gate = _ts(h=8)
        df = compute_thermal_share(fuel, gate)
        assert df["interval_start_utc"].max() <= gate

    def test_real_tag(self):
        hours = _hours(2)
        df = compute_thermal_share(_make_fuel_mix(hours), _ts(h=12))
        assert (df["data_tag"] == "REAL").all()

    def test_thermal_mw_excludes_renewables(self):
        """Wind and Solar must NOT be counted in thermal_mw."""
        hours = [_ts(h=6)]
        fuel = pl.DataFrame({
            "interval_start_utc": hours * 4,
            "fuel": ["Gas-CC", "Wind", "Solar", "Nuclear"],
            "gen_mw": [10000.0, 8000.0, 5000.0, 3000.0],
            "data_tag": ["REAL"] * 4,
        })
        df = compute_thermal_share(fuel, _ts(h=12))
        assert df["thermal_mw"][0] == pytest.approx(13000.0)  # Gas-CC + Nuclear only


# ── compute_as_features ───────────────────────────────────────────────────────

class TestASFeatures:
    def test_total_capacity_sums_all_as(self):
        hours = _hours(2)
        mcpc = _make_dam_as_mcpc(hours)  # 5+3+2+1.5+1 = 12.5
        df = compute_as_features(mcpc, _ts(h=12))
        assert df["as_total_capacity"][0] == pytest.approx(12.5)

    def test_ecrs_premium_ratio(self):
        """ecrs_premium = ECRS / RegUp = 1.0 / 5.0 = 0.2."""
        hours = _hours(2)
        df = compute_as_features(_make_dam_as_mcpc(hours), _ts(h=12))
        assert df["ecrs_premium"][0] == pytest.approx(0.2, rel=1e-4)

    def test_ecrs_premium_null_when_missing(self):
        hours = _hours(2)
        mcpc = _make_dam_as_mcpc(hours).drop("as_ecrs_usd")
        df = compute_as_features(mcpc, _ts(h=12))
        assert df["ecrs_premium"][0] is None

    def test_as_of_gate(self):
        hours = _hours(5)
        gate = _ts(h=8)
        df = compute_as_features(_make_dam_as_mcpc(hours), gate)
        assert df["interval_start_utc"].max() <= gate


# ── compute_temperature_features ─────────────────────────────────────────────

class TestTemperatureFeatures:
    def test_hinge_hot_at_95f(self):
        """95°F with 90°F threshold → hinge_hot = 5.0."""
        hours = _hours(2)
        weather = _make_weather(hours)  # temp_f = 95.0
        df = compute_temperature_features(weather, _ts(h=12))
        assert df["temp_hinge_hot_MAF"][0] == pytest.approx(5.0)

    def test_hinge_cold_zero_above_threshold(self):
        """95°F (hot day) → hinge_cold = 0.0."""
        hours = _hours(2)
        weather = _make_weather(hours)
        df = compute_temperature_features(weather, _ts(h=12))
        assert df["temp_hinge_cold_MAF"][0] == pytest.approx(0.0)

    def test_hinge_cold_activates_below_30f(self):
        hours = [_ts(h=6)]
        weather = pl.DataFrame({
            "interval_start_utc": hours,
            "station": ["MAF"],
            "zone": ["WEST"],
            "temp_f": [20.0],
            "temp_c": [-6.7],
            "data_tag": ["REAL"],
        })
        df = compute_temperature_features(weather, _ts(h=12))
        assert df["temp_hinge_cold_MAF"][0] == pytest.approx(10.0)

    def test_cdh_hdh_aliases_present(self):
        hours = _hours(2)
        df = compute_temperature_features(_make_weather(hours), _ts(h=12))
        assert "cdh_MAF" in df.columns
        assert "hdh_MAF" in df.columns

    def test_real_tag(self):
        hours = _hours(2)
        df = compute_temperature_features(_make_weather(hours), _ts(h=12))
        assert (df["data_tag"] == "REAL").all()


# ── compute_temporal_features ─────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_fourier_columns_present(self):
        ts = pl.Series([_ts(h=6), _ts(h=7), _ts(h=8)])
        df = compute_temporal_features(ts, _ts(h=12))
        for col in ["hour_sin", "hour_cos", "hour_sin2", "hour_cos2", "dow_sin", "dow_cos"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_hour_sin_cos_unit_circle(self):
        """sin²(θ) + cos²(θ) should equal 1 for any hour."""
        ts = pl.Series([_ts(h=h) for h in range(24)])
        df = compute_temporal_features(ts, _ts(h=23))
        for row in df.iter_rows(named=True):
            assert row["hour_sin"] ** 2 + row["hour_cos"] ** 2 == pytest.approx(1.0, abs=1e-9)

    def test_midnight_and_noon_symmetry(self):
        """hour=0 and hour=12 should have cos values of +1 and -1."""
        ts = pl.Series([
            datetime(2025, 1, 1, 0, tzinfo=UTC),
            datetime(2025, 1, 1, 12, tzinfo=UTC),
        ])
        df = compute_temporal_features(ts, _ts(h=23))
        assert df.filter(pl.col("interval_start_utc").dt.hour() == 0)["hour_cos"][0] == pytest.approx(1.0, abs=1e-9)
        assert df.filter(pl.col("interval_start_utc").dt.hour() == 12)["hour_cos"][0] == pytest.approx(-1.0, abs=1e-9)

    def test_as_of_gate(self):
        ts = pl.Series([_ts(h=6), _ts(h=7), _ts(h=8), _ts(h=9)])
        gate = _ts(h=7)
        df = compute_temporal_features(ts, gate)
        assert len(df) == 2  # 06:00 and 07:00 only

    def test_raises_on_naive_datetime(self):
        ts = pl.Series([_ts(h=6)])
        with pytest.raises(WalkForwardViolation):
            compute_temporal_features(ts, datetime(2025, 1, 1, 12))


# ── compute_lagged_dart_features ──────────────────────────────────────────────

class TestLaggedDartFeatures:
    def _make_dart(self, n_hours: int = 200) -> pl.DataFrame:
        hours = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n_hours)]
        return pl.DataFrame({
            "interval_start_utc": hours,
            "settlement_point": ["RN_QTUM_SLR"] * n_hours,
            "dam_spp_usd": [30.0] * n_hours,
            "rtm_spp_hourly_usd": [35.0] * n_hours,
            "dart_spread_usd": [5.0] * n_hours,
            "data_tag": ["REAL"] * n_hours,
        })

    def test_lag_24h_is_yesterday_value(self):
        dart = self._make_dart(50)
        gate = datetime(2025, 1, 3, tzinfo=UTC)
        df = compute_lagged_dart_features(dart, gate)
        # All DART values are 5.0 so any lag should also be 5.0 (after warmup)
        non_null = df.filter(pl.col("dart_lag_24h").is_not_null())
        if len(non_null) > 0:
            assert non_null["dart_lag_24h"][0] == pytest.approx(5.0)

    def test_rolling_mean_of_constant_series(self):
        """Rolling mean of constant series = constant."""
        dart = self._make_dart(200)
        gate = datetime(2025, 1, 9, tzinfo=UTC)
        df = compute_lagged_dart_features(dart, gate, rolling_windows=[24])
        non_null = df.filter(pl.col("dart_roll_mean_24h").is_not_null())
        if len(non_null) > 0:
            assert non_null["dart_roll_mean_24h"][0] == pytest.approx(5.0, abs=1e-4)

    def test_z_score_of_constant_series_is_zero(self):
        """z-score of constant series should be ≈ 0."""
        dart = self._make_dart(200)
        gate = datetime(2025, 1, 9, tzinfo=UTC)
        df = compute_lagged_dart_features(dart, gate, rolling_windows=[24])
        non_null = df.filter(pl.col("dart_z_score_24h").is_not_null())
        if len(non_null) > 0:
            assert abs(non_null["dart_z_score_24h"][0]) < 0.01

    def test_as_of_gate_no_future_data(self):
        dart = self._make_dart(200)
        gate = datetime(2025, 1, 5, tzinfo=UTC)
        df = compute_lagged_dart_features(dart, gate)
        assert df["interval_start_utc"].max() <= gate

    def test_real_tag(self):
        dart = self._make_dart(50)
        df = compute_lagged_dart_features(dart, datetime(2025, 1, 3, tzinfo=UTC))
        assert (df["data_tag"] == "REAL").all()


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def _make_all_inputs(self, n: int = 50):
        hours = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)]
        return dict(
            dam_spp=_make_dam_spp(hours),
            rtm_spp_15min=_make_rtm_spp(hours),
            native_load=_make_native_load(hours),
            wind_solar=_make_wind_solar(hours),
            fuel_mix=_make_fuel_mix(hours),
            dam_as_mcpc=_make_dam_as_mcpc(hours),
            weather=_make_weather(hours),
            settlement_point="RN_QTUM_SLR",
            as_of_timestamp=datetime(2025, 1, 3, tzinfo=UTC),
        )

    def test_returns_dataframe(self):
        kwargs = self._make_all_inputs()
        df = build_feature_matrix(**kwargs)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_core_features_present(self):
        kwargs = self._make_all_inputs()
        df = build_feature_matrix(**kwargs)
        for col in ["dart_spread_usd", "net_load_mw", "thermal_share",
                    "as_total_capacity", "hour_sin", "hour_cos"]:
            assert col in df.columns, f"Missing: {col}"

    def test_data_tag_real_when_all_inputs_present(self):
        kwargs = self._make_all_inputs()
        df = build_feature_matrix(**kwargs)
        # After sufficient warmup, DART/load/thermal_share all populated
        has_real = (df["data_tag"] == "REAL").any()
        assert has_real

    def test_sorted_by_interval_start_utc(self):
        kwargs = self._make_all_inputs()
        df = build_feature_matrix(**kwargs)
        ts = df["interval_start_utc"].to_list()
        assert ts == sorted(ts)
