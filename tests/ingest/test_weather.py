"""Tests for Open-Meteo weather ingest module.

Walk-forward contract tests verify that as_of_date correctly gates
historical temperature data to prevent leakage in backtest contexts.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.ingest.exceptions import ExternalDataError, MissingDataError
from src.ingest.external import weather


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_openmeteo_response(
    zone: str,
    start: date,
    hours: int = 24,
    base_temp_c: float = 20.0,
) -> dict:
    """Build a minimal Open-Meteo archive response."""
    times = []
    temps = []
    for h in range(hours):
        dt = datetime(start.year, start.month, start.day, h, tzinfo=timezone.utc)
        times.append(dt.strftime("%Y-%m-%dT%H:%M"))
        temps.append(base_temp_c + h * 0.1)
    return {"hourly": {"time": times, "temperature_2m": temps}}


def _mock_resp(payload: dict) -> MagicMock:
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = payload
    return m


# ── Zone validation ───────────────────────────────────────────────────────────

class TestZoneValidation:
    def test_invalid_zone_raises(self):
        with pytest.raises(ValueError, match="Unknown zone"):
            weather.fetch_zone_historical("INVALID", date(2025, 1, 1), date(2025, 1, 1))

    def test_case_insensitive_zone(self):
        """Zone names are normalized to uppercase internally."""
        payload = _make_openmeteo_response("NORTH", date(2025, 1, 1))
        # Verify that the zone station lookup works regardless of input case.
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = _mock_resp(payload)
            df = weather.fetch_zone_historical(
                "north", date(2025, 1, 1), date(2025, 1, 1),
                as_of_date=date(2025, 1, 3),
            )
        assert df["zone"][0] == "NORTH"


# ── fetch_zone_historical ─────────────────────────────────────────────────────

class TestFetchZoneHistorical:
    def test_returns_canonical_schema(self):
        """Response is parsed into the expected column set."""
        payload = _make_openmeteo_response("NORTH", date(2025, 6, 1))
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = _mock_resp(payload)
            df = weather.fetch_zone_historical(
                "NORTH", date(2025, 6, 1), date(2025, 6, 1),
                as_of_date=date(2025, 6, 3),
            )

        assert set(df.columns) == {"interval_start_utc", "zone", "temp_c", "temp_f", "data_tag"}

    def test_temp_f_conversion(self):
        """Celsius to Fahrenheit conversion: 0°C = 32°F."""
        payload = {
            "hourly": {
                "time": ["2025-06-01T00:00"],
                "temperature_2m": [0.0],
            }
        }
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = _mock_resp(payload)
            df = weather.fetch_zone_historical(
                "NORTH", date(2025, 6, 1), date(2025, 6, 1),
                as_of_date=date(2025, 6, 3),
            )

        assert df["temp_f"][0] == pytest.approx(32.0)

    def test_real_tag(self):
        payload = _make_openmeteo_response("HOUSTON", date(2025, 3, 1))
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = _mock_resp(payload)
            df = weather.fetch_zone_historical(
                "HOUSTON", date(2025, 3, 1), date(2025, 3, 1),
                as_of_date=date(2025, 3, 3),
            )

        assert (df["data_tag"] == "REAL").all()

    def test_as_of_date_caps_end(self):
        """as_of_date − 1 day is the actual ceiling passed to the API."""
        captured_params = {}

        def fake_get(url, params=None, **kwargs):
            captured_params.update(params or {})
            payload = _make_openmeteo_response("NORTH", date(2025, 1, 1))
            return _mock_resp(payload)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get = fake_get
            weather.fetch_zone_historical(
                "NORTH",
                date(2025, 1, 1),
                date(2025, 12, 31),           # requested far future
                as_of_date=date(2025, 1, 3),  # gate: cap to Jan 2
            )

        assert captured_params["end_date"] == "2025-01-02"

    def test_raises_on_empty_hourly(self):
        payload = {"hourly": {"time": [], "temperature_2m": []}}
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = _mock_resp(payload)
            with pytest.raises(MissingDataError):
                weather.fetch_zone_historical(
                    "NORTH", date(2025, 1, 1), date(2025, 1, 1),
                    as_of_date=date(2025, 1, 3),
                )

    def test_raises_on_missing_hourly_key(self):
        payload = {}
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = _mock_resp(payload)
            with pytest.raises(MissingDataError):
                weather.fetch_zone_historical(
                    "NORTH", date(2025, 1, 1), date(2025, 1, 1),
                    as_of_date=date(2025, 1, 3),
                )


# ── Zone station coverage ─────────────────────────────────────────────────────

class TestZoneStationMap:
    def test_all_five_zones_defined(self):
        zones = {"NORTH", "HOUSTON", "SOUTH", "WEST", "COAST"}
        assert set(weather.ZONE_STATIONS.keys()) == zones

    def test_lat_lon_bounds(self):
        """All stations should be in Texas (rough bounding box)."""
        for zone, station in weather.ZONE_STATIONS.items():
            assert 25.0 < station["lat"] < 37.0, f"{zone} lat out of TX range"
            assert -107.0 < station["lon"] < -93.0, f"{zone} lon out of TX range"


# ── save_to_parquet ───────────────────────────────────────────────────────────

class TestSaveToParquet:
    def _sample_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "interval_start_utc": [
                datetime(2024, 12, 31, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "zone": ["NORTH", "NORTH"],
            "temp_c": [5.0, 6.0],
            "temp_f": [41.0, 42.8],
            "data_tag": ["REAL", "REAL"],
        })

    def test_partitions_by_year(self, tmp_path):
        written = weather.save_to_parquet(self._sample_df(), tmp_path)
        assert set(written.keys()) == {2024, 2025}

    def test_parquet_round_trips(self, tmp_path):
        df = self._sample_df()
        written = weather.save_to_parquet(df, tmp_path)
        back = pl.read_parquet(written[2025])
        assert back["temp_c"][0] == pytest.approx(6.0)
