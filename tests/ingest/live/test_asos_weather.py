"""Tests for Iowa State ASOS weather client.

All HTTP calls are mocked — no real network required in CI.
Walk-forward gate tests are highest priority.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.exceptions import ExternalDataError, MissingDataError
from src.ingest.live.asos_weather import (
    STATIONS,
    ALL_STATIONS,
    WEST_TX_STATIONS,
    fetch_station,
    save_to_parquet,
    _parse_asos_csv,
)

UTC = timezone.utc


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_CSV = """#
station,valid,tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,gust,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,ice_accretion_1hr,ice_accretion_3hr,ice_accretion_6hr,peak_wind_gust,peak_wind_drct,peak_wind_time,feel,metar,snowdepth
MAF,2025-06-01 12:00,95.0,60.0,30.5,180,15,0.00,29.82,1010.0,10.0,M,CLR,M,M,M,M,M,M,M,M,M,M,M,M,M,M,88.0,METAR...,M
MAF,2025-06-01 13:00,97.0,61.0,29.0,190,18,0.00,29.80,1009.5,10.0,M,CLR,M,M,M,M,M,M,M,M,M,M,M,M,M,M,91.0,METAR...,M
"""

MISSING_CSV = """#
station,valid,tmpf
MAF,2025-06-01 12:00,M
MAF,2025-06-01 13:00,M
"""


def _mock_resp(content: str, status: int = 200) -> MagicMock:
    m = MagicMock()
    m.text = content
    m.raise_for_status.return_value = None
    m.status_code = status
    return m


# ── Station validation ────────────────────────────────────────────────────────

class TestStationValidation:
    def test_invalid_station_raises(self):
        with pytest.raises(ValueError, match="Unknown ASOS station"):
            fetch_station("XYZ", date(2025, 1, 1), date(2025, 1, 31))

    def test_all_stations_defined(self):
        assert set(WEST_TX_STATIONS) == {"MAF", "LBB", "SJT"}
        assert len(ALL_STATIONS) == 7

    def test_every_station_has_zone(self):
        for sta, info in STATIONS.items():
            assert "zone" in info, f"{sta} missing zone"
            assert "lat" in info, f"{sta} missing lat"
            assert "lon" in info, f"{sta} missing lon"


# ── _parse_asos_csv ───────────────────────────────────────────────────────────

class TestParseAsosCSV:
    def test_parses_valid_rows(self):
        df = _parse_asos_csv(SAMPLE_CSV, "MAF")
        assert len(df) == 2

    def test_canonical_schema(self):
        df = _parse_asos_csv(SAMPLE_CSV, "MAF")
        assert set(df.columns) == {"interval_start_utc", "station", "zone", "temp_f", "temp_c", "data_tag"}

    def test_celsius_conversion(self):
        df = _parse_asos_csv(SAMPLE_CSV, "MAF")
        row = df.filter(df["temp_f"] == 95.0)
        # 95°F = 35°C
        assert abs(row["temp_c"][0] - 35.0) < 0.01

    def test_real_tag(self):
        df = _parse_asos_csv(SAMPLE_CSV, "MAF")
        assert (df["data_tag"] == "REAL").all()

    def test_skips_missing_observations(self):
        df = _parse_asos_csv(MISSING_CSV, "MAF")
        assert len(df) == 0

    def test_station_and_zone_columns(self):
        df = _parse_asos_csv(SAMPLE_CSV, "MAF")
        assert df["station"][0] == "MAF"
        assert df["zone"][0] == "WEST"

    def test_empty_content_returns_empty(self):
        df = _parse_asos_csv("# just comments\n", "MAF")
        assert len(df) == 0


# ── fetch_station ─────────────────────────────────────────────────────────────

class TestFetchStation:
    def test_returns_canonical_schema(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__.return_value.get.return_value = _mock_resp(SAMPLE_CSV)
            df = fetch_station("MAF", date(2025, 6, 1), date(2025, 6, 1))
        assert "interval_start_utc" in df.columns
        assert "temp_f" in df.columns
        assert "data_tag" in df.columns

    def test_as_of_date_caps_end(self):
        """as_of_date → cap end_date to as_of_date - 1 day."""
        captured_params: dict = {}

        def fake_get(url, params=None, **kwargs):
            captured_params.update(params or {})
            return _mock_resp(SAMPLE_CSV)

        with patch("httpx.Client") as mc:
            mc.return_value.__enter__.return_value.get = fake_get
            fetch_station(
                "MAF",
                date(2025, 1, 1),
                date(2025, 12, 31),
                as_of_date=date(2025, 6, 3),
            )

        # end_date should be as_of_date - 1 = 2025-06-02
        assert captured_params["year2"] == 2025
        assert captured_params["month2"] == 6
        assert captured_params["day2"] == 2

    def test_raises_on_missing_observations(self):
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__.return_value.get.return_value = _mock_resp(MISSING_CSV)
            with pytest.raises(MissingDataError):
                fetch_station("MAF", date(2025, 6, 1), date(2025, 6, 1))

    def test_raises_on_http_error(self):
        import httpx
        with patch("httpx.Client") as mc:
            mc.return_value.__enter__.return_value.get.side_effect = \
                httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock(status_code=503))
            with pytest.raises(ExternalDataError):
                fetch_station("LBB", date(2025, 1, 1), date(2025, 1, 31))


# ── save_to_parquet ───────────────────────────────────────────────────────────

class TestSaveToParquet:
    def _sample_df(self):
        from datetime import datetime, timezone
        import polars as pl
        return pl.DataFrame({
            "interval_start_utc": [
                datetime(2025, 6, 1, 12, tzinfo=timezone.utc),
                datetime(2025, 6, 1, 13, tzinfo=timezone.utc),
            ],
            "station": ["MAF", "MAF"],
            "zone": ["WEST", "WEST"],
            "temp_f": [95.0, 97.0],
            "temp_c": [35.0, 36.1],
            "data_tag": ["REAL", "REAL"],
        })

    def test_creates_per_station_parquet(self, tmp_path):
        df = self._sample_df()
        written = save_to_parquet(df, tmp_path)
        assert "MAF" in written
        assert written["MAF"].exists()

    def test_merge_with_existing(self, tmp_path):
        """Calling save_to_parquet twice merges without duplicates."""
        df = self._sample_df()
        save_to_parquet(df, tmp_path)
        save_to_parquet(df, tmp_path)  # same data again

        import polars as pl
        back = pl.read_parquet(tmp_path / "MAF.parquet")
        assert len(back) == 2  # no duplication

    def test_data_tag_preserved(self, tmp_path):
        df = self._sample_df()
        written = save_to_parquet(df, tmp_path)
        import polars as pl
        back = pl.read_parquet(written["MAF"])
        assert (back["data_tag"] == "REAL").all()
