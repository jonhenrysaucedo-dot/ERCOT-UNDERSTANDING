"""Tests for ERCOT file parsers.

All tests use synthetic minimal fixtures (no real file I/O except where noted).
Walk-forward gate tests are the most safety-critical.
"""

from __future__ import annotations

import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.ingest.ercot.parsers import (
    NativeLoadParser,
    WindSolarParser,
    DAMASMCPCParser,
    PVGRPPForecastParser,
)
from src.ingest.ercot.timestamps import (
    hour_ending_to_interval_start_utc,
    parse_combined_ct_timestamp,
)
from src.ingest.exceptions import ERCOTParseError, MissingDataError


UTC = timezone.utc


# ── Timestamp utilities ───────────────────────────────────────────────────────

class TestTimestamps:
    def test_he1_standard_time(self):
        """HE=1 on a non-DST day is 06:00 UTC (CST = UTC-6)."""
        ts = hour_ending_to_interval_start_utc("01/01/2023", "01:00", "N")
        assert ts == datetime(2023, 1, 1, 6, 0, 0, tzinfo=UTC)

    def test_he1_daylight_time(self):
        """HE=1 in summer is 05:00 UTC (CDT = UTC-5)."""
        ts = hour_ending_to_interval_start_utc("07/04/2023", "01:00", "N")
        assert ts == datetime(2023, 7, 4, 5, 0, 0, tzinfo=UTC)

    def test_he24_wraps_to_midnight_next_day(self):
        """HE=24 maps to 00:00 the next day CT."""
        ts = hour_ending_to_interval_start_utc("01/31/2023", "24:00", "N")
        # 00:00 CST Feb 1 = 06:00 UTC Feb 1
        assert ts == datetime(2023, 2, 1, 6, 0, 0, tzinfo=UTC)

    def test_fall_back_first_occurrence(self):
        """First HE=2 on fall-back day is CDT = UTC−5, so 01:00 CDT → 06:00 UTC."""
        ts = hour_ending_to_interval_start_utc("11/05/2023", "02:00", "N")
        assert ts == datetime(2023, 11, 5, 6, 0, 0, tzinfo=UTC)

    def test_fall_back_second_occurrence(self):
        """Second HE=2 (RepeatedHourFlag=Y) is CST = UTC−6, so 01:00 CST → 07:00 UTC."""
        ts = hour_ending_to_interval_start_utc("11/05/2023", "02:00", "Y")
        assert ts == datetime(2023, 11, 5, 7, 0, 0, tzinfo=UTC)

    def test_combined_ct_string(self):
        """parse_combined_ct_timestamp correctly converts HH:MM hour-ending to UTC."""
        ts = parse_combined_ct_timestamp("01/01/2023 01:00", "N")
        assert ts == datetime(2023, 1, 1, 6, 0, 0, tzinfo=UTC)

    def test_integer_hour_ending(self):
        """Integer HourEnding (as from Excel) is handled."""
        ts = hour_ending_to_interval_start_utc("2023-07-04", 13, "N")
        # HE=13 → 12:00 CDT → 17:00 UTC
        assert ts == datetime(2023, 7, 4, 17, 0, 0, tzinfo=UTC)

    def test_native_load_dst_suffix(self):
        """ERCOT Native Load uses '02:00 DST' for CDT fall-back first occurrence."""
        ts = parse_combined_ct_timestamp("11/02/2025 02:00 DST")
        # CDT = UTC-5; HE=2 start = 01:00 CDT = 06:00 UTC
        assert ts == datetime(2025, 11, 2, 6, 0, 0, tzinfo=UTC)

    def test_hour_24_midnight_next_day(self):
        """HE=24 in combined string maps to 00:00 CST next day."""
        ts = parse_combined_ct_timestamp("01/31/2025 24:00")
        # 00:00 CST Feb 1 = 06:00 UTC Feb 1
        assert ts == datetime(2025, 2, 1, 6, 0, 0, tzinfo=UTC)


# ── NativeLoadParser ──────────────────────────────────────────────────────────

class TestNativeLoadParser:
    def _make_xlsx(self, tmp_path: Path) -> Path:
        """Write a minimal Native_Load_*.xlsx fixture."""
        df = pd.DataFrame({
            "Hour Ending": ["01/01/2025 01:00", "01/01/2025 02:00"],
            "COAST": [8500.0, 8400.0],
            "NORTH": [900.0, 910.0],
            "ERCOT": [35000.0, 34500.0],
        })
        path = tmp_path / "Native_Load_2025.xlsx"
        df.to_excel(path, index=False)
        return path

    def test_parses_zones(self, tmp_path):
        self._make_xlsx(tmp_path)
        parser = NativeLoadParser()
        df = parser.parse(tmp_path)
        zones = df["zone"].unique().sort().to_list()
        assert "COAST" in zones
        assert "NORTH" in zones
        assert "ERCOT" in zones

    def test_load_mw_values(self, tmp_path):
        self._make_xlsx(tmp_path)
        parser = NativeLoadParser()
        df = parser.parse(tmp_path)
        coast_rows = df.filter(pl.col("zone") == "COAST") if False else \
                     df.filter(df["zone"] == "COAST")
        # Just verify values are float
        assert all(isinstance(v, float) for v in df["load_mw"].to_list())

    def test_real_tag(self, tmp_path):
        self._make_xlsx(tmp_path)
        df = NativeLoadParser().parse(tmp_path)
        assert (df["data_tag"] == "REAL").all()

    def test_as_of_gate(self, tmp_path):
        """as_of_timestamp excludes future rows."""
        self._make_xlsx(tmp_path)
        parser = NativeLoadParser()
        # Gate: only allow HE=1 (2025-01-01 06:00 UTC); HE=2 is excluded
        gate = datetime(2025, 1, 1, 6, 30, 0, tzinfo=UTC)  # between HE1 and HE2
        df = parser.parse(tmp_path, as_of_timestamp=gate)
        # interval_start for HE=1 = 06:00 UTC (within gate)
        # interval_start for HE=2 = 07:00 UTC (beyond gate)
        assert len(df) == len([z for z in NativeLoadParser.ZONES if z in ["COAST", "NORTH", "ERCOT"]])

    def test_raises_on_missing_files(self, tmp_path):
        with pytest.raises(MissingDataError):
            NativeLoadParser().parse(tmp_path)


# ── DAMASMCPCParser ───────────────────────────────────────────────────────────

class TestDAMASMCPCParser:
    def _make_csv(self, tmp_path: Path) -> Path:
        content = (
            "Delivery Date,Hour Ending,Repeated Hour Flag,REGDN,REGUP ,RRS,NSPIN,ECRS\n"
            "01/01/2025,01:00,N,4.69,1.95,1.45,0.97,\n"
            "01/01/2025,02:00,N,1.49,1.10,1.45,1.00,2.50\n"
        )
        path = tmp_path / "DAMASMCPC_2025.csv"
        path.write_text(content)
        return path

    def test_parses_as_prices(self, tmp_path):
        self._make_csv(tmp_path)
        df = DAMASMCPCParser().parse(tmp_path)
        assert "as_regup_usd" in df.columns
        assert "as_ecrs_usd" in df.columns
        assert len(df) == 2

    def test_real_tag(self, tmp_path):
        self._make_csv(tmp_path)
        df = DAMASMCPCParser().parse(tmp_path)
        assert (df["data_tag"] == "REAL").all()

    def test_null_ecrs_handled(self, tmp_path):
        """ECRS was not introduced until 2023; earlier rows have no value."""
        self._make_csv(tmp_path)
        df = DAMASMCPCParser().parse(tmp_path)
        # First row has no ECRS value
        ecrs = df.sort("interval_start_utc")["as_ecrs_usd"].to_list()
        assert ecrs[0] is None

    def test_as_of_gate(self, tmp_path):
        self._make_csv(tmp_path)
        gate = datetime(2025, 1, 1, 6, 30, 0, tzinfo=UTC)
        df = DAMASMCPCParser().parse(tmp_path, as_of_timestamp=gate)
        assert len(df) == 1  # only HE=1 (06:00 UTC) passes the gate

    def test_raises_on_missing_files(self, tmp_path):
        with pytest.raises(MissingDataError):
            DAMASMCPCParser().parse(tmp_path)


# ── PVGRPPForecastParser ──────────────────────────────────────────────────────

class TestPVGRPPForecastParser:
    def _make_csv(self, tmp_path: Path) -> Path:
        content = (
            "DeliveryDate,HourEnding,Region,Value,Model,InUseFlag,DSTFlag\n"
            "05/18/2026,14,CENTEREAST,4236.6,PVGRPP1,Y,N\n"
            "05/18/2026,14,CENTERWEST,3760.7,PVGRPP1,Y,N\n"
            "05/18/2026,14,CENTEREAST,4100.0,PVGRPP2,N,N\n"  # not in use
        )
        name = "HRLYSTPPFPVGRPPFCSTMODLNP4443.csv"
        path = tmp_path / name
        path.write_text(content)
        return path

    def test_in_use_filter(self, tmp_path):
        self._make_csv(tmp_path)
        df = PVGRPPForecastParser().parse(tmp_path, in_use_only=True)
        # Only InUseFlag=Y rows
        assert len(df) == 2

    def test_no_filter(self, tmp_path):
        self._make_csv(tmp_path)
        df = PVGRPPForecastParser().parse(tmp_path, in_use_only=False)
        assert len(df) == 3

    def test_real_tag(self, tmp_path):
        self._make_csv(tmp_path)
        df = PVGRPPForecastParser().parse(tmp_path)
        assert (df["data_tag"] == "REAL").all()

    def test_region_column(self, tmp_path):
        self._make_csv(tmp_path)
        df = PVGRPPForecastParser().parse(tmp_path)
        regions = df["region"].unique().sort().to_list()
        assert "CENTEREAST" in regions
        assert "CENTERWEST" in regions


# ── Import guard ──────────────────────────────────────────────────────────────
# polars needs to be importable for the filter syntax in the test above
import polars as pl  # noqa: E402 — must be at module level for pytest collection
