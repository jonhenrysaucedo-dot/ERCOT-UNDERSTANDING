"""Tests for EIA gas price ingest module.

Walk-forward contract tests are the most critical: verify that as_of_date
prevents future data from entering backtest feature matrices.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.ingest.exceptions import ExternalDataError, MissingDataError, StaleDataError
from src.ingest.external import gas_prices


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _eia_response(records: list[dict]) -> dict:
    return {"response": {"data": records, "total": len(records)}}


def _record(period: str, value: float) -> dict:
    return {"period": period, "series": "RNGWHHD", "value": str(value)}


# ── fetch_historical ──────────────────────────────────────────────────────────

class TestFetchHistorical:
    def test_returns_real_tagged_dataframe(self, monkeypatch):
        """Successful API call produces [REAL]-tagged polars DataFrame."""
        monkeypatch.setenv("EIA_API_KEY", "test_key")
        payload = _eia_response([
            _record("2025-01-01", 3.10),
            _record("2025-01-02", 3.20),
        ])
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
            df = gas_prices.fetch_historical(
                date(2025, 1, 1), date(2025, 1, 2),
                as_of_date=date(2025, 1, 3),
            )

        assert df["data_tag"].to_list() == ["REAL", "REAL"]
        assert "henry_hub_usd_per_mmbtu" in df.columns
        assert "hsc_usd_per_mmbtu" in df.columns

    def test_hsc_basis_applied(self, monkeypatch):
        """HSC price equals Henry Hub minus $0.10."""
        monkeypatch.setenv("EIA_API_KEY", "test_key")
        payload = _eia_response([_record("2025-03-01", 2.50)])
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
            df = gas_prices.fetch_historical(
                date(2025, 3, 1), date(2025, 3, 1),
                as_of_date=date(2025, 3, 2),
            )

        hh = df["henry_hub_usd_per_mmbtu"][0]
        hsc = df["hsc_usd_per_mmbtu"][0]
        assert abs(hsc - (hh + gas_prices.HSC_BASIS_USD)) < 1e-9

    def test_as_of_date_caps_end(self, monkeypatch):
        """as_of_date prevents fetching data beyond the walk-forward boundary."""
        monkeypatch.setenv("EIA_API_KEY", "test_key")
        captured_params = {}

        def fake_get(url, params=None, **kwargs):
            captured_params.update(params or {})
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = _eia_response([_record("2025-01-01", 3.00)])
            return mock_resp

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get = fake_get
            gas_prices.fetch_historical(
                date(2025, 1, 1),
                date(2025, 12, 31),          # requested end in the future
                as_of_date=date(2025, 1, 2), # as_of caps it to 2025-01-02
            )

        assert captured_params["end"] == "2025-01-02"

    def test_raises_missing_data_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("EIA_API_KEY", raising=False)
        with pytest.raises(MissingDataError, match="EIA_API_KEY"):
            gas_prices.fetch_historical(date(2025, 1, 1), date(2025, 1, 31))

    def test_raises_external_data_on_http_error(self, monkeypatch):
        import httpx
        monkeypatch.setenv("EIA_API_KEY", "test_key")
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.side_effect = (
                httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock(status_code=404))
            )
            with pytest.raises(ExternalDataError):
                gas_prices.fetch_historical(date(2025, 1, 1), date(2025, 1, 2),
                                            as_of_date=date(2025, 1, 3))

    def test_raises_missing_data_on_empty_rows(self, monkeypatch):
        monkeypatch.setenv("EIA_API_KEY", "test_key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _eia_response([])
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
            with pytest.raises(MissingDataError):
                gas_prices.fetch_historical(date(2025, 1, 1), date(2025, 1, 2),
                                            as_of_date=date(2025, 1, 3))


# ── save_to_parquet ───────────────────────────────────────────────────────────

class TestSaveToParquet:
    def test_partitions_by_year(self, tmp_path):
        """Each year gets its own directory."""
        df = pl.DataFrame({
            "price_date": [date(2024, 12, 31), date(2025, 1, 1)],
            "henry_hub_usd_per_mmbtu": [3.0, 3.1],
            "hsc_usd_per_mmbtu": [2.9, 3.0],
            "data_tag": ["REAL", "REAL"],
        })
        written = gas_prices.save_to_parquet(df, tmp_path)
        assert set(written.keys()) == {2024, 2025}
        assert written[2024].exists()
        assert written[2025].exists()

    def test_parquet_round_trips(self, tmp_path):
        """Written Parquet reads back to the same data."""
        df = pl.DataFrame({
            "price_date": [date(2025, 6, 1)],
            "henry_hub_usd_per_mmbtu": [2.75],
            "hsc_usd_per_mmbtu": [2.65],
            "data_tag": ["REAL"],
        })
        written = gas_prices.save_to_parquet(df, tmp_path)
        back = pl.read_parquet(written[2025])
        assert back["henry_hub_usd_per_mmbtu"][0] == pytest.approx(2.75)

    def test_data_tag_preserved(self, tmp_path):
        df = pl.DataFrame({
            "price_date": [date(2025, 1, 1)],
            "henry_hub_usd_per_mmbtu": [3.0],
            "hsc_usd_per_mmbtu": [2.9],
            "data_tag": ["REAL"],
        })
        written = gas_prices.save_to_parquet(df, tmp_path)
        back = pl.read_parquet(written[2025])
        assert back["data_tag"][0] == "REAL"
