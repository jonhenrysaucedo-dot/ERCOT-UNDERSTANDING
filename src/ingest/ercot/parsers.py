"""ERCOT report parsers for data already in data/raw/uploads/.

Each parser reads one report family, normalises to a canonical Polars DataFrame
with `interval_start_utc` as the time index, and returns it tagged [REAL].

Parsers:
    NativeLoadParser       — Native Load by Zone (NP6-345-CD) .xlsx
    WindSolarParser        — Hourly Wind/Solar Output (NP4-732-CD) .xlsx
    DAMASMCPCParser        — DAM AS MCPC ancillary service prices (NP4-188-CD) .csv
    IntGenByFuelParser     — Interval Generation by Fuel (15-min wide) .xlsx
    PVGRPPForecastParser   — Hourly Solar Forecast PVGRPP (NP4-743-CD) .csv

Walk-forward safety:
    Each parser accepts an `as_of_timestamp` keyword.  Rows with
    `interval_start_utc > as_of_timestamp` are dropped before return.
    Callers MUST pass as_of_timestamp in backtest loops.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import structlog

from src.ingest.ercot.timestamps import (
    polars_convert_he_to_utc,
    parse_combined_ct_timestamp,
    ERCOT_TZ,
)
from src.ingest.exceptions import ERCOTParseError, MissingDataError

logger = structlog.get_logger(__name__)
_DATA_TAG = "REAL"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _apply_as_of_gate(df: pl.DataFrame, as_of: Optional[datetime]) -> pl.DataFrame:
    """Drop rows where interval_start_utc > as_of (walk-forward gate)."""
    if as_of is None:
        return df
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    return df.filter(pl.col("interval_start_utc") <= as_of_utc)


def _glob_files(raw_dir: Path, pattern: str) -> list[Path]:
    """Find files matching a glob pattern; raise if none found."""
    files = sorted(raw_dir.glob(f"**/{pattern}"))
    if not files:
        raise MissingDataError(
            f"No files matching '{pattern}' in {raw_dir}. "
            "Check data/raw/uploads/ — files may need to be downloaded first."
        )
    return files


# ── NativeLoadParser ──────────────────────────────────────────────────────────

class NativeLoadParser:
    """Parse ERCOT Native Load by Zone XLSX files (NP6-345-CD).

    Source columns: Hour Ending, COAST, EAST, FWEST, NORTH, NCENT, SOUTH, SCENT, WEST, ERCOT
    Output: tidy long-format with zone column + load_mw.
    Data tag: [REAL]

    Walk-forward safety: as_of_timestamp gates interval_start_utc.
    """

    # Zones present in the ERCOT Native Load report
    ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST", "ERCOT"]

    def parse(
        self,
        raw_dir: Path,
        as_of_timestamp: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Parse all Native Load XLSX files found under raw_dir.

        Args:
            raw_dir: Directory containing Native_Load_*.xlsx files.
            as_of_timestamp: Walk-forward gate; rows after this are dropped.

        Returns:
            Polars DataFrame:
                interval_start_utc | zone | load_mw | data_tag
        """
        files = _glob_files(raw_dir, "*Native_Load*.xlsx")
        frames: list[pl.DataFrame] = []
        for path in files:
            try:
                df = self._parse_file(path)
                frames.append(df)
                logger.info("native_load_parsed", file=path.name, rows=len(df))
            except Exception as exc:
                raise ERCOTParseError(f"NativeLoad parse failed: {path.name}") from exc

        out = pl.concat(frames).sort(["interval_start_utc", "zone"]).unique(
            subset=["interval_start_utc", "zone"]
        )
        return _apply_as_of_gate(out, as_of_timestamp)

    def _parse_file(self, path: Path) -> pl.DataFrame:
        raw = pd.read_excel(path, dtype=str)

        if "Hour Ending" not in raw.columns:
            raise ERCOTParseError(f"Expected 'Hour Ending' column in {path.name}")

        # Parse timestamps
        utc_times = [
            parse_combined_ct_timestamp(he, "N")  # Native Load has no RHF column
            for he in raw["Hour Ending"]
        ]

        records = []
        for i, ts in enumerate(utc_times):
            row = raw.iloc[i]
            for zone in self.ZONES:
                if zone not in raw.columns:
                    continue
                val = row[zone]
                try:
                    load_mw = float(str(val).replace(",", "")) if pd.notna(val) else None
                except (ValueError, TypeError):
                    load_mw = None
                if load_mw is not None:
                    records.append({
                        "interval_start_utc": ts,
                        "zone": zone,
                        "load_mw": load_mw,
                        "data_tag": _DATA_TAG,
                    })

        return pl.DataFrame(records, schema={
            "interval_start_utc": pl.Datetime("us", "UTC"),
            "zone": pl.Utf8,
            "load_mw": pl.Float64,
            "data_tag": pl.Utf8,
        })


# ── WindSolarParser ───────────────────────────────────────────────────────────

class WindSolarParser:
    """Parse ERCOT Hourly Wind/Solar Output XLSX files (NP4-732-CD style).

    Source columns: Time (Hour-Ending), Date, ERCOT.LOAD, ERCOT.WIND.GEN, ...
    Output: one row per hour with wind_gen_mw, load_mw.
    Data tag: [REAL]

    Walk-forward safety: as_of_timestamp gates interval_start_utc.
    """

    def parse(
        self,
        raw_dir: Path,
        as_of_timestamp: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Parse all ERCOT_*_Hourly_WindSolar_Output.xlsx files under raw_dir."""
        files = _glob_files(raw_dir, "*WindSolar_Output.xlsx")
        frames: list[pl.DataFrame] = []
        seen: set[str] = set()
        for path in files:
            # De-duplicate: ERCOT sometimes publishes the same report with different
            # upload hashes. Use the canonical report filename segment to detect dups.
            canonical = path.name.split("-", 1)[-1]  # strip hash prefix
            if canonical in seen:
                logger.debug("wind_solar_dedup_skip", file=path.name)
                continue
            seen.add(canonical)
            try:
                df = self._parse_file(path)
                frames.append(df)
                logger.info("wind_solar_parsed", file=path.name, rows=len(df))
            except Exception as exc:
                raise ERCOTParseError(f"WindSolar parse failed: {path.name}") from exc

        out = (
            pl.concat(frames)
            .sort("interval_start_utc")
            .unique(subset=["interval_start_utc"])
        )
        return _apply_as_of_gate(out, as_of_timestamp)

    def _parse_file(self, path: Path) -> pl.DataFrame:
        raw = pd.read_excel(path, parse_dates=False)

        # Detect the datetime column (varies between report versions)
        time_col = next(
            (c for c in raw.columns if "hour" in c.lower() or "time" in c.lower()), None
        )
        if time_col is None:
            raise ERCOTParseError(f"Cannot find time column in {path.name}: {raw.columns.tolist()}")

        # Map common column name variants
        wind_col = next((c for c in raw.columns if "wind" in c.lower() and "gen" in c.lower()), None)
        load_col = next((c for c in raw.columns if "load" in c.lower()), None)

        utc_times = []
        for val in raw[time_col]:
            ts_str = str(val).strip()
            try:
                # File format: "2023-01-01 01:00:00" or "01/01/2023 01:00"
                utc_times.append(parse_combined_ct_timestamp(ts_str))
            except ValueError:
                utc_times.append(None)

        records = []
        for i, ts in enumerate(utc_times):
            if ts is None:
                continue
            row = raw.iloc[i]
            rec: dict = {"interval_start_utc": ts, "data_tag": _DATA_TAG}
            if wind_col:
                try:
                    rec["wind_gen_mw"] = float(row[wind_col]) if pd.notna(row[wind_col]) else None
                except (ValueError, TypeError):
                    rec["wind_gen_mw"] = None
            if load_col:
                try:
                    rec["load_mw"] = float(row[load_col]) if pd.notna(row[load_col]) else None
                except (ValueError, TypeError):
                    rec["load_mw"] = None
            records.append(rec)

        schema: dict = {
            "interval_start_utc": pl.Datetime("us", "UTC"),
            "data_tag": pl.Utf8,
        }
        if wind_col:
            schema["wind_gen_mw"] = pl.Float64
        if load_col:
            schema["load_mw"] = pl.Float64

        return pl.DataFrame(records, schema=schema)


# ── DAMASMCPCParser ───────────────────────────────────────────────────────────

class DAMASMCPCParser:
    """Parse DAM Ancillary Service MCPC CSVs (NP4-188-CD).

    Source: DAMASMCPC_YYYY.csv
    Columns: Delivery Date, Hour Ending, Repeated Hour Flag, REGDN, REGUP, RRS, NSPIN, ECRS
    Output: tidy with one row per interval + all AS price columns.
    Data tag: [REAL]

    Walk-forward safety: as_of_timestamp gates interval_start_utc.
    """

    AS_COLS = ["REGDN", "REGUP", "RRS", "NSPIN", "ECRS"]
    # Rename map: strip trailing spaces from ERCOT column names
    _RENAME = {"REGUP ": "REGUP"}

    def parse(
        self,
        raw_dir: Path,
        as_of_timestamp: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Parse all DAMASMCPC_*.csv files under raw_dir."""
        files = _glob_files(raw_dir, "*DAMASMCPC*.csv")
        frames: list[pl.DataFrame] = []
        for path in files:
            try:
                df = self._parse_file(path)
                frames.append(df)
                logger.info("damasmcpc_parsed", file=path.name, rows=len(df))
            except Exception as exc:
                raise ERCOTParseError(f"DAMASMCPC parse failed: {path.name}") from exc

        out = (
            pl.concat(frames)
            .sort("interval_start_utc")
            .unique(subset=["interval_start_utc"])
        )
        return _apply_as_of_gate(out, as_of_timestamp)

    def _parse_file(self, path: Path) -> pl.DataFrame:
        raw = pd.read_csv(path, dtype=str)
        # Normalise column names (strip trailing spaces)
        raw.columns = [c.strip() for c in raw.columns]

        required = {"Delivery Date", "Hour Ending"}
        if not required.issubset(set(raw.columns)):
            raise ERCOTParseError(
                f"DAMASMCPC missing required columns. Have: {raw.columns.tolist()}"
            )

        rhf_present = "Repeated Hour Flag" in raw.columns

        utc_times = [
            parse_combined_ct_timestamp(
                f"{row['Delivery Date']} {row['Hour Ending']}",
                row.get("Repeated Hour Flag", "N") if rhf_present else "N",
            )
            for _, row in raw.iterrows()
        ]

        records = []
        for i, ts in enumerate(utc_times):
            row = raw.iloc[i]
            rec: dict = {"interval_start_utc": ts, "data_tag": _DATA_TAG}
            for col in self.AS_COLS:
                if col in raw.columns:
                    v = row[col]
                    try:
                        rec[f"as_{col.lower()}_usd"] = (
                            float(v) if pd.notna(v) and str(v).strip() != "" else None
                        )
                    except (ValueError, TypeError):
                        rec[f"as_{col.lower()}_usd"] = None
            records.append(rec)

        schema: dict = {"interval_start_utc": pl.Datetime("us", "UTC"), "data_tag": pl.Utf8}
        for col in self.AS_COLS:
            schema[f"as_{col.lower()}_usd"] = pl.Float64

        return pl.DataFrame(records, schema=schema)


# ── IntGenByFuelParser ────────────────────────────────────────────────────────

class IntGenByFuelParser:
    """Parse ERCOT Interval Generation by Fuel XLSX (15-min wide format).

    Source: IntGenbyFuel*.xlsx — monthly tabs (Jan..Dec), wide 15-min columns.
    Output: tidy long-format with fuel, settlement_type, interval_start_utc, gen_mw.
    Data tag: [REAL]

    Walk-forward safety: as_of_timestamp gates interval_start_utc.
    """

    MONTHLY_TABS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    FUEL_ROW_COL = "Fuel"
    DATE_COL = "Date"
    SETTLE_COL = "Settlement Type"
    TOTAL_COL = "Total"

    def parse(
        self,
        raw_dir: Path,
        as_of_timestamp: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Parse all IntGenbyFuel*.xlsx files under raw_dir."""
        files = _glob_files(raw_dir, "*IntGenbyFuel*.xlsx")
        frames: list[pl.DataFrame] = []
        for path in files:
            try:
                df = self._parse_file(path)
                if df is not None:
                    frames.append(df)
                    logger.info("intgenbyfuel_parsed", file=path.name, rows=len(df))
            except Exception as exc:
                raise ERCOTParseError(f"IntGenByFuel parse failed: {path.name}") from exc

        if not frames:
            raise MissingDataError("IntGenByFuel: no usable data found in any file")

        out = (
            pl.concat(frames)
            .sort(["interval_start_utc", "fuel"])
            .unique(subset=["interval_start_utc", "fuel", "settlement_type"])
        )
        return _apply_as_of_gate(out, as_of_timestamp)

    def _parse_file(self, path: Path) -> Optional[pl.DataFrame]:
        xl = pd.ExcelFile(path)
        tabs = [t for t in self.MONTHLY_TABS if t in xl.sheet_names]
        # Also try 'data_Summary_1' for a quick check
        if not tabs:
            # Try data_Summary_1 or similar
            summary_tabs = [t for t in xl.sheet_names if "data_" in t.lower()]
            if summary_tabs:
                tabs = summary_tabs[:1]
        if not tabs:
            logger.warning("intgenbyfuel_no_month_tabs", file=path.name, sheets=xl.sheet_names)
            return None

        frames: list[pl.DataFrame] = []
        for tab in tabs:
            try:
                df = self._parse_tab(xl, tab, path.name)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except Exception as exc:
                logger.warning("intgenbyfuel_tab_error", file=path.name, tab=tab, error=str(exc))

        return pl.concat(frames) if frames else None

    def _parse_tab(self, xl: pd.ExcelFile, tab: str, filename: str) -> Optional[pl.DataFrame]:
        raw = pd.read_excel(xl, sheet_name=tab, header=None)
        if raw.empty or raw.shape[1] < 5:
            return None

        # Row 0 is the header; identify columns
        header = raw.iloc[0].tolist()
        if str(header[0]).strip() != "Date":
            return None  # Not a data tab

        raw.columns = [str(h).strip() for h in header]
        raw = raw.iloc[1:].reset_index(drop=True)

        # Wide 15-min columns after "Total" column
        interval_cols = [c for c in raw.columns if ":" in str(c)]

        records = []
        for _, row in raw.iterrows():
            date_val = row.get("Date")
            fuel_val = str(row.get("Fuel", "")).strip()
            settle_val = str(row.get("Settlement Type", "FINAL")).strip()
            if not fuel_val or fuel_val == "nan":
                continue
            if pd.isna(date_val):
                continue

            try:
                date_dt = pd.to_datetime(date_val).date()
            except Exception:
                continue

            for col in interval_cols:
                # col format: "H:MM" e.g. "0:15", "1:00", "23:45", "0:00" (midnight next day)
                try:
                    h_str, m_str = str(col).split(":")
                    h, m = int(h_str), int(m_str)
                except ValueError:
                    continue

                from datetime import timedelta
                naive_start = datetime(date_dt.year, date_dt.month, date_dt.day, h, m, 0)
                # "0:00" at end of day = midnight next day
                if col == "0:00" and h == 0 and m == 0 and interval_cols.index(col) == len(interval_cols) - 1:
                    naive_start += timedelta(days=1)

                from zoneinfo import ZoneInfo
                from src.ingest.ercot.timestamps import ERCOT_TZ, UTC_TZ
                aware_ct = naive_start.replace(tzinfo=ERCOT_TZ)
                ts_utc = aware_ct.astimezone(UTC_TZ)

                gen_val = row.get(col)
                try:
                    gen_mw = float(gen_val) if pd.notna(gen_val) else None
                except (ValueError, TypeError):
                    gen_mw = None

                if gen_mw is not None:
                    records.append({
                        "interval_start_utc": ts_utc,
                        "fuel": fuel_val,
                        "settlement_type": settle_val,
                        "gen_mw": gen_mw,
                        "data_tag": _DATA_TAG,
                    })

        if not records:
            return None

        return pl.DataFrame(records, schema={
            "interval_start_utc": pl.Datetime("us", "UTC"),
            "fuel": pl.Utf8,
            "settlement_type": pl.Utf8,
            "gen_mw": pl.Float64,
            "data_tag": pl.Utf8,
        })


# ── PVGRPPForecastParser ──────────────────────────────────────────────────────

class PVGRPPForecastParser:
    """Parse ERCOT Hourly Solar Forecast (PVGRPP) CSVs (NP4-743-CD).

    Source: HRLYSTPPFPVGRPPFCSTMODLNP4443.csv (daily rolling forecast files)
    Columns: DeliveryDate, HourEnding, Region, Value, Model, InUseFlag, DSTFlag
    Output: one row per (interval_start_utc, region) with solar_forecast_mw.
    Data tag: [REAL]

    Walk-forward safety: as_of_timestamp gates interval_start_utc.
    """

    def parse(
        self,
        raw_dir: Path,
        as_of_timestamp: Optional[datetime] = None,
        in_use_only: bool = True,
    ) -> pl.DataFrame:
        """Parse all PVGRPP forecast CSVs under raw_dir.

        Args:
            raw_dir: Directory containing HRLYSTPPF*.csv files.
            as_of_timestamp: Walk-forward gate.
            in_use_only: If True, keep only rows where InUseFlag == 'Y'.
        """
        files = _glob_files(raw_dir, "*HRLYSTPPF*PVGRPP*.csv")
        frames: list[pl.DataFrame] = []
        for path in files:
            try:
                df = self._parse_file(path, in_use_only)
                frames.append(df)
                logger.info("pvgrpp_parsed", file=path.name, rows=len(df))
            except Exception as exc:
                raise ERCOTParseError(f"PVGRPP parse failed: {path.name}") from exc

        out = (
            pl.concat(frames)
            .sort(["interval_start_utc", "region"])
            .unique(subset=["interval_start_utc", "region", "model"])
        )
        return _apply_as_of_gate(out, as_of_timestamp)

    def _parse_file(self, path: Path, in_use_only: bool) -> pl.DataFrame:
        raw = pd.read_csv(path, dtype=str)

        required = {"DeliveryDate", "HourEnding", "Region", "Value"}
        if not required.issubset(set(raw.columns)):
            raise ERCOTParseError(
                f"PVGRPP missing columns. Have: {raw.columns.tolist()}"
            )

        if in_use_only and "InUseFlag" in raw.columns:
            raw = raw[raw["InUseFlag"].str.upper().str.strip() == "Y"]

        rhf_present = "DSTFlag" in raw.columns

        utc_times = []
        for _, row in raw.iterrows():
            rhf = "Y" if rhf_present and str(row.get("DSTFlag", "N")).upper() == "Y" else "N"
            try:
                ts = parse_combined_ct_timestamp(
                    f"{row['DeliveryDate']} {int(float(row['HourEnding'])):02d}:00",
                    rhf,
                )
            except (ValueError, TypeError):
                ts = None
            utc_times.append(ts)

        records = []
        for i, ts in enumerate(utc_times):
            if ts is None:
                continue
            row = raw.iloc[i]
            try:
                val = float(row["Value"]) if pd.notna(row["Value"]) else None
            except (ValueError, TypeError):
                val = None
            records.append({
                "interval_start_utc": ts,
                "region": str(row.get("Region", "")).strip(),
                "model": str(row.get("Model", "")).strip(),
                "solar_forecast_mw": val,
                "data_tag": _DATA_TAG,
            })

        return pl.DataFrame(records, schema={
            "interval_start_utc": pl.Datetime("us", "UTC"),
            "region": pl.Utf8,
            "model": pl.Utf8,
            "solar_forecast_mw": pl.Float64,
            "data_tag": pl.Utf8,
        })
