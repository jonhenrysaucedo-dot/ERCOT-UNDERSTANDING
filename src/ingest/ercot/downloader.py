"""ERCOT public data downloader — no MIS credentials required.

Downloads market settlement data from ERCOT's public API and data portal.
All data fetched here is freely available on ercot.com without login.

Reports covered:
    NP4-190-CD — DAM Settlement Point Prices (hourly, all settlement points)
    NP6-905-CD — RTM Settlement Point Prices (15-min, all settlement points)
    NP4-742-CD — STWPF / WGRPP Wind Forecast (hourly)

Public API base: https://api.ercot.com/api/public-reports/
Report detail page: https://www.ercot.com/mp/data-products/data-product-details?id=<ID>

Walk-forward safety:
    download_report() returns local file paths only — parsing is separate.
    The caller controls what gets fed into the feature matrix.
"""

from __future__ import annotations

import re
import time
import zipfile
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
import polars as pl
import structlog

from src.ingest.ercot.timestamps import parse_combined_ct_timestamp, ERCOT_TZ, UTC_TZ
from src.ingest.exceptions import ERCOTParseError, ExternalDataError, MissingDataError

logger = structlog.get_logger(__name__)

# ── ERCOT public endpoints ─────────────────────────────────────────────────────
ERCOT_API_BASE = "https://api.ercot.com/api/public-reports"
ERCOT_FILE_BASE = "https://www.ercot.com/misdownload/servlets/mirDownload"

# Report IDs
REPORT_IDS = {
    "dam_spp": "np4-190-cd",
    "rtm_spp": "np6-905-cd",
    "wind_forecast": "np4-742-cd",
    "native_load": "np6-345-cd",
    "dam_as_mcpc": "np4-188-cd",
}

MAX_RETRIES = 4
RETRY_DELAYS = [2, 4, 8, 16]  # seconds


# ── Low-level HTTP helpers ─────────────────────────────────────────────────────

def _get(url: str, params: Optional[dict] = None, timeout: float = 60.0) -> httpx.Response:
    """GET with exponential-backoff retry on network / 5xx errors."""
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
            logger.warning("ercot_http_5xx", status=resp.status_code, attempt=attempt, url=url)
        except httpx.RequestError as exc:
            logger.warning("ercot_network_error", error=str(exc), attempt=attempt)
        if attempt < MAX_RETRIES:
            time.sleep(delay)
    raise ExternalDataError(f"ERCOT download failed after {MAX_RETRIES} attempts: {url}")


# ── ERCOT Public API — report file listing ─────────────────────────────────────

def list_report_files(
    report_key: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[dict]:
    """List available files for a report from the ERCOT public API.

    Args:
        report_key: Key from REPORT_IDS (e.g. "dam_spp").
        start_date: Optional filter — files with postDatetime >= start_date.
        end_date: Optional filter — files with postDatetime <= end_date.

    Returns:
        List of dicts with keys: fileName, postDatetime, docId, size, url
    """
    report_id = REPORT_IDS.get(report_key)
    if report_id is None:
        raise ValueError(f"Unknown report key '{report_key}'. Valid: {list(REPORT_IDS)}")

    url = f"{ERCOT_API_BASE}/{report_id}"
    params: dict = {"size": 500, "page": 1}

    all_files: list[dict] = []
    while True:
        resp = _get(url, params=params)
        data = resp.json()

        # ERCOT API response shape: {"data": [...], "meta": {"total": N, ...}}
        items = data.get("data", data.get("files", []))
        if not items:
            break

        for item in items:
            # Normalise field names (API may use camelCase or snake_case)
            post_dt_str = (
                item.get("postDatetime") or item.get("post_datetime") or item.get("date", "")
            )
            doc_id = item.get("docId") or item.get("doc_id") or item.get("id", "")
            filename = item.get("fileName") or item.get("file_name") or item.get("name", "")
            size = item.get("size", 0)

            try:
                post_date = _parse_ercot_date(post_dt_str)
            except ValueError:
                post_date = None

            # Date filters
            if start_date and post_date and post_date < start_date:
                continue
            if end_date and post_date and post_date > end_date:
                continue

            all_files.append({
                "fileName": filename,
                "postDatetime": post_dt_str,
                "postDate": post_date,
                "docId": doc_id,
                "size": size,
                "raw": item,
            })

        # Pagination
        meta = data.get("meta", {})
        total = meta.get("total", len(all_files))
        if params["page"] * params["size"] >= total:
            break
        params["page"] += 1

    logger.info("ercot_file_listing", report=report_key, count=len(all_files))
    return all_files


def download_report_file(
    doc_id: str,
    dest_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """Download a single ERCOT report file by its document ID.

    Files are typically ZIP archives containing one or more CSVs.

    Args:
        doc_id: ERCOT MIS document ID from list_report_files().
        dest_dir: Directory to save the file.
        filename: Override destination filename; auto-detected if None.

    Returns:
        Path to the saved file.
    """
    # Try the ERCOT MIS public download endpoint first
    url = f"{ERCOT_FILE_BASE}?doclookupId={doc_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("ercot_download_start", doc_id=doc_id)
    resp = _get(url, timeout=120.0)

    # Detect filename from Content-Disposition header
    if filename is None:
        cd = resp.headers.get("content-disposition", "")
        m = re.search(r'filename="?([^";\r\n]+)"?', cd)
        filename = m.group(1).strip() if m else f"ercot_{doc_id}.zip"

    dest = dest_dir / filename
    dest.write_bytes(resp.content)
    logger.info("ercot_download_done", file=str(dest), bytes=len(resp.content))
    return dest


# ── High-level: bulk-download a year of DAM SPP ───────────────────────────────

def download_dam_spp(
    years: list[int],
    raw_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Download DAM Settlement Point Price files for the given years.

    Files are saved to raw_dir / dam_spp / filename.

    Walk-forward safety: no timestamps consumed here — parsing is separate.
    """
    dest_dir = raw_dir / "dam_spp"
    dest_dir.mkdir(parents=True, exist_ok=True)

    start = date(min(years), 1, 1)
    end = date(max(years), 12, 31)

    files = list_report_files("dam_spp", start_date=start, end_date=end)
    if not files:
        raise MissingDataError(
            f"ERCOT public API returned 0 DAM SPP files for {years}. "
            "Check connectivity or whether the report is truly public."
        )

    downloaded: list[Path] = []
    for f in files:
        dest = dest_dir / f["fileName"]
        if dest.exists() and not overwrite:
            logger.debug("ercot_skip_existing", file=dest.name)
            downloaded.append(dest)
            continue
        try:
            path = download_report_file(f["docId"], dest_dir, f["fileName"])
            downloaded.append(path)
        except ExternalDataError as exc:
            logger.error("dam_spp_download_failed", file=f["fileName"], error=str(exc))

    logger.info("dam_spp_download_complete", files=len(downloaded), years=years)
    return downloaded


def download_rtm_spp(
    years: list[int],
    raw_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Download RTM Settlement Point Price files (15-min) for the given years."""
    dest_dir = raw_dir / "rtm_spp"
    dest_dir.mkdir(parents=True, exist_ok=True)

    start = date(min(years), 1, 1)
    end = date(max(years), 12, 31)

    files = list_report_files("rtm_spp", start_date=start, end_date=end)
    if not files:
        raise MissingDataError(
            f"ERCOT public API returned 0 RTM SPP files for {years}."
        )

    downloaded: list[Path] = []
    for f in files:
        dest = dest_dir / f["fileName"]
        if dest.exists() and not overwrite:
            downloaded.append(dest)
            continue
        try:
            path = download_report_file(f["docId"], dest_dir, f["fileName"])
            downloaded.append(path)
        except ExternalDataError as exc:
            logger.error("rtm_spp_download_failed", file=f["fileName"], error=str(exc))

    logger.info("rtm_spp_download_complete", files=len(downloaded), years=years)
    return downloaded


def download_wind_forecast(
    years: list[int],
    raw_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Download STWPF / WGRPP Wind Forecast files for the given years."""
    dest_dir = raw_dir / "wind_forecast"
    dest_dir.mkdir(parents=True, exist_ok=True)

    start = date(min(years), 1, 1)
    end = date(max(years), 12, 31)

    files = list_report_files("wind_forecast", start_date=start, end_date=end)
    if not files:
        raise MissingDataError(
            f"ERCOT public API returned 0 Wind Forecast files for {years}."
        )

    downloaded: list[Path] = []
    for f in files:
        dest = dest_dir / f["fileName"]
        if dest.exists() and not overwrite:
            downloaded.append(dest)
            continue
        try:
            path = download_report_file(f["docId"], dest_dir, f["fileName"])
            downloaded.append(path)
        except ExternalDataError as exc:
            logger.error("wind_forecast_download_failed", file=f["fileName"], error=str(exc))

    logger.info("wind_forecast_download_complete", files=len(downloaded), years=years)
    return downloaded


# ── CSV parsers for downloaded reports ────────────────────────────────────────

def parse_dam_spp_file(
    path: Path,
    settlement_points: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Parse a DAM SPP ZIP or CSV into a canonical Polars DataFrame.

    ERCOT DAM SPP format (NP4-190-CD):
        Delivery Date, Hour Ending, Repeated Hour Flag, Settlement Point Name,
        Settlement Point Type, Settlement Point Price

    Args:
        path: Path to a ZIP or CSV file.
        settlement_points: Filter to these settlement point names (e.g. ["HB_NORTH"]).
            If None, all points are returned.

    Returns:
        Polars DataFrame:
            interval_start_utc | settlement_point | spp_type | dam_spp_usd | data_tag
    """
    csv_content = _read_zip_or_csv(path)
    import io
    import pandas as pd
    raw = pd.read_csv(io.StringIO(csv_content), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]

    # Flexible column name detection
    date_col = _find_col(raw, ["Delivery Date", "DeliveryDate", "delivery_date"])
    he_col = _find_col(raw, ["Hour Ending", "HourEnding", "hour_ending"])
    rhf_col = _find_col(raw, ["Repeated Hour Flag", "RepeatedHourFlag"], required=False)
    sp_col = _find_col(raw, ["Settlement Point Name", "SettlementPoint", "settlement_point"])
    spt_col = _find_col(raw, ["Settlement Point Type", "SettlementPointType"], required=False)
    price_col = _find_col(raw, ["Settlement Point Price", "SettlementPointPrice", "price"])

    if settlement_points:
        raw = raw[raw[sp_col].str.strip().isin(settlement_points)]
        if raw.empty:
            logger.warning("dam_spp_no_matching_sp", file=path.name, filter=settlement_points)

    records = []
    for _, row in raw.iterrows():
        rhf = str(row.get(rhf_col, "N")).strip() if rhf_col else "N"
        try:
            ts = parse_combined_ct_timestamp(
                f"{row[date_col].strip()} {row[he_col].strip()}", rhf
            )
        except (ValueError, AttributeError):
            continue

        try:
            price = float(row[price_col]) if row[price_col] and row[price_col] != "nan" else None
        except (ValueError, TypeError):
            price = None

        records.append({
            "interval_start_utc": ts,
            "settlement_point": str(row[sp_col]).strip(),
            "spp_type": str(row[spt_col]).strip() if spt_col and pd.notna(row.get(spt_col)) else None,
            "dam_spp_usd": price,
            "data_tag": "REAL",
        })

    return pl.DataFrame(records, schema={
        "interval_start_utc": pl.Datetime("us", "UTC"),
        "settlement_point": pl.Utf8,
        "spp_type": pl.Utf8,
        "dam_spp_usd": pl.Float64,
        "data_tag": pl.Utf8,
    }).sort(["interval_start_utc", "settlement_point"])


def parse_rtm_spp_file(
    path: Path,
    settlement_points: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Parse an RTM SPP ZIP or CSV into a canonical Polars DataFrame.

    ERCOT RTM SPP format (NP6-905-CD):
        Delivery Date, Delivery Hour, Delivery Interval, Repeated Hour Flag,
        Settlement Point Name, Settlement Point Type, Settlement Point Price

    Each row is a 15-minute interval. We keep native 15-min granularity;
    callers aggregate to hourly as needed.

    Returns:
        Polars DataFrame:
            interval_start_utc | settlement_point | spp_type | rtm_spp_usd | data_tag
    """
    import io
    import pandas as pd

    csv_content = _read_zip_or_csv(path)
    raw = pd.read_csv(io.StringIO(csv_content), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]

    date_col = _find_col(raw, ["Delivery Date", "DeliveryDate"])
    hour_col = _find_col(raw, ["Delivery Hour", "DeliveryHour", "Hour Ending", "HourEnding"])
    intv_col = _find_col(raw, ["Delivery Interval", "DeliveryInterval", "Interval"], required=False)
    rhf_col = _find_col(raw, ["Repeated Hour Flag", "RepeatedHourFlag"], required=False)
    sp_col = _find_col(raw, ["Settlement Point Name", "SettlementPoint"])
    spt_col = _find_col(raw, ["Settlement Point Type", "SettlementPointType"], required=False)
    price_col = _find_col(raw, ["Settlement Point Price", "SettlementPointPrice", "price"])

    if settlement_points:
        raw = raw[raw[sp_col].str.strip().isin(settlement_points)]

    records = []
    for _, row in raw.iterrows():
        rhf = str(row.get(rhf_col, "N")).strip() if rhf_col else "N"
        try:
            # RTM: hour is 1-indexed; interval is 1-4 (each 15 min within the hour)
            hour = int(float(str(row[hour_col]).strip()))
            intv = int(float(str(row[intv_col]).strip())) if intv_col and str(row.get(intv_col, "")).strip() else 1
            minute_offset = (intv - 1) * 15

            # Build combined CT datetime string for parse_combined_ct_timestamp
            # hour-ending → convert to hour-starting before adding minute offset
            ts = parse_combined_ct_timestamp(
                f"{str(row[date_col]).strip()} {hour:02d}:00", rhf
            )
            # Add minute offset for sub-hourly interval
            from datetime import timedelta
            ts = ts + timedelta(minutes=minute_offset)
        except (ValueError, TypeError, AttributeError):
            continue

        try:
            price = float(row[price_col]) if str(row[price_col]).strip() not in ("", "nan") else None
        except (ValueError, TypeError):
            price = None

        records.append({
            "interval_start_utc": ts,
            "settlement_point": str(row[sp_col]).strip(),
            "spp_type": str(row[spt_col]).strip() if spt_col and str(row.get(spt_col, "")).strip() not in ("", "nan") else None,
            "rtm_spp_usd": price,
            "data_tag": "REAL",
        })

    return pl.DataFrame(records, schema={
        "interval_start_utc": pl.Datetime("us", "UTC"),
        "settlement_point": pl.Utf8,
        "spp_type": pl.Utf8,
        "rtm_spp_usd": pl.Float64,
        "data_tag": pl.Utf8,
    }).sort(["interval_start_utc", "settlement_point"])


# ── Parquet persistence ─────────────────────────────────────────────────────────

def save_spp_to_parquet(
    df: pl.DataFrame,
    base_dir: Path,
    price_col: str,
) -> dict[int, Path]:
    """Partition a DAM or RTM SPP DataFrame by year and write to Parquet.

    Walk-forward safety: data is stored as-is; the as_of gate is applied at read time.
    """
    written: dict[int, Path] = {}
    years = df.with_columns(pl.col("interval_start_utc").dt.year().alias("_yr"))["_yr"].unique().to_list()

    for year in sorted(years):
        subset = df.filter(pl.col("interval_start_utc").dt.year() == year)
        p = base_dir / f"year={year}"
        p.mkdir(parents=True, exist_ok=True)
        path = p / "data.parquet"
        subset.write_parquet(path)
        written[year] = path
        logger.info("spp_saved", col=price_col, year=year, rows=len(subset), path=str(path))

    return written


# ── CLI entry points ──────────────────────────────────────────────────────────

def run_download_all(
    years: list[int],
    raw_dir: Path,
    node_name: str,
    overwrite: bool = False,
) -> None:
    """Download DAM SPP, RTM SPP, and Wind Forecast for the given years.

    Args:
        years: e.g. [2023, 2024, 2025]
        raw_dir: Root raw data directory (data/raw/)
        node_name: Target settlement point name (read from config/nodes.yaml)
        overwrite: Re-download files that already exist locally.
    """
    logger.info("ercot_download_all_start", years=years, node=node_name)

    download_dam_spp(years, raw_dir, overwrite)
    download_rtm_spp(years, raw_dir, overwrite)
    download_wind_forecast(years, raw_dir, overwrite)

    logger.info("ercot_download_all_complete", years=years)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_zip_or_csv(path: Path) -> str:
    """Read a ZIP or CSV file and return the CSV text content."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ERCOTParseError(f"No CSVs inside ZIP: {path.name}")
            # If multiple CSVs, concatenate (ERCOT sometimes splits by zone)
            parts = []
            for i, name in enumerate(csv_names):
                content = zf.read(name).decode("utf-8", errors="replace")
                if i > 0:
                    # Strip header row from subsequent files
                    lines = content.splitlines()
                    content = "\n".join(lines[1:])
                parts.append(content)
            return "\n".join(parts)
    elif path.suffix.lower() in (".csv", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ERCOTParseError(f"Unsupported file extension: {path.suffix}")


def _find_col(df, candidates: list[str], required: bool = True) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    import pandas as pd
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    if required:
        raise ERCOTParseError(
            f"Cannot find any of {candidates} in columns: {list(df.columns)}"
        )
    return None


def _parse_ercot_date(s: str) -> Optional[date]:
    """Try to extract a date from an ERCOT postDatetime string."""
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).date()
        except ValueError:
            continue
    return None
