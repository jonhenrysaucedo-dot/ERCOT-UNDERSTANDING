"""Iowa State Mesonet ASOS archive client — hourly temperature for ERCOT zones.

Source: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
Free, no authentication required.

West Texas stations for RN_QTUM_SLR (West Zone solar):
    MAF — Midland-Odessa International  (primary — nearest to site)
    LBB — Lubbock Preston Smith International
    SJT — San Angelo Regional

All-zone stations for zone-level thermal features:
    DFW — Dallas-Fort Worth (North Zone)
    HOU — Houston-Hobby    (Houston Zone)
    AUS — Austin-Bergstrom (South Zone)
    SAT — San Antonio Int'l (South Zone)

Writes to: data/raw/weather/<STATION>.parquet
Tag: [REAL] — ASOS is real meteorological observation data.

Walk-forward safety:
    fetch_station() accepts as_of_date and returns data up to that date.
    ASOS data has ~24h latency so real-time inference uses T-1 day.
"""

from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import polars as pl
import structlog

from src.ingest.exceptions import ExternalDataError, MissingDataError, StaleDataError

logger = structlog.get_logger(__name__)
UTC = timezone.utc

ASOS_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]

# Stations from config/nodes.yaml (duplicated here for module self-containment)
STATIONS: dict[str, dict] = {
    "MAF": {"name": "Midland-Odessa International", "lat": 31.94, "lon": -102.20, "zone": "WEST"},
    "LBB": {"name": "Lubbock Preston Smith International", "lat": 33.66, "lon": -101.82, "zone": "WEST"},
    "SJT": {"name": "San Angelo Regional", "lat": 31.36, "lon": -100.50, "zone": "WEST"},
    "DFW": {"name": "Dallas-Fort Worth International", "lat": 32.90, "lon": -97.04, "zone": "NORTH"},
    "HOU": {"name": "Houston-Hobby", "lat": 29.65, "lon": -95.28, "zone": "HOUSTON"},
    "AUS": {"name": "Austin-Bergstrom International", "lat": 30.19, "lon": -97.67, "zone": "SOUTH"},
    "SAT": {"name": "San Antonio International", "lat": 29.53, "lon": -98.47, "zone": "SOUTH"},
}

# West Texas stations used for the primary solar node (RN_QTUM_SLR)
WEST_TX_STATIONS = ["MAF", "LBB", "SJT"]
ALL_STATIONS = list(STATIONS.keys())


def fetch_station(
    station: str,
    start_date: date,
    end_date: date,
    as_of_date: Optional[date] = None,
    timeout_s: float = 60.0,
) -> pl.DataFrame:
    """Fetch hourly temperature observations from Iowa State ASOS for one station.

    Walk-forward safety:
        Results are capped at min(end_date, as_of_date - 1 day).
        ASOS has ~24h latency so yesterday is the freshest available.

    Args:
        station: ICAO station code, e.g. "MAF", "LBB".
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        as_of_date: Walk-forward gate. Observations at or after this date are dropped.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Polars DataFrame:
            interval_start_utc | station | zone | temp_f | temp_c | data_tag

    Raises:
        ValueError: Unknown station code.
        ExternalDataError: HTTP failure after retries.
        MissingDataError: No valid observations returned.
    """
    station = station.upper()
    if station not in STATIONS:
        raise ValueError(f"Unknown ASOS station '{station}'. Valid: {list(STATIONS)}")

    if as_of_date:
        end_date = min(end_date, as_of_date - timedelta(days=1))

    params = {
        "station": station,
        "data": "tmpf",                # temperature in °F
        "year1": start_date.year,
        "month1": start_date.month,
        "day1": start_date.day,
        "year2": end_date.year,
        "month2": end_date.month,
        "day2": end_date.day,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",            # METAR (hourly)
    }

    log = logger.bind(station=station, start=str(start_date), end=str(end_date))

    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            log.info("asos_request", attempt=attempt)
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get(ASOS_BASE_URL, params=params)
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            if attempt == MAX_RETRIES:
                raise ExternalDataError(
                    f"ASOS HTTP {exc.response.status_code} for station {station}"
                ) from exc
            log.warning("asos_retry", status=exc.response.status_code, attempt=attempt)
            time.sleep(delay)
        except httpx.RequestError as exc:
            if attempt == MAX_RETRIES:
                raise ExternalDataError(f"ASOS network error for {station}: {exc}") from exc
            log.warning("asos_retry", error=str(exc), attempt=attempt)
            time.sleep(delay)

    df = _parse_asos_csv(resp.text, station)

    if df.is_empty():
        raise MissingDataError(
            f"ASOS {station}: no valid temperature observations [{start_date} – {end_date}]."
        )

    log.info("asos_fetch_done", rows=len(df))
    return df


def fetch_all_stations(
    start_date: date,
    end_date: date,
    stations: Optional[list[str]] = None,
    as_of_date: Optional[date] = None,
    timeout_s: float = 60.0,
) -> pl.DataFrame:
    """Fetch and concatenate observations from multiple ASOS stations.

    Walk-forward safety: delegates to fetch_station().

    Args:
        stations: List of station codes. Defaults to ALL_STATIONS.
    """
    if stations is None:
        stations = ALL_STATIONS

    frames: list[pl.DataFrame] = []
    for sta in stations:
        try:
            df = fetch_station(sta, start_date, end_date, as_of_date, timeout_s)
            frames.append(df)
        except (ExternalDataError, MissingDataError) as exc:
            logger.error("asos_station_failed", station=sta, error=str(exc))
            raise

    return pl.concat(frames).sort(["interval_start_utc", "station"])


def save_to_parquet(df: pl.DataFrame, base_dir: Path) -> dict[str, Path]:
    """Partition weather DataFrame by station and write to Parquet.

    Each station gets its own file: base_dir/{STATION}.parquet
    (no year partitioning since files are small and queries span multiple years).

    Returns:
        Mapping of station code → written Path.
    """
    written: dict[str, Path] = {}
    base_dir.mkdir(parents=True, exist_ok=True)

    stations = df["station"].unique().to_list()
    for station in sorted(stations):
        subset = df.filter(pl.col("station") == station)
        path = base_dir / f"{station}.parquet"
        # Merge with existing data if present
        if path.exists():
            existing = pl.read_parquet(path)
            subset = pl.concat([existing, subset]).unique(
                subset=["interval_start_utc", "station"]
            ).sort("interval_start_utc")
        subset.write_parquet(path)
        written[station] = path
        logger.info("asos_saved", station=station, rows=len(subset), path=str(path))

    return written


def run_ingest(
    start_year: int,
    end_year: int,
    stations: Optional[list[str]] = None,
    base_dir: Optional[Path] = None,
    as_of_date: Optional[date] = None,
) -> None:
    """Fetch and save ASOS temperature for [start_year, end_year] inclusive."""
    if base_dir is None:
        base_dir = Path("data/raw/weather")
    if stations is None:
        stations = ALL_STATIONS

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    df = fetch_all_stations(start, end, stations=stations, as_of_date=as_of_date)
    save_to_parquet(df, base_dir)
    logger.info("asos_ingest_complete", start_year=start_year, end_year=end_year,
                stations=stations)


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_asos_csv(content: str, station: str) -> pl.DataFrame:
    """Parse the comma-separated ASOS response into a Polars DataFrame.

    ASOS CSV format:
        # Comment lines
        station,valid,tmpf,...
        MAF,2023-01-01 00:00,41.0,...
    """
    lines = [l for l in content.splitlines() if l and not l.startswith("#")]
    if len(lines) < 2:
        return pl.DataFrame()  # no data rows

    header = [h.strip() for h in lines[0].split(",")]
    try:
        ts_idx = header.index("valid")
        tmp_idx = header.index("tmpf")
    except ValueError:
        return pl.DataFrame()

    zone = STATIONS[station]["zone"]
    records = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(ts_idx, tmp_idx):
            continue
        ts_str = parts[ts_idx].strip()
        tmp_str = parts[tmp_idx].strip()

        if not ts_str or tmp_str in ("M", "T", "", "null"):
            continue

        try:
            naive_utc = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
            ts_utc = naive_utc.replace(tzinfo=UTC)
        except ValueError:
            continue

        try:
            temp_f = float(tmp_str)
        except ValueError:
            continue

        temp_c = (temp_f - 32.0) * 5.0 / 9.0
        records.append({
            "interval_start_utc": ts_utc,
            "station": station,
            "zone": zone,
            "temp_f": temp_f,
            "temp_c": temp_c,
            "data_tag": "REAL",
        })

    if not records:
        return pl.DataFrame()

    return pl.DataFrame(records, schema={
        "interval_start_utc": pl.Datetime("us", "UTC"),
        "station": pl.Utf8,
        "zone": pl.Utf8,
        "temp_f": pl.Float64,
        "temp_c": pl.Float64,
        "data_tag": pl.Utf8,
    })
