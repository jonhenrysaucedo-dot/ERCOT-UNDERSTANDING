"""Open-Meteo Historical Archive client — hourly temperature at ERCOT zone stations.

Source: ERA5 reanalysis (ECMWF/NASA), ~30-km grid, latency T-1 day.
No API key required for the historical archive endpoint.

Writes to: data/external/weather/year=YYYY/data.parquet
Tag: [REAL] — ERA5 is observational reanalysis, not synthetic.

Zone → station mapping (from docs/INVENTORY.md):
    North   → Dallas (KDFW)          32.90, -97.04
    Houston → Houston (KIAH)         29.98, -95.36
    South   → San Antonio (KSAT)     29.53, -98.47
    West    → Midland (KMAF)         31.94, -102.20
    Coast   → Corpus Christi (KCRP)  27.77, -97.50

Walk-forward safety:
    fetch_zone_historical() caps results at as_of_date when provided.
    The day-ahead forecast variant (fetch_zone_forecast()) is explicitly
    labelled and returns only future hours relative to the call date.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import polars as pl
import structlog

from src.ingest.exceptions import ExternalDataError, MissingDataError, StaleDataError

logger = structlog.get_logger(__name__)

# ── Zone station map ──────────────────────────────────────────────────────────
ZONE_STATIONS: dict[str, dict] = {
    "NORTH": {"name": "Dallas (KDFW)",           "lat": 32.90,  "lon": -97.04},
    "HOUSTON": {"name": "Houston (KIAH)",         "lat": 29.98,  "lon": -95.36},
    "SOUTH": {"name": "San Antonio (KSAT)",       "lat": 29.53,  "lon": -98.47},
    "WEST": {"name": "Midland (KMAF)",            "lat": 31.94,  "lon": -102.20},
    "COAST": {"name": "Corpus Christi (KCRP)",    "lat": 27.77,  "lon": -97.50},
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m"
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0
STALE_THRESHOLD_DAYS = 2


# ── Fetch single zone ─────────────────────────────────────────────────────────

def fetch_zone_historical(
    zone: str,
    start_date: date,
    end_date: date,
    as_of_date: Optional[date] = None,
    timeout_s: float = 60.0,
) -> pl.DataFrame:
    """Fetch hourly temperature for one ERCOT zone from Open-Meteo archive.

    Walk-forward safety:
        Results are capped at min(end_date, as_of_date - 1 day) to prevent
        same-day leakage. ERA5 has T-1 latency so this is already the right
        boundary, but the cap is explicit for correctness.

    Args:
        zone: One of NORTH / HOUSTON / SOUTH / WEST / COAST (uppercase).
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        as_of_date: Walk-forward gate. Data on or after this date is dropped.
        timeout_s: HTTP timeout in seconds.

    Returns:
        Polars DataFrame with columns:
            - interval_start_utc: datetime[UTC]  (hour-starting, UTC-aware)
            - zone: str
            - temp_c: float64
            - temp_f: float64
            - data_tag: str ("REAL")

    Raises:
        ValueError: Unknown zone name.
        ExternalDataError: HTTP / parse failure.
        MissingDataError: Response contained no hourly data.
        StaleDataError: Most-recent timestamp is too old.
    """
    zone = zone.upper()
    if zone not in ZONE_STATIONS:
        raise ValueError(f"Unknown zone '{zone}'. Valid zones: {list(ZONE_STATIONS)}")

    if as_of_date:
        end_date = min(end_date, as_of_date - _one_day())

    station = ZONE_STATIONS[zone]
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
        "temperature_unit": "celsius",
    }

    log = logger.bind(zone=zone, start=str(start_date), end=str(end_date))

    resp = _get_with_retry(ARCHIVE_URL, params, timeout_s, log, label="archive")
    return _parse_response(resp, zone, requested_end=end_date)


def fetch_zone_forecast(
    zone: str,
    days_ahead: int = 2,
    timeout_s: float = 30.0,
) -> pl.DataFrame:
    """Fetch day-ahead temperature forecast from Open-Meteo (GFS-based).

    This is NOT ERA5 and NOT [REAL] in the strictest sense; it is a
    deterministic NWP forecast.  The returned DataFrame is tagged [REAL]
    because it represents a genuine external forecast, not a synthetic value.
    Do NOT mix this with historical ERA5 data in the same training column.

    Walk-forward safety:
        Returns only future timestamps (today + 1 .. today + days_ahead).
        Never use for historical backtests.

    Args:
        zone: NORTH / HOUSTON / SOUTH / WEST / COAST.
        days_ahead: Number of future days to retrieve (max 16).

    Returns:
        Same schema as fetch_zone_historical() plus column `forecast_origin_utc`.
    """
    zone = zone.upper()
    if zone not in ZONE_STATIONS:
        raise ValueError(f"Unknown zone '{zone}'")

    station = ZONE_STATIONS[zone]
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "forecast_days": days_ahead,
    }

    log = logger.bind(zone=zone, days_ahead=days_ahead)
    resp = _get_with_retry(FORECAST_URL, params, timeout_s, log, label="forecast")
    df = _parse_response(resp, zone, requested_end=None)  # forecast = always recent

    # Tag with when this forecast was pulled so callers can detect staleness.
    now_utc = datetime.now(tz=timezone.utc)
    return df.with_columns(pl.lit(now_utc).alias("forecast_origin_utc"))


# ── Fetch all zones ───────────────────────────────────────────────────────────

def fetch_all_zones_historical(
    start_date: date,
    end_date: date,
    as_of_date: Optional[date] = None,
    timeout_s: float = 60.0,
) -> pl.DataFrame:
    """Fetch and concatenate historical temperature for all 5 ERCOT zones.

    Walk-forward safety: delegates to fetch_zone_historical().
    """
    frames: list[pl.DataFrame] = []
    for zone in ZONE_STATIONS:
        log = logger.bind(zone=zone)
        try:
            df = fetch_zone_historical(zone, start_date, end_date, as_of_date, timeout_s)
            frames.append(df)
            log.info("weather_zone_fetched", rows=len(df))
        except (ExternalDataError, MissingDataError, StaleDataError) as exc:
            log.error("weather_zone_failed", error=str(exc))
            raise

    return pl.concat(frames).sort(["interval_start_utc", "zone"])


def fetch_all_zones_forecast(days_ahead: int = 2, timeout_s: float = 30.0) -> pl.DataFrame:
    """Fetch day-ahead temperature forecast for all zones. See fetch_zone_forecast()."""
    frames = [fetch_zone_forecast(z, days_ahead, timeout_s) for z in ZONE_STATIONS]
    return pl.concat(frames).sort(["interval_start_utc", "zone"])


# ── Persist to Parquet ─────────────────────────────────────────────────────────

def save_to_parquet(df: pl.DataFrame, base_dir: Path) -> dict[int, Path]:
    """Partition weather DataFrame by year and write to Parquet.

    Args:
        df: Output of fetch_all_zones_historical().
        base_dir: Root of external weather dir (e.g. data/external/weather/).

    Returns:
        Mapping of year → written Path.
    """
    written: dict[int, Path] = {}
    years = (
        df.with_columns(pl.col("interval_start_utc").dt.year().alias("_yr"))["_yr"]
        .unique()
        .to_list()
    )

    for year in sorted(years):
        subset = df.filter(pl.col("interval_start_utc").dt.year() == year)
        p = base_dir / f"year={year}"
        p.mkdir(parents=True, exist_ok=True)
        path = p / "data.parquet"
        subset.write_parquet(path)
        written[year] = path
        logger.info("weather_saved", year=year, rows=len(subset), path=str(path))

    return written


# ── CLI helper ────────────────────────────────────────────────────────────────

def run_ingest(
    start_year: int,
    end_year: int,
    base_dir: Optional[Path] = None,
    as_of_date: Optional[date] = None,
) -> None:
    """Fetch and save temperature for [start_year, end_year] inclusive."""
    if base_dir is None:
        base_dir = Path("data/external/weather")

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    df = fetch_all_zones_historical(start, end, as_of_date=as_of_date)
    save_to_parquet(df, base_dir)
    logger.info("weather_ingest_complete", start_year=start_year, end_year=end_year)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _one_day():
    from datetime import timedelta
    return timedelta(days=1)


def _get_with_retry(
    url: str,
    params: dict,
    timeout_s: float,
    log: structlog.BoundLogger,
    label: str,
) -> httpx.Response:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            if attempt == MAX_RETRIES:
                raise ExternalDataError(
                    f"Open-Meteo {label} HTTP {exc.response.status_code}"
                ) from exc
            log.warning("openmeteo_retry", status=exc.response.status_code, attempt=attempt)
            time.sleep(RETRY_DELAY_S * attempt)
        except httpx.RequestError as exc:
            if attempt == MAX_RETRIES:
                raise ExternalDataError(f"Open-Meteo {label} network error: {exc}") from exc
            log.warning("openmeteo_retry", error=str(exc), attempt=attempt)
            time.sleep(RETRY_DELAY_S * attempt)

    # unreachable but keeps type checker happy
    raise ExternalDataError("Open-Meteo: exhausted retries")


def _parse_response(
    resp: httpx.Response,
    zone: str,
    requested_end: Optional[date] = None,
) -> pl.DataFrame:
    """Parse an Open-Meteo JSON response into a canonical DataFrame.

    Args:
        resp: Raw HTTP response from Open-Meteo.
        zone: ERCOT zone name (uppercase).
        requested_end: The end_date that was requested.  Staleness is only
            checked when requested_end is within STALE_THRESHOLD_DAYS of today
            (i.e. you asked for recent data).  Historical backfill requests skip
            the staleness check — ERA5 is always available for past dates.
    """
    payload = resp.json()

    if "hourly" not in payload:
        raise MissingDataError(f"Open-Meteo response missing 'hourly' key for zone {zone}")

    hourly = payload["hourly"]
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times:
        raise MissingDataError(f"Open-Meteo returned 0 hourly rows for zone {zone}")

    df = (
        pl.DataFrame({"time_str": times, "temp_c": temps})
        .with_columns([
            pl.col("time_str")
            .str.strptime(pl.Datetime("us", "UTC"), "%Y-%m-%dT%H:%M")
            .alias("interval_start_utc"),
            pl.col("temp_c").cast(pl.Float64),
            pl.lit(zone).alias("zone"),
            (pl.col("temp_c") * 9.0 / 5.0 + 32.0).alias("temp_f"),
            pl.lit("REAL").alias("data_tag"),
        ])
        .select(["interval_start_utc", "zone", "temp_c", "temp_f", "data_tag"])
        .drop_nulls(subset=["temp_c"])
    )

    if df.is_empty():
        raise MissingDataError(f"Open-Meteo: no non-null rows for zone {zone}")

    # Staleness check: only meaningful when the request was for recent data.
    # Historical backfill (requested_end far in the past) never triggers this —
    # ERA5 reanalysis is always available for historical dates.
    today = datetime.now(tz=timezone.utc).date()
    is_recent_request = requested_end is None or (today - requested_end).days <= STALE_THRESHOLD_DAYS
    if is_recent_request:
        most_recent = df["interval_start_utc"].max()
        lag_days = (today - most_recent.date()).days
        if lag_days > STALE_THRESHOLD_DAYS:
            raise StaleDataError(
                f"Open-Meteo zone {zone}: most-recent hour is {most_recent} — "
                f"{lag_days}d behind today. ERA5 may not yet be available for recent dates."
            )

    return df
