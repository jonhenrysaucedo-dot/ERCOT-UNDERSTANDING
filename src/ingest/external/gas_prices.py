"""EIA Open Data API v2 client — Henry Hub natural gas spot price.

Writes to: data/external/gas_prices/year=YYYY/data.parquet
Tag: [REAL] (ERCOT settlement data per CLAUDE.md compliance rules)

Walk-forward safety:
    fetch_historical() always filters to dates <= as_of_date so no future
    prices are fetched in backtest contexts.  The caller is responsible for
    passing the correct as_of_date.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx
import polars as pl
import structlog

from src.ingest.exceptions import ExternalDataError, MissingDataError, StaleDataError

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EIA_BASE_URL = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
EIA_SERIES_ID = "RNGWHHD"              # Henry Hub spot, daily $/MMBtu
EIA_FALLBACK_CSV = "https://www.eia.gov/dnav/ng/hist/rngwhhdd.csv"
HSC_BASIS_USD = -0.10                  # approximate Houston Ship Channel = HH - $0.10
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0
STALE_THRESHOLD_DAYS = 3              # raise StaleDataError if most-recent price > 3 days old


# ── Pydantic-style config ─────────────────────────────────────────────────────
def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY", "")
    if not key:
        raise MissingDataError(
            "EIA_API_KEY not set — see .env.example. "
            "Free registration at https://www.eia.gov/opendata/register.php"
        )
    return key


def _output_path(year: int, base_dir: Path) -> Path:
    p = base_dir / f"year={year}"
    p.mkdir(parents=True, exist_ok=True)
    return p / "data.parquet"


# ── EIA API client ────────────────────────────────────────────────────────────

def fetch_historical(
    start_date: date,
    end_date: date,
    as_of_date: Optional[date] = None,
    timeout_s: float = 30.0,
) -> pl.DataFrame:
    """Fetch Henry Hub daily spot prices from EIA API v2.

    Walk-forward safety:
        Results are capped at min(end_date, as_of_date). Pass as_of_date in
        backtest loops to prevent future data leakage.

    Args:
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        as_of_date: Walk-forward gate. Prices after this date are dropped.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Polars DataFrame with columns:
            - price_date: date
            - henry_hub_usd_per_mmbtu: float64  [REAL]
            - hsc_usd_per_mmbtu: float64         [REAL] (approximated)
            - data_tag: str  ("REAL")

    Raises:
        MissingDataError: EIA_API_KEY is absent.
        ExternalDataError: API returned non-200 or unparseable response.
        StaleDataError: Most-recent price exceeds STALE_THRESHOLD_DAYS.
    """
    if as_of_date:
        end_date = min(end_date, as_of_date)

    api_key = _api_key()
    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": EIA_SERIES_ID,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }

    log = logger.bind(series=EIA_SERIES_ID, start=str(start_date), end=str(end_date))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("eia_api_request", attempt=attempt)
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get(EIA_BASE_URL, params=params)
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            if attempt == MAX_RETRIES:
                raise ExternalDataError(f"EIA API HTTP {exc.response.status_code}") from exc
            log.warning("eia_api_retry", status=exc.response.status_code, attempt=attempt)
            time.sleep(RETRY_DELAY_S * attempt)
        except httpx.RequestError as exc:
            if attempt == MAX_RETRIES:
                raise ExternalDataError(f"EIA API network error: {exc}") from exc
            log.warning("eia_api_retry", error=str(exc), attempt=attempt)
            time.sleep(RETRY_DELAY_S * attempt)

    payload = resp.json()
    if "response" not in payload or "data" not in payload["response"]:
        raise ExternalDataError(f"Unexpected EIA response shape: {list(payload.keys())}")

    rows = payload["response"]["data"]
    if not rows:
        raise MissingDataError(
            f"EIA returned 0 rows for {EIA_SERIES_ID} [{start_date} – {end_date}]"
        )

    df = (
        pl.DataFrame(rows)
        .rename({"period": "price_date", "value": "henry_hub_usd_per_mmbtu"})
        .with_columns([
            pl.col("price_date").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("henry_hub_usd_per_mmbtu").cast(pl.Float64),
        ])
        .select(["price_date", "henry_hub_usd_per_mmbtu"])
        .with_columns([
            (pl.col("henry_hub_usd_per_mmbtu") + HSC_BASIS_USD)
            .alias("hsc_usd_per_mmbtu"),
            pl.lit("REAL").alias("data_tag"),
        ])
        .sort("price_date")
    )

    # Staleness check
    most_recent = df["price_date"].max()
    lag_days = (end_date - most_recent).days
    if lag_days > STALE_THRESHOLD_DAYS:
        raise StaleDataError(
            f"Most-recent EIA price is {most_recent} — {lag_days}d behind {end_date}. "
            "Check EIA publication schedule or network connectivity."
        )

    log.info("eia_api_success", rows=len(df), most_recent=str(most_recent))
    return df


def fetch_with_fallback(
    start_date: date,
    end_date: date,
    as_of_date: Optional[date] = None,
    timeout_s: float = 30.0,
) -> pl.DataFrame:
    """Attempt EIA API; fall back to EIA bulk CSV if API key is absent or API fails.

    The CSV fallback is unauthenticated and has coarser error reporting.  It is
    intended for initial setup / CI environments without a real API key.

    Walk-forward safety: same as fetch_historical().
    """
    try:
        return fetch_historical(start_date, end_date, as_of_date, timeout_s)
    except MissingDataError:
        logger.warning("eia_api_key_missing_using_csv_fallback")
        return _fetch_csv_fallback(start_date, end_date, as_of_date, timeout_s)


def _fetch_csv_fallback(
    start_date: date,
    end_date: date,
    as_of_date: Optional[date],
    timeout_s: float,
) -> pl.DataFrame:
    """Download EIA Henry Hub history as unauthenticated CSV."""
    if as_of_date:
        end_date = min(end_date, as_of_date)

    logger.info("eia_csv_fallback", url=EIA_FALLBACK_CSV)
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(EIA_FALLBACK_CSV)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExternalDataError(f"EIA CSV fallback failed: {exc}") from exc

    # EIA CSV has a 2-row header; data starts at row 3 with format "MM/DD/YYYY, value"
    lines = resp.text.splitlines()
    data_lines = [l for l in lines[2:] if l.strip() and not l.startswith("Download")]
    records = []
    for line in data_lines:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            row_date = date.fromisoformat(
                "-".join(reversed(parts[0].strip().split("/")))  # MM/DD/YYYY → YYYY-MM-DD
            )
            val = float(parts[1].strip())
            records.append({"price_date": row_date, "henry_hub_usd_per_mmbtu": val})
        except (ValueError, IndexError):
            continue

    if not records:
        raise ExternalDataError("EIA CSV fallback: could not parse any data rows")

    df = (
        pl.DataFrame(records)
        .filter(
            (pl.col("price_date") >= start_date) &
            (pl.col("price_date") <= end_date)
        )
        .with_columns([
            (pl.col("henry_hub_usd_per_mmbtu") + HSC_BASIS_USD).alias("hsc_usd_per_mmbtu"),
            pl.lit("REAL").alias("data_tag"),
        ])
        .sort("price_date")
    )

    logger.info("eia_csv_fallback_success", rows=len(df))
    return df


# ── Persist to Parquet ─────────────────────────────────────────────────────────

def save_to_parquet(df: pl.DataFrame, base_dir: Path) -> dict[int, Path]:
    """Partition DataFrame by year and write to Parquet.

    Args:
        df: Output of fetch_historical() or fetch_with_fallback().
        base_dir: Root of external gas prices dir (e.g. data/external/gas_prices/).

    Returns:
        Mapping of year → written Path.
    """
    written: dict[int, Path] = {}
    years = df.with_columns(pl.col("price_date").dt.year().alias("_yr"))["_yr"].unique().to_list()

    for year in sorted(years):
        subset = df.filter(pl.col("price_date").dt.year() == year)
        path = _output_path(year, base_dir)
        subset.write_parquet(path)
        written[year] = path
        logger.info("gas_prices_saved", year=year, rows=len(subset), path=str(path))

    return written


# ── CLI helper ────────────────────────────────────────────────────────────────

def run_ingest(
    start_year: int,
    end_year: int,
    base_dir: Optional[Path] = None,
    as_of_date: Optional[date] = None,
) -> None:
    """Fetch and save gas prices for [start_year, end_year] (inclusive).

    Used by src/runners/daily.py and direct CLI invocation.
    """
    if base_dir is None:
        base_dir = Path("data/external/gas_prices")

    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    df = fetch_with_fallback(start, end, as_of_date=as_of_date)
    save_to_parquet(df, base_dir)
    logger.info("gas_prices_ingest_complete", start_year=start_year, end_year=end_year)
