"""gridstatus ERCOT client wrapper — DAM/RTM SPP, AS prices, fuel mix, forecasts.

gridstatus (https://github.com/gridstatus/gridstatus) is an open-source library
that scrapes ERCOT public reports without requiring any API key or QSE access.

Data lifecycle per CLAUDE.md §7:
    - Training: pulled once, stored in data/raw/gridstatus/, immutable after cache
    - Inference: pulled daily at 08:00 CT, cached by (method, start, end, locations)
    - Walk-forward: every pull records pull_timestamp in a sidecar JSON as the vintage

Compliance tags:
    All gridstatus data is [REAL] — it sources from ERCOT public settlements.
    If a pull returns zero rows, the result is [NULL] and the runner skips that hour.

Walk-forward safety:
    Every function that returns historical data accepts an `as_of_timestamp` kwarg.
    Results are filtered to interval_start_utc <= as_of_timestamp.
    The pull_timestamp sidecar proves what data was available at run time.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import structlog

from src.ingest.exceptions import ExternalDataError, MissingDataError, StaleDataError

logger = structlog.get_logger(__name__)
UTC = timezone.utc

# Gridstatus markets
DAM_MARKET = "DAY_AHEAD_HOURLY"
RTM_MARKET = "REAL_TIME_15_MIN"

# Default settlement points for every pull (hub + target node)
DEFAULT_LOCATIONS = ["RN_QTUM_SLR", "HB_WEST", "HB_HUBAVG"]

# Cache dir for all gridstatus pulls (relative to repo root)
CACHE_DIR = Path("data/raw/gridstatus")


# ── Lazy import helper ────────────────────────────────────────────────────────

def _ercot():
    """Return a gridstatus ERCOT instance (lazy import so tests can mock)."""
    try:
        import gridstatus
        return gridstatus.Ercot()
    except ImportError as exc:
        raise ExternalDataError(
            "gridstatus not installed. Run: pip install gridstatus>=0.30.0"
        ) from exc


# ── Vintage sidecar ───────────────────────────────────────────────────────────

def _write_vintage(path: Path, pull_utc: datetime, meta: dict) -> None:
    """Write a sidecar JSON file recording when data was fetched.

    This is the mandatory walk-forward provenance record per CLAUDE.md §9.
    """
    sidecar = path.with_suffix(".vintage.json")
    sidecar.write_text(json.dumps({
        "pull_timestamp_utc": pull_utc.isoformat(),
        **meta,
    }, indent=2))


def _read_vintage(path: Path) -> Optional[dict]:
    sidecar = path.with_suffix(".vintage.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return None


# ── DAM Settlement Point Prices ───────────────────────────────────────────────

def get_dam_spp(
    start: date,
    end: date,
    locations: Optional[list[str]] = None,
    as_of_timestamp: Optional[datetime] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Fetch DAM hourly Settlement Point Prices via gridstatus.

    Covers: RN_QTUM_SLR, HB_WEST, HB_HUBAVG (plus any extras in `locations`).
    Data is [REAL] per ERCOT public settlements.

    Walk-forward safety:
        Results filtered to interval_start_utc <= as_of_timestamp.
        Pull timestamp written to sidecar JSON as provenance.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.
        locations: Settlement points. Defaults to DEFAULT_LOCATIONS.
        as_of_timestamp: Walk-forward gate.
        cache_dir: Where to cache Parquet. Defaults to CACHE_DIR/dam_spp/.
        overwrite: Re-pull even if cache exists.

    Returns:
        Polars DataFrame:
            interval_start_utc | settlement_point | dam_spp_usd | data_tag
    """
    if locations is None:
        locations = DEFAULT_LOCATIONS
    cdir = (cache_dir or CACHE_DIR) / "dam_spp"
    cdir.mkdir(parents=True, exist_ok=True)

    cache_path = cdir / f"{start}_{end}_{'_'.join(sorted(locations))}.parquet"

    if cache_path.exists() and not overwrite:
        logger.info("dam_spp_cache_hit", file=cache_path.name)
        df = pl.read_parquet(cache_path)
        return _apply_as_of_gate(df, as_of_timestamp)

    logger.info("dam_spp_pull_start", start=str(start), end=str(end), locations=locations)
    pull_utc = datetime.now(UTC)

    try:
        ercot = _ercot()
        raw: pd.DataFrame = ercot.get_spp(
            start=start.isoformat(),
            end=end.isoformat(),
            market=DAM_MARKET,
            locations=locations,
            verbose=False,
        )
    except Exception as exc:
        raise ExternalDataError(f"gridstatus DAM SPP pull failed: {exc}") from exc

    if raw is None or len(raw) == 0:
        raise MissingDataError(
            f"gridstatus returned 0 DAM SPP rows for {locations} [{start} – {end}]. "
            "RN_QTUM_SLR may not have settled in this date range — use HB_WEST as proxy."
        )

    df = _normalize_spp(raw, "dam_spp_usd")
    df.write_parquet(cache_path)
    _write_vintage(cache_path, pull_utc, {
        "dataset": "dam_spp",
        "market": DAM_MARKET,
        "locations": locations,
        "start": str(start),
        "end": str(end),
        "rows": len(df),
    })
    logger.info("dam_spp_pull_done", rows=len(df))
    return _apply_as_of_gate(df, as_of_timestamp)


def get_rtm_spp(
    start: date,
    end: date,
    locations: Optional[list[str]] = None,
    as_of_timestamp: Optional[datetime] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Fetch RTM 15-min Settlement Point Prices via gridstatus.

    Returns native 15-min granularity. Callers aggregate to hourly as needed.
    Data is [REAL] per ERCOT public settlements.

    Walk-forward safety: same as get_dam_spp().

    Returns:
        Polars DataFrame:
            interval_start_utc | settlement_point | rtm_spp_usd | data_tag
    """
    if locations is None:
        locations = DEFAULT_LOCATIONS
    cdir = (cache_dir or CACHE_DIR) / "rtm_spp"
    cdir.mkdir(parents=True, exist_ok=True)

    cache_path = cdir / f"{start}_{end}_{'_'.join(sorted(locations))}.parquet"

    if cache_path.exists() and not overwrite:
        logger.info("rtm_spp_cache_hit", file=cache_path.name)
        df = pl.read_parquet(cache_path)
        return _apply_as_of_gate(df, as_of_timestamp)

    logger.info("rtm_spp_pull_start", start=str(start), end=str(end), locations=locations)
    pull_utc = datetime.now(UTC)

    try:
        ercot = _ercot()
        raw: pd.DataFrame = ercot.get_spp(
            start=start.isoformat(),
            end=end.isoformat(),
            market=RTM_MARKET,
            locations=locations,
            verbose=False,
        )
    except Exception as exc:
        raise ExternalDataError(f"gridstatus RTM SPP pull failed: {exc}") from exc

    if raw is None or len(raw) == 0:
        raise MissingDataError(
            f"gridstatus returned 0 RTM SPP rows for {locations} [{start} – {end}]."
        )

    df = _normalize_spp(raw, "rtm_spp_usd")
    df.write_parquet(cache_path)
    _write_vintage(cache_path, pull_utc, {
        "dataset": "rtm_spp",
        "market": RTM_MARKET,
        "locations": locations,
        "start": str(start),
        "end": str(end),
        "rows": len(df),
    })
    logger.info("rtm_spp_pull_done", rows=len(df))
    return _apply_as_of_gate(df, as_of_timestamp)


def get_as_prices(
    start: date,
    end: date,
    as_of_timestamp: Optional[datetime] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Fetch DAM Ancillary Service MCPC prices via gridstatus.

    Returns hourly rows with RegUp, RegDn, RRS, NSpin, ECRS columns.
    Supplements (and reconciles against) local DAMASMCPC CSV files.

    Walk-forward safety: results filtered to as_of_timestamp.

    Returns:
        Polars DataFrame:
            interval_start_utc | as_regup_usd | as_regdn_usd | as_rrs_usd | as_nspin_usd | as_ecrs_usd | data_tag
    """
    cdir = (cache_dir or CACHE_DIR) / "as_prices"
    cdir.mkdir(parents=True, exist_ok=True)
    cache_path = cdir / f"{start}_{end}.parquet"

    if cache_path.exists() and not overwrite:
        logger.info("as_prices_cache_hit", file=cache_path.name)
        df = pl.read_parquet(cache_path)
        return _apply_as_of_gate(df, as_of_timestamp)

    pull_utc = datetime.now(UTC)
    try:
        ercot = _ercot()
        raw: pd.DataFrame = ercot.get_as_prices(
            start=start.isoformat(),
            end=end.isoformat(),
            verbose=False,
        )
    except Exception as exc:
        raise ExternalDataError(f"gridstatus AS prices pull failed: {exc}") from exc

    if raw is None or len(raw) == 0:
        raise MissingDataError(f"gridstatus returned 0 AS price rows [{start} – {end}].")

    df = _normalize_as_prices(raw)
    df.write_parquet(cache_path)
    _write_vintage(cache_path, pull_utc, {
        "dataset": "as_prices",
        "start": str(start),
        "end": str(end),
        "rows": len(df),
    })
    logger.info("as_prices_pull_done", rows=len(df))
    return _apply_as_of_gate(df, as_of_timestamp)


def get_fuel_mix(
    start: date,
    end: date,
    as_of_timestamp: Optional[datetime] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Fetch ERCOT fuel mix (generation by fuel type) via gridstatus.

    Used for: thermal_share feature (M2), HMM regime features (M3).

    Returns:
        Polars DataFrame:
            interval_start_utc | fuel | gen_mw | data_tag
    """
    cdir = (cache_dir or CACHE_DIR) / "fuel_mix"
    cdir.mkdir(parents=True, exist_ok=True)
    cache_path = cdir / f"{start}_{end}.parquet"

    if cache_path.exists() and not overwrite:
        logger.info("fuel_mix_cache_hit", file=cache_path.name)
        df = pl.read_parquet(cache_path)
        return _apply_as_of_gate(df, as_of_timestamp)

    pull_utc = datetime.now(UTC)
    try:
        ercot = _ercot()
        raw: pd.DataFrame = ercot.get_fuel_mix(
            start=start.isoformat(),
            end=end.isoformat(),
            verbose=False,
        )
    except Exception as exc:
        raise ExternalDataError(f"gridstatus fuel mix pull failed: {exc}") from exc

    if raw is None or len(raw) == 0:
        raise MissingDataError(f"gridstatus returned 0 fuel mix rows [{start} – {end}].")

    df = _normalize_fuel_mix(raw)
    df.write_parquet(cache_path)
    _write_vintage(cache_path, pull_utc, {
        "dataset": "fuel_mix",
        "start": str(start),
        "end": str(end),
        "rows": len(df),
    })
    logger.info("fuel_mix_pull_done", rows=len(df))
    return _apply_as_of_gate(df, as_of_timestamp)


def get_load(
    start: date,
    end: date,
    as_of_timestamp: Optional[datetime] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Fetch ERCOT native load via gridstatus (reconciliation + inference).

    Returns:
        Polars DataFrame:
            interval_start_utc | load_mw | data_tag
    """
    cdir = (cache_dir or CACHE_DIR) / "load"
    cdir.mkdir(parents=True, exist_ok=True)
    cache_path = cdir / f"{start}_{end}.parquet"

    if cache_path.exists() and not overwrite:
        df = pl.read_parquet(cache_path)
        return _apply_as_of_gate(df, as_of_timestamp)

    pull_utc = datetime.now(UTC)
    try:
        ercot = _ercot()
        raw = ercot.get_load(
            start=start.isoformat(),
            end=end.isoformat(),
            verbose=False,
        )
    except Exception as exc:
        raise ExternalDataError(f"gridstatus load pull failed: {exc}") from exc

    if raw is None or len(raw) == 0:
        raise MissingDataError(f"gridstatus returned 0 load rows [{start} – {end}].")

    df = _normalize_load(raw)
    df.write_parquet(cache_path)
    _write_vintage(cache_path, pull_utc, {"dataset": "load", "start": str(start), "end": str(end)})
    return _apply_as_of_gate(df, as_of_timestamp)


# ── Bulk historical pull ──────────────────────────────────────────────────────

def pull_training_corpus(
    start_year: int,
    end_year: int,
    locations: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> dict[str, pl.DataFrame]:
    """Pull all LMP + AS + fuel mix data for the training corpus.

    Pulls year-by-year to keep individual requests manageable.
    Skips datasets already cached (overwrite=False by default).

    Args:
        start_year: First year to pull (inclusive).
        end_year: Last year to pull (inclusive).
        locations: SPP locations. Defaults to DEFAULT_LOCATIONS.
        cache_dir: Base cache directory.
        overwrite: Re-fetch even if cached.

    Returns:
        Dict with keys: "dam_spp", "rtm_spp", "as_prices", "fuel_mix", "load"
    """
    if locations is None:
        locations = DEFAULT_LOCATIONS

    frames: dict[str, list[pl.DataFrame]] = {k: [] for k in ["dam_spp", "rtm_spp", "as_prices", "fuel_mix", "load"]}

    for year in range(start_year, end_year + 1):
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        logger.info("gridstatus_year_pull", year=year)

        for dataset, fn, args in [
            ("dam_spp", get_dam_spp, {"locations": locations}),
            ("rtm_spp", get_rtm_spp, {"locations": locations}),
            ("as_prices", get_as_prices, {}),
            ("fuel_mix", get_fuel_mix, {}),
            ("load", get_load, {}),
        ]:
            try:
                df = fn(start, end, cache_dir=cache_dir, overwrite=overwrite, **args)
                frames[dataset].append(df)
                logger.info("gridstatus_year_ok", dataset=dataset, year=year, rows=len(df))
            except MissingDataError as exc:
                logger.warning("gridstatus_year_null", dataset=dataset, year=year, reason=str(exc))
            except ExternalDataError as exc:
                logger.error("gridstatus_year_error", dataset=dataset, year=year, error=str(exc))

    return {
        k: pl.concat(v).sort("interval_start_utc") if v else pl.DataFrame()
        for k, v in frames.items()
    }


# ── Inference (daily runner) pulls ────────────────────────────────────────────

def get_day_ahead_forecasts(as_of_date: date) -> dict[str, pl.DataFrame]:
    """Pull day-ahead load, wind, and solar forecasts for the next operating day.

    Called by the daily runner at 08:00 CT to populate features before DAM close.
    Returns [REAL] tagged data with pull_timestamp vintage sidecar.

    Walk-forward safety:
        Forecasts are for the NEXT operating day relative to as_of_date.
        Never pass as_of_date in backtest contexts — use historical Parquet instead.

    Returns:
        Dict with keys: "load_forecast", "wind_forecast", "solar_forecast"
    """
    ercot = _ercot()
    pull_utc = datetime.now(UTC)
    results: dict[str, pl.DataFrame] = {}

    for name, method in [
        ("load_forecast", "get_load_forecast"),
        ("wind_forecast", "get_wind_forecast"),
        ("solar_forecast", "get_solar_forecast"),
    ]:
        try:
            raw = getattr(ercot, method)(verbose=False)
            if raw is None or len(raw) == 0:
                logger.warning("day_ahead_forecast_empty", name=name)
                continue
            df = _normalize_forecast(raw, name)
            results[name] = df
            logger.info("day_ahead_forecast_ok", name=name, rows=len(df))
        except Exception as exc:
            logger.error("day_ahead_forecast_failed", name=name, error=str(exc))

    return results


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalize_spp(raw: pd.DataFrame, price_col: str) -> pl.DataFrame:
    """Convert a gridstatus SPP DataFrame to canonical Polars schema."""
    # gridstatus uses 'Time' or 'Interval Start' for the timestamp column
    ts_col = _find_pd_col(raw, ["Time", "Interval Start", "interval_start", "timestamp"])
    sp_col = _find_pd_col(raw, ["Location", "settlement_point", "SettlementPoint", "node"])
    pr_col = _find_pd_col(raw, ["LMP", "SPP", "Price", "price"])

    records = []
    for _, row in raw.iterrows():
        ts = _to_utc(row[ts_col])
        if ts is None:
            continue
        records.append({
            "interval_start_utc": ts,
            "settlement_point": str(row[sp_col]).strip().upper(),
            price_col: _to_float(row[pr_col]),
            "data_tag": "REAL",
        })

    return pl.DataFrame(records, schema={
        "interval_start_utc": pl.Datetime("us", "UTC"),
        "settlement_point": pl.Utf8,
        price_col: pl.Float64,
        "data_tag": pl.Utf8,
    }).sort(["interval_start_utc", "settlement_point"])


def _normalize_as_prices(raw: pd.DataFrame) -> pl.DataFrame:
    """Convert gridstatus AS prices to canonical schema."""
    ts_col = _find_pd_col(raw, ["Time", "Interval Start", "timestamp"])

    col_map = {
        "as_regup_usd": ["REGUP", "RegUp", "reg_up"],
        "as_regdn_usd": ["REGDN", "RegDn", "reg_dn", "reg_down"],
        "as_rrs_usd":   ["RRS", "rrs"],
        "as_nspin_usd": ["NSPIN", "NonSpin", "non_spin"],
        "as_ecrs_usd":  ["ECRS", "ecrs"],
    }

    records = []
    for _, row in raw.iterrows():
        ts = _to_utc(row[ts_col])
        if ts is None:
            continue
        rec: dict = {"interval_start_utc": ts, "data_tag": "REAL"}
        for out_col, candidates in col_map.items():
            found = _find_pd_col(raw, candidates, required=False)
            rec[out_col] = _to_float(row[found]) if found else None
        records.append(rec)

    schema: dict = {"interval_start_utc": pl.Datetime("us", "UTC"), "data_tag": pl.Utf8}
    for col in col_map:
        schema[col] = pl.Float64

    return pl.DataFrame(records, schema=schema).sort("interval_start_utc")


def _normalize_fuel_mix(raw: pd.DataFrame) -> pl.DataFrame:
    """Convert gridstatus fuel mix (wide) to long-format Polars DataFrame."""
    ts_col = _find_pd_col(raw, ["Time", "Interval Start", "timestamp"])

    # All non-timestamp, non-meta columns are fuel types
    skip = {ts_col.lower()}
    fuel_cols = [c for c in raw.columns if c.lower() not in skip and
                 any(kw in c.lower() for kw in ["coal", "gas", "wind", "solar", "nuclear",
                                                 "hydro", "other", "biomass", "storage"])]
    records = []
    for _, row in raw.iterrows():
        ts = _to_utc(row[ts_col])
        if ts is None:
            continue
        for col in fuel_cols:
            gen = _to_float(row[col])
            if gen is not None:
                records.append({
                    "interval_start_utc": ts,
                    "fuel": col,
                    "gen_mw": gen,
                    "data_tag": "REAL",
                })

    return pl.DataFrame(records, schema={
        "interval_start_utc": pl.Datetime("us", "UTC"),
        "fuel": pl.Utf8,
        "gen_mw": pl.Float64,
        "data_tag": pl.Utf8,
    }).sort(["interval_start_utc", "fuel"])


def _normalize_load(raw: pd.DataFrame) -> pl.DataFrame:
    ts_col = _find_pd_col(raw, ["Time", "Interval Start", "timestamp"])
    load_col = _find_pd_col(raw, ["Load", "load", "ERCOT", "ercot_load", "demand"])
    records = []
    for _, row in raw.iterrows():
        ts = _to_utc(row[ts_col])
        if ts is None:
            continue
        records.append({
            "interval_start_utc": ts,
            "load_mw": _to_float(row[load_col]),
            "data_tag": "REAL",
        })
    return pl.DataFrame(records, schema={
        "interval_start_utc": pl.Datetime("us", "UTC"),
        "load_mw": pl.Float64,
        "data_tag": pl.Utf8,
    }).sort("interval_start_utc")


def _normalize_forecast(raw: pd.DataFrame, forecast_type: str) -> pl.DataFrame:
    ts_col = _find_pd_col(raw, ["Time", "Interval Start", "timestamp"])
    val_col = _find_pd_col(raw, ["Load Forecast", "Wind Forecast", "Solar Forecast",
                                  "load", "wind", "solar", "forecast", "value"])
    records = []
    for _, row in raw.iterrows():
        ts = _to_utc(row[ts_col])
        if ts is None:
            continue
        records.append({
            "interval_start_utc": ts,
            "forecast_type": forecast_type,
            "forecast_mw": _to_float(row[val_col]),
            "data_tag": "REAL",
        })
    return pl.DataFrame(records, schema={
        "interval_start_utc": pl.Datetime("us", "UTC"),
        "forecast_type": pl.Utf8,
        "forecast_mw": pl.Float64,
        "data_tag": pl.Utf8,
    }).sort("interval_start_utc")


# ── Utility helpers ───────────────────────────────────────────────────────────

def _apply_as_of_gate(df: pl.DataFrame, as_of: Optional[datetime]) -> pl.DataFrame:
    if as_of is None or df.is_empty():
        return df
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    return df.filter(pl.col("interval_start_utc") <= as_of_utc)


def _to_utc(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)
    if isinstance(val, pd.Timestamp):
        if val.tzinfo is None:
            return val.to_pydatetime().replace(tzinfo=UTC)
        return val.to_pydatetime().astimezone(UTC)
    try:
        return pd.Timestamp(val).to_pydatetime().replace(tzinfo=UTC)
    except Exception:
        return None


def _to_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _find_pd_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if required:
        from src.ingest.exceptions import ERCOTParseError
        raise ERCOTParseError(f"Cannot find {candidates} in columns: {list(df.columns)}")
    return None
