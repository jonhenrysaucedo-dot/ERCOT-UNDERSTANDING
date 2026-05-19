"""DST-aware ERCOT CT timestamp → UTC conversion utilities.

ERCOT publishes timestamps in two formats:
  1. HourEnding notation: Delivery Date + Hour Ending (01:00 .. 24:00) + Repeated Hour Flag
  2. Combined datetime: "MM/DD/YYYY HH:MM" (also hour-ending, Central Time)

This module converts both to `interval_start_utc` (UTC-aware, hour-starting).
See src/ingest/timestamps.md for the DST convention.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl

ERCOT_TZ = ZoneInfo("America/Chicago")
UTC_TZ = timezone.utc


def hour_ending_to_interval_start_utc(
    delivery_date: str,
    hour_ending: str,
    repeated_hour_flag: str = "N",
) -> datetime:
    """Convert an ERCOT HourEnding row to a UTC interval-start datetime.

    Args:
        delivery_date: "MM/DD/YYYY" or "YYYY-MM-DD"
        hour_ending: "01:00" .. "24:00" (or int 1..24)
        repeated_hour_flag: "Y" for the second occurrence of the fall-back hour.

    Returns:
        timezone-aware UTC datetime representing the start of the interval.

    Raises:
        ValueError: If the combination is ambiguous and cannot be resolved.
    """
    # Normalise hour_ending
    if isinstance(hour_ending, (int, float)):
        he = int(hour_ending)
    else:
        he = int(str(hour_ending).split(":")[0])

    # Parse delivery_date
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            d = datetime.strptime(str(delivery_date).strip(), fmt).date()
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse delivery_date: {delivery_date!r}")

    # HourEnding 24 wraps to midnight of next day (00:00 hour-starting next day)
    if he == 24:
        naive_start = datetime(d.year, d.month, d.day, 0, 0, 0) + timedelta(days=1)
    else:
        naive_start = datetime(d.year, d.month, d.day, he - 1, 0, 0)

    # DST disambiguation
    try:
        aware_ct = naive_start.replace(tzinfo=ERCOT_TZ)
        # Validate: ZoneInfo raises no error but fold can disambiguate
        if repeated_hour_flag == "Y":
            # Fall-back second occurrence → standard time (fold=1)
            aware_ct = naive_start.replace(tzinfo=ERCOT_TZ, fold=1)
        return aware_ct.astimezone(UTC_TZ)
    except Exception as exc:
        raise ValueError(
            f"Cannot convert {delivery_date} HE={he} RHF={repeated_hour_flag}: {exc}"
        ) from exc


def parse_combined_ct_timestamp(ts_str: str, repeated_hour_flag: str = "N") -> datetime:
    """Parse a combined "MM/DD/YYYY HH:MM" hour-ending CT string to UTC.

    Handles ERCOT's "24:00" notation (hour 24 = midnight of next day) which
    Python's strptime cannot parse natively.

    Args:
        ts_str: e.g. "01/01/2023 01:00" or "01/31/2023 24:00"
        repeated_hour_flag: "Y" for fall-back duplicate hour.

    Returns:
        UTC-aware datetime at the interval start.
    """
    ts_str = ts_str.strip()

    # Handle ERCOT hour-24: "MM/DD/YYYY 24:00" → midnight next day
    if " 24:00" in ts_str or "T24:00" in ts_str:
        date_part = ts_str.split()[0] if " " in ts_str else ts_str.split("T")[0]
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                d = datetime.strptime(date_part, fmt).date()
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse date in hour-24 string: {ts_str!r}")
        # Hour-ending 24 = start of next day at midnight (hour-starting 00:00)
        naive_start = datetime(d.year, d.month, d.day, 0, 0, 0) + timedelta(days=1)
        fold = 1 if repeated_hour_flag == "Y" else 0
        aware_ct = naive_start.replace(tzinfo=ERCOT_TZ, fold=fold)
        return aware_ct.astimezone(UTC_TZ)

    # Handle ERCOT Native Load DST suffix: "11/02/2025 02:00 DST"
    # " DST" = CDT (daylight, first occurrence); no suffix for ambiguous hour = CST (second)
    dst_override = None
    if ts_str.upper().endswith(" DST"):
        ts_str = ts_str[:-4].strip()
        dst_override = False  # CDT = fold=0
    elif ts_str.upper().endswith(" CST"):
        ts_str = ts_str[:-4].strip()
        dst_override = True   # CST = fold=1

    for fmt in ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            naive_he = datetime.strptime(ts_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse timestamp: {ts_str!r}")

    # hour-ending → hour-starting: subtract 1 hour
    naive_start = naive_he - timedelta(hours=1)

    # Resolve DST fold: explicit suffix overrides RepeatedHourFlag
    if dst_override is not None:
        fold = 1 if dst_override else 0  # True = CST = fold=1; False = CDT = fold=0
    else:
        fold = 1 if repeated_hour_flag == "Y" else 0
    aware_ct = naive_start.replace(tzinfo=ERCOT_TZ, fold=fold)
    return aware_ct.astimezone(UTC_TZ)


def polars_convert_he_to_utc(
    df: pl.DataFrame,
    date_col: str = "Delivery Date",
    he_col: str = "Hour Ending",
    rhf_col: str | None = "Repeated Hour Flag",
) -> pl.DataFrame:
    """Add `interval_start_utc` column to a Polars DataFrame with ERCOT HourEnding cols.

    Uses Python-level loop (slow but correct for DST edge cases).
    For non-DST dates this is equivalent to vectorised arithmetic.
    Returns the DataFrame with `interval_start_utc` prepended.
    """
    rhf_values = df[rhf_col].to_list() if rhf_col and rhf_col in df.columns else ["N"] * len(df)

    utc_times = [
        hour_ending_to_interval_start_utc(d, he, rhf)
        for d, he, rhf in zip(
            df[date_col].to_list(),
            df[he_col].to_list(),
            rhf_values,
        )
    ]

    return df.with_columns(
        pl.Series("interval_start_utc", utc_times, dtype=pl.Datetime("us", "UTC"))
    )
