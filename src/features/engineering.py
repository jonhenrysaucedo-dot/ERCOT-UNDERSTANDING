"""Feature engineering for the ERCOT DART trading strategy.

All feature builders in this module accept an `as_of_timestamp` argument
(CLAUDE.md §3 — enforced at the type level). Any function without it cannot
be in src/features/.

Canonical output schema per feature group is documented below each function.
The full feature matrix is assembled by build_feature_matrix().

Compliance tags propagate per CLAUDE.md §4:
    [REAL] ∩ [REAL] → [REAL]
    [REAL] ∩ [NULL] → [NULL]   (never fabricate)

Walk-forward safety contract:
    Every function drops rows where interval_start_utc > as_of_timestamp
    BEFORE any computation. Features are only built from data that was
    available at the DAM submission deadline.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import polars as pl

from src.ingest.exceptions import MissingDataError, WalkForwardViolation

UTC = timezone.utc

# DAM submission deadline: 10:00 CT = 15:00 or 16:00 UTC depending on DST.
# The as_of_timestamp contract enforces this at the call site.
DAM_DEADLINE_HOUR_UTC_CST = 16  # 10:00 CST = 16:00 UTC
DAM_DEADLINE_HOUR_UTC_CDT = 15  # 10:00 CDT = 15:00 UTC


# ── Guard helper ─────────────────────────────────────────────────────────────

def _gate(df: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Drop rows with interval_start_utc > as_of. Raise if as_of is naive."""
    if as_of.tzinfo is None:
        raise WalkForwardViolation(
            f"as_of_timestamp must be timezone-aware. Got naive datetime: {as_of!r}"
        )
    as_of_utc = as_of.astimezone(UTC)
    return df.filter(pl.col("interval_start_utc") <= as_of_utc)


def _require_col(df: pl.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MissingDataError(
            f"{context}: required columns missing from input: {missing}. "
            f"Available: {df.columns}"
        )


# ── M2.1 — DART Spread (target variable) ────────────────────────────────────

def compute_dart_spread(
    dam_spp: pl.DataFrame,
    rtm_spp_15min: pl.DataFrame,
    settlement_point: str,
    as_of_timestamp: datetime,
) -> pl.DataFrame:
    """Compute DART spread = hourly mean RTM SPP - DAM SPP.

    DART is the primary alpha signal and the regression target variable (M5).
    A positive DART means RT > DA — favorable for DEC (buy DA / sell RT).
    A negative DART means RT < DA — favorable for INC (sell DA / buy RT).

    Walk-forward safety:
        Both DAM and RTM inputs are gated to as_of_timestamp before joining.
        RTM is aggregated to hourly arithmetic mean before subtraction.

    Args:
        dam_spp: DataFrame with [interval_start_utc, settlement_point, dam_spp_usd]
        rtm_spp_15min: DataFrame with [interval_start_utc, settlement_point, rtm_spp_usd]
        settlement_point: e.g. "RN_QTUM_SLR" or "HB_WEST"
        as_of_timestamp: Walk-forward gate.

    Returns:
        Polars DataFrame:
            interval_start_utc | settlement_point | dam_spp_usd | rtm_spp_hourly_usd
            | dart_spread_usd | data_tag
    """
    _require_col(dam_spp, ["interval_start_utc", "settlement_point", "dam_spp_usd"], "dam_spp")
    _require_col(rtm_spp_15min, ["interval_start_utc", "settlement_point", "rtm_spp_usd"], "rtm_spp")

    dam = _gate(
        dam_spp.filter(pl.col("settlement_point") == settlement_point),
        as_of_timestamp,
    )
    rtm_raw = _gate(
        rtm_spp_15min.filter(pl.col("settlement_point") == settlement_point),
        as_of_timestamp,
    )

    if dam.is_empty():
        raise MissingDataError(f"No DAM SPP rows for {settlement_point} after gate [{as_of_timestamp}]")
    if rtm_raw.is_empty():
        raise MissingDataError(f"No RTM SPP rows for {settlement_point} after gate [{as_of_timestamp}]")

    # Aggregate RTM 15-min → hourly arithmetic mean
    rtm_hourly = (
        rtm_raw.with_columns(
            pl.col("interval_start_utc").dt.truncate("1h").alias("interval_start_utc")
        )
        .group_by("interval_start_utc")
        .agg(pl.col("rtm_spp_usd").mean().alias("rtm_spp_hourly_usd"))
    )

    joined = dam.join(rtm_hourly, on="interval_start_utc", how="inner")

    return joined.with_columns([
        (pl.col("rtm_spp_hourly_usd") - pl.col("dam_spp_usd")).alias("dart_spread_usd"),
        pl.lit("REAL").alias("data_tag"),
    ]).select([
        "interval_start_utc", "settlement_point",
        "dam_spp_usd", "rtm_spp_hourly_usd", "dart_spread_usd", "data_tag",
    ]).sort("interval_start_utc")


# ── M2.2 — Net Load ──────────────────────────────────────────────────────────

def compute_net_load(
    native_load: pl.DataFrame,
    wind_solar: pl.DataFrame,
    as_of_timestamp: datetime,
) -> pl.DataFrame:
    """Compute net load = ERCOT system load − wind generation − solar generation.

    Net load is a key HMM regime feature and a proxy for thermal dispatch pressure.

    Walk-forward safety:
        Both inputs gated to as_of_timestamp. No RTM data used.

    Args:
        native_load: DataFrame from NativeLoadParser with [interval_start_utc, zone, load_mw]
        wind_solar: DataFrame from WindSolarParser with [interval_start_utc, wind_gen_mw, load_mw]
        as_of_timestamp: Walk-forward gate.

    Returns:
        Polars DataFrame:
            interval_start_utc | ercot_load_mw | wind_gen_mw | net_load_mw | data_tag
    """
    _require_col(native_load, ["interval_start_utc", "zone", "load_mw"], "native_load")
    _require_col(wind_solar, ["interval_start_utc", "wind_gen_mw"], "wind_solar")

    # ERCOT system-wide load
    load = _gate(
        native_load.filter(pl.col("zone") == "ERCOT"),
        as_of_timestamp,
    ).select(["interval_start_utc", "load_mw"]).rename({"load_mw": "ercot_load_mw"})

    wind = _gate(wind_solar, as_of_timestamp).select(["interval_start_utc", "wind_gen_mw"])

    joined = load.join(wind, on="interval_start_utc", how="inner")
    return joined.with_columns([
        (pl.col("ercot_load_mw") - pl.col("wind_gen_mw")).alias("net_load_mw"),
        pl.lit("REAL").alias("data_tag"),
    ]).sort("interval_start_utc")


# ── M2.3 — Thermal Share ─────────────────────────────────────────────────────

def compute_thermal_share(
    fuel_mix: pl.DataFrame,
    as_of_timestamp: datetime,
) -> pl.DataFrame:
    """Compute thermal share = (Coal + Gas + Gas-CC + Nuclear) / Total Generation.

    Thermal share is the primary HMM regime feature:
    - High thermal share → scarcity / high price regime
    - Low thermal share (solar/wind dominant) → potential negative congestion

    Walk-forward safety:
        Fuel mix gated to as_of_timestamp.

    Args:
        fuel_mix: Long-format DataFrame with [interval_start_utc, fuel, gen_mw]
        as_of_timestamp: Walk-forward gate.

    Returns:
        Polars DataFrame:
            interval_start_utc | thermal_mw | total_gen_mw | thermal_share | data_tag
    """
    _require_col(fuel_mix, ["interval_start_utc", "fuel", "gen_mw"], "fuel_mix")

    gated = _gate(fuel_mix, as_of_timestamp)

    thermal_fuels = {"Coal", "Gas", "Gas-CC", "Nuclear", "coal", "gas", "gas-cc", "nuclear"}

    pivoted = (
        gated.with_columns(
            pl.col("fuel").str.to_lowercase().alias("fuel_lower")
        )
        .with_columns(
            pl.when(
                pl.col("fuel_lower").is_in({"coal", "gas", "gas-cc", "nuclear"})
            ).then(pl.col("gen_mw")).otherwise(0.0).alias("thermal_mw_contrib")
        )
        .group_by("interval_start_utc")
        .agg([
            pl.col("thermal_mw_contrib").sum().alias("thermal_mw"),
            pl.col("gen_mw").sum().alias("total_gen_mw"),
        ])
        .with_columns([
            (pl.col("thermal_mw") / (pl.col("total_gen_mw") + 1e-6)).alias("thermal_share"),
            pl.lit("REAL").alias("data_tag"),
        ])
        .sort("interval_start_utc")
    )

    return pivoted.select(["interval_start_utc", "thermal_mw", "total_gen_mw", "thermal_share", "data_tag"])


# ── M2.4 — AS Total Capacity + ECRS Premium ──────────────────────────────────

def compute_as_features(
    dam_as_mcpc: pl.DataFrame,
    as_of_timestamp: datetime,
) -> pl.DataFrame:
    """Compute AS total capacity price and ECRS premium features.

    as_total_capacity: sum of all AS MCPC prices — proxy for RTC+B opportunity cost.
    ecrs_premium: ECRS / RegUp — detects the ECRS-distortion regime (2023–2025).

    Walk-forward safety:
        DAM AS MCPC is a day-ahead price, published before DAM deadline.
        Gated to as_of_timestamp.

    Args:
        dam_as_mcpc: DataFrame from DAMASMCPCParser with [interval_start_utc, as_*_usd cols]
        as_of_timestamp: Walk-forward gate.

    Returns:
        Polars DataFrame:
            interval_start_utc | as_total_capacity | ecrs_premium | data_tag
            plus individual as_*_usd columns passed through
    """
    _require_col(dam_as_mcpc, ["interval_start_utc"], "dam_as_mcpc")

    gated = _gate(dam_as_mcpc, as_of_timestamp)

    as_cols = [c for c in gated.columns if c.startswith("as_") and c.endswith("_usd")]

    if not as_cols:
        raise MissingDataError("dam_as_mcpc has no as_*_usd columns")

    # Sum of all non-null AS prices
    total_expr = sum(pl.col(c).fill_null(0.0) for c in as_cols)

    result = gated.with_columns([
        total_expr.alias("as_total_capacity"),
    ])

    # ECRS premium = ECRS / RegUp (null if either missing)
    if "as_ecrs_usd" in gated.columns and "as_regup_usd" in gated.columns:
        result = result.with_columns([
            (pl.col("as_ecrs_usd") / (pl.col("as_regup_usd") + 1e-6))
            .alias("ecrs_premium"),
        ])
    else:
        result = result.with_columns(pl.lit(None).cast(pl.Float64).alias("ecrs_premium"))

    return result.with_columns(pl.lit("REAL").alias("data_tag")).sort("interval_start_utc")


# ── M2.5 — Temperature Hinge Features ────────────────────────────────────────

def compute_temperature_features(
    weather: pl.DataFrame,
    as_of_timestamp: datetime,
    hot_threshold_f: float = 90.0,
    cold_threshold_f: float = 30.0,
    stations: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Compute piecewise temperature hinge features and degree-day metrics.

    Features per station:
        temp_hinge_hot_{STATION}:  max(0, temp_f − hot_threshold)
        temp_hinge_cold_{STATION}: max(0, cold_threshold − temp_f)
        cdh_{STATION}:  cooling degree-hours (same as temp_hinge_hot)
        hdh_{STATION}:  heating degree-hours (same as temp_hinge_cold)

    Walk-forward safety:
        Weather gated to as_of_timestamp.  ASOS has ~24h latency so features
        at T use observations from T-1 in real-time inference.

    Args:
        weather: DataFrame from asos_weather with [interval_start_utc, station, zone, temp_f]
        as_of_timestamp: Walk-forward gate.
        hot_threshold_f: °F above which cooling demand activates.
        cold_threshold_f: °F below which heating demand activates.
        stations: Stations to pivot. Defaults to all stations in weather DataFrame.

    Returns:
        Polars DataFrame wide-format with one row per interval_start_utc,
        one column set per station.
    """
    _require_col(weather, ["interval_start_utc", "station", "temp_f"], "weather")

    gated = _gate(weather, as_of_timestamp)
    if stations:
        gated = gated.filter(pl.col("station").is_in(stations))

    result = (
        gated.with_columns([
            pl.max_horizontal(pl.col("temp_f") - hot_threshold_f, pl.lit(0.0))
            .alias("hinge_hot"),
            pl.max_horizontal(pl.lit(cold_threshold_f) - pl.col("temp_f"), pl.lit(0.0))
            .alias("hinge_cold"),
        ])
    )

    # Pivot: one column per station
    pivot_hot = result.pivot(
        values="hinge_hot", index="interval_start_utc", on="station",
        aggregate_function="mean",
    )
    pivot_cold = result.pivot(
        values="hinge_cold", index="interval_start_utc", on="station",
        aggregate_function="mean",
    )

    # Rename columns to canonical names
    sta_cols_hot = {c: f"temp_hinge_hot_{c}" for c in pivot_hot.columns if c != "interval_start_utc"}
    sta_cols_cold = {c: f"temp_hinge_cold_{c}" for c in pivot_cold.columns if c != "interval_start_utc"}

    pivot_hot = pivot_hot.rename(sta_cols_hot)
    pivot_cold = pivot_cold.rename(sta_cols_cold)

    # CDH and HDH are aliases for the same hinge values (both included for explainability)
    cdh_aliases = {v: v.replace("temp_hinge_hot_", "cdh_") for v in sta_cols_hot.values()}
    hdh_aliases = {v: v.replace("temp_hinge_cold_", "hdh_") for v in sta_cols_cold.values()}

    joined = pivot_hot.join(pivot_cold, on="interval_start_utc", how="full", coalesce=True)
    # Add CDH/HDH alias columns
    for src, dst in {**cdh_aliases, **hdh_aliases}.items():
        if src in joined.columns:
            joined = joined.with_columns(pl.col(src).alias(dst))

    return joined.with_columns(pl.lit("REAL").alias("data_tag")).sort("interval_start_utc")


# ── M2.6 — Fourier Temporal Features ─────────────────────────────────────────

def compute_temporal_features(
    timestamps: pl.Series,
    as_of_timestamp: datetime,
) -> pl.DataFrame:
    """Compute Fourier time-of-day and day-of-week encoding features.

    Features:
        hour_sin, hour_cos   — k=1,2 harmonics of hour-of-day (24h period)
        hour_sin2, hour_cos2
        dow_sin, dow_cos     — k=1 harmonic of day-of-week (7-day period)

    Walk-forward safety:
        Temporal features are deterministic — no data leakage possible.
        The gate is applied to be consistent with other feature builders.

    Args:
        timestamps: UTC-aware interval_start_utc Series.
        as_of_timestamp: Walk-forward gate.

    Returns:
        Polars DataFrame with temporal features aligned to gated timestamps.
    """
    df = pl.DataFrame({"interval_start_utc": timestamps})

    if as_of_timestamp.tzinfo is None:
        raise WalkForwardViolation("as_of_timestamp must be timezone-aware")
    df = df.filter(pl.col("interval_start_utc") <= as_of_timestamp.astimezone(UTC))

    return df.with_columns([
        pl.col("interval_start_utc").dt.hour().cast(pl.Float64).alias("_h"),
        pl.col("interval_start_utc").dt.weekday().cast(pl.Float64).alias("_dow"),
    ]).with_columns([
        (2 * math.pi * pl.col("_h") / 24.0).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("_h") / 24.0).cos().alias("hour_cos"),
        (4 * math.pi * pl.col("_h") / 24.0).sin().alias("hour_sin2"),
        (4 * math.pi * pl.col("_h") / 24.0).cos().alias("hour_cos2"),
        (2 * math.pi * pl.col("_dow") / 7.0).sin().alias("dow_sin"),
        (2 * math.pi * pl.col("_dow") / 7.0).cos().alias("dow_cos"),
    ]).drop(["_h", "_dow"])


# ── M2.7 — Lagged DART Features ──────────────────────────────────────────────

def compute_lagged_dart_features(
    dart_spread: pl.DataFrame,
    as_of_timestamp: datetime,
    lag_hours: list[int] = (24, 48, 168),
    rolling_windows: list[int] = (24, 168),
) -> pl.DataFrame:
    """Compute lagged and rolling statistics of the DART spread.

    Features:
        dart_lag_{h}h:       DART spread h hours ago
        dart_roll_mean_{w}h: rolling mean over the last w hours
        dart_roll_std_{w}h:  rolling std over the last w hours
        dart_z_score_{w}h:   (dart - roll_mean) / (roll_std + ε)

    Walk-forward safety:
        dart_spread is gated to as_of_timestamp. The lag/rolling windows
        are computed on the gated series only — no future values can appear.

    Args:
        dart_spread: DataFrame from compute_dart_spread() with [interval_start_utc, dart_spread_usd]
        as_of_timestamp: Walk-forward gate.
        lag_hours: Lag offsets in hours (all must be positive integers).
        rolling_windows: Rolling window sizes in hours.

    Returns:
        Polars DataFrame with lagged and rolling features.
    """
    _require_col(dart_spread, ["interval_start_utc", "dart_spread_usd"], "dart_spread")

    gated = _gate(dart_spread, as_of_timestamp).sort("interval_start_utc")

    exprs: list[pl.Expr] = []

    for h in lag_hours:
        exprs.append(pl.col("dart_spread_usd").shift(h).alias(f"dart_lag_{h}h"))

    for w in rolling_windows:
        min_p = max(w // 4, 1)
        roll_mean = pl.col("dart_spread_usd").rolling_mean(window_size=w, min_samples=min_p)
        roll_std = pl.col("dart_spread_usd").rolling_std(window_size=w, min_samples=min_p)
        exprs.append(roll_mean.alias(f"dart_roll_mean_{w}h"))
        exprs.append(roll_std.alias(f"dart_roll_std_{w}h"))
        exprs.append(
            ((pl.col("dart_spread_usd") - roll_mean) / (roll_std + 1e-6))
            .alias(f"dart_z_score_{w}h")
        )

    return (
        gated.with_columns(exprs)
        .select(
            ["interval_start_utc"]
            + [f"dart_lag_{h}h" for h in lag_hours]
            + [col for w in rolling_windows for col in (
                f"dart_roll_mean_{w}h", f"dart_roll_std_{w}h", f"dart_z_score_{w}h"
            )]
        )
        .with_columns(pl.lit("REAL").alias("data_tag"))
    )


# ── M2 — Full Feature Matrix ─────────────────────────────────────────────────

def build_feature_matrix(
    *,
    dam_spp: pl.DataFrame,
    rtm_spp_15min: pl.DataFrame,
    native_load: pl.DataFrame,
    wind_solar: pl.DataFrame,
    fuel_mix: pl.DataFrame,
    dam_as_mcpc: pl.DataFrame,
    weather: pl.DataFrame,
    settlement_point: str,
    as_of_timestamp: datetime,
    hot_threshold_f: float = 90.0,
    cold_threshold_f: float = 30.0,
    weather_stations: Optional[list[str]] = None,
    lag_hours: list[int] = (24, 48, 168),
    rolling_windows: list[int] = (24, 168),
) -> pl.DataFrame:
    """Assemble the complete hourly feature matrix for model training/inference.

    Joins all feature groups on interval_start_utc. Rows where any required
    feature is NULL (due to missing upstream data) are NOT dropped here —
    the compliance tag propagates and the caller (runner/backtest) decides
    whether to skip those hours.

    Walk-forward safety:
        All constituent builders are called with as_of_timestamp.
        The join is on interval_start_utc only — no future data can bleed in.

    Args:
        dam_spp, rtm_spp_15min, native_load, wind_solar, fuel_mix,
        dam_as_mcpc, weather: Parsed DataFrames from ingest layer.
        settlement_point: Target settlement point name.
        as_of_timestamp: Walk-forward gate — passed to every builder.
        hot_threshold_f: Temperature hinge for cooling demand.
        cold_threshold_f: Temperature hinge for heating demand.
        weather_stations: Which ASOS stations to include. None = all.
        lag_hours: Lag offsets for DART lagged features.
        rolling_windows: Rolling window sizes for DART features.

    Returns:
        Wide Polars DataFrame keyed by interval_start_utc with all M2 features.
        Includes a `data_tag` column: "REAL" if all inputs are real, "NULL" if any
        required upstream input was missing for that hour.
    """
    # Build each feature group independently
    dart = compute_dart_spread(dam_spp, rtm_spp_15min, settlement_point, as_of_timestamp)
    net_load = compute_net_load(native_load, wind_solar, as_of_timestamp)
    thermal = compute_thermal_share(fuel_mix, as_of_timestamp)
    as_feat = compute_as_features(dam_as_mcpc, as_of_timestamp)
    temp_feat = compute_temperature_features(weather, as_of_timestamp,
                                              hot_threshold_f, cold_threshold_f,
                                              weather_stations)
    temporal = compute_temporal_features(dart["interval_start_utc"], as_of_timestamp)
    lagged = compute_lagged_dart_features(dart, as_of_timestamp, lag_hours, rolling_windows)

    # Sequential left-joins on interval_start_utc (dart is the spine)
    matrix = dart.drop("data_tag")  # each group contributes its own null tracking
    for feat_df in [net_load, thermal, as_feat, temp_feat, temporal, lagged]:
        drop_cols = ["data_tag", "settlement_point"] if "settlement_point" in feat_df.columns else ["data_tag"]
        drop_cols = [c for c in drop_cols if c in feat_df.columns]
        matrix = matrix.join(
            feat_df.drop(drop_cols),
            on="interval_start_utc",
            how="left",
        )

    # Compliance tag: REAL if no nulls in critical columns, NULL otherwise
    critical_cols = ["dart_spread_usd", "ercot_load_mw", "thermal_share"]
    available_critical = [c for c in critical_cols if c in matrix.columns]
    any_null_expr = pl.lit(False)
    for c in available_critical:
        any_null_expr = any_null_expr | pl.col(c).is_null()

    matrix = matrix.with_columns(
        pl.when(any_null_expr).then(pl.lit("NULL")).otherwise(pl.lit("REAL")).alias("data_tag")
    )

    return matrix.sort("interval_start_utc")
