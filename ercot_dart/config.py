"""
Configuration dataclasses, constants, and schema definitions for the ERCOT DART system.

ERCOT DAM gate closes at 10:00 AM CPT (16:00 UTC). All features must be
constructed strictly from information available before that cutoff to
prevent look-ahead bias.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Market constants
# ---------------------------------------------------------------------------

DAM_GATE_CLOSE: Final[time] = time(10, 0, 0)   # 10:00 AM CPT
SETTLEMENT_INTERVAL_MIN: Final[int] = 15         # RTM settles every 15 min
HOURS_PER_DAY: Final[int] = 24
INTERVALS_PER_HOUR: Final[int] = 4
MAX_OFFER_IDS_PER_NODE: Final[int] = 35          # ERCOT compliance limit
NUM_OFFER_TIERS: Final[int] = 10
PRICE_CAP: Final[float] = 5_000.0               # ERCOT LCAP ($/MWh)
PRICE_FLOOR: Final[float] = -250.0              # ERCOT price floor ($/MWh)
MIN_MW: Final[float] = 0.1                       # Minimum MW threshold

# Fourier seasonality harmonics for the regression model (Phase 2)
FOURIER_ORDER: Final[int] = 4

# Temperature hinge knots (°F) for the spline features (Phase 2)
TEMP_HINGE_KNOTS: Final[tuple[float, ...]] = (45.0, 65.0, 85.0, 95.0)

# ---------------------------------------------------------------------------
# Column schema constants — ERCOT 60-day disclosure CSVs
# ---------------------------------------------------------------------------

# 60d_DAM_EnergyOnlyOffers columns
DAM_OFFER_COLS: Final[list[str]] = [
    "DeliveryDate",
    "HourEnding",
    "SettlementPoint",
    "QSE",
    "ResourceName",
    "MW1", "Price1",
    "MW2", "Price2",
    "MW3", "Price3",
    "MW4", "Price4",
    "MW5", "Price5",
    "MW6", "Price6",
    "MW7", "Price7",
    "MW8", "Price8",
    "MW9", "Price9",
    "MW10", "Price10",
]

# 60d_DAM_EnergyBids columns
DAM_BID_COLS: Final[list[str]] = [
    "DeliveryDate",
    "HourEnding",
    "SettlementPoint",
    "QSE",
    "MW1", "Price1",
    "MW2", "Price2",
    "MW3", "Price3",
    "MW4", "Price4",
    "MW5", "Price5",
    "MW6", "Price6",
    "MW7", "Price7",
    "MW8", "Price8",
    "MW9", "Price9",
    "MW10", "Price10",
]

# DAM Settlement Point Price columns
DAM_SPP_COLS: Final[list[str]] = [
    "DeliveryDate",
    "HourEnding",
    "SettlementPoint",
    "SettlementPointType",
    "SettlementPointPrice",
]

# SCED (15-min Real-Time) Settlement Point Price columns
SCED_SPP_COLS: Final[list[str]] = [
    "SCEDTimestamp",
    "RepeatedHourFlag",
    "SettlementPoint",
    "LMP",
]

# ERCOT Load Forecast columns
LOAD_FORECAST_COLS: Final[list[str]] = [
    "DeliveryDate",
    "HourEnding",
    "SystemTotal",       # ERCOT system-wide load forecast (MW)
    "North",
    "Houston",
    "South",
    "West",
]

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataPaths:
    """Filesystem paths for raw and processed data."""
    raw_dir: Path = field(default_factory=lambda: Path(
        os.getenv("ERCOT_RAW_DIR", "data/raw")
    ))
    processed_dir: Path = field(default_factory=lambda: Path(
        os.getenv("ERCOT_PROCESSED_DIR", "data/processed")
    ))
    cache_dir: Path = field(default_factory=lambda: Path(
        os.getenv("ERCOT_CACHE_DIR", "data/cache")
    ))

    def __post_init__(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ParserConfig:
    """Controls parsing behaviour for ERCOT disclosure ZIP/CSV files."""
    dam_offers_glob: str = "60d_DAM_EnergyOnlyOffers_*.zip"
    dam_bids_glob: str = "60d_DAM_EnergyBids_*.zip"
    dam_spp_glob: str = "60d_DAM_SPP_*.zip"
    sced_spp_glob: str = "60d_SCED_SPP_*.zip"
    load_forecast_glob: str = "60d_LoadForecast_*.zip"
    encoding: str = "utf-8"
    chunk_size: int = 200_000           # rows per CSV chunk for low-memory parsing
    n_jobs: int = -1                    # parallel workers (-1 = all cores)
    cache_parsed: bool = True


@dataclass
class FeatureConfig:
    """Controls which features the FeatureEngineer constructs."""
    target_nodes: list[str] = field(default_factory=lambda: [
        "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON",
        "HB_BUSAVG", "LZ_NORTH", "LZ_SOUTH", "LZ_WEST", "LZ_HOUSTON",
    ])
    # Minimum historical hours required before a feature row is considered valid
    min_history_hours: int = 168        # 1 week
    # Rolling windows (hours) for statistical features
    rolling_windows: tuple[int, ...] = (24, 48, 168)
    # Gas price proxy: Henry Hub or Houston Ship Channel
    gas_price_source: str = "henry_hub"
    fourier_order: int = FOURIER_ORDER
    temp_hinge_knots: tuple[float, ...] = TEMP_HINGE_KNOTS


@dataclass
class ETLConfig:
    """Top-level configuration object passed to the ETL factory."""
    paths: DataPaths = field(default_factory=DataPaths)
    parser: ParserConfig = field(default_factory=ParserConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    timezone: str = "America/Chicago"   # ERCOT operates in CPT/CDT
    validate_schema: bool = True
    verbose: bool = True
