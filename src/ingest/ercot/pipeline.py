"""ERCOT ingest pipeline — parse all raw files → Parquet store.

Orchestrates:
    1. Parse existing uploads with the 5 file parsers
    2. Persist each dataset to data/processed/<dataset>/year=YYYY/data.parquet
    3. Log compliance tags and row counts to structlog

Walk-forward safety:
    All parsers accept an `as_of_timestamp`. The pipeline runner passes it
    through to every parser.  Callers (backtest, daily runner) must set it.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import polars as pl
import structlog

from src.ingest.ercot.parsers import (
    NativeLoadParser,
    WindSolarParser,
    DAMASMCPCParser,
    IntGenByFuelParser,
    PVGRPPForecastParser,
)
from src.ingest.exceptions import MissingDataError

logger = structlog.get_logger(__name__)


class ERCOTParsedStore:
    """Container for all parsed ERCOT DataFrames."""

    def __init__(
        self,
        native_load: Optional[pl.DataFrame] = None,
        wind_solar: Optional[pl.DataFrame] = None,
        dam_as_mcpc: Optional[pl.DataFrame] = None,
        int_gen_by_fuel: Optional[pl.DataFrame] = None,
        pvgrpp_forecast: Optional[pl.DataFrame] = None,
    ) -> None:
        self.native_load = native_load
        self.wind_solar = wind_solar
        self.dam_as_mcpc = dam_as_mcpc
        self.int_gen_by_fuel = int_gen_by_fuel
        self.pvgrpp_forecast = pvgrpp_forecast

    def tag_summary(self) -> dict[str, str]:
        """Return REAL/NULL tags for each dataset."""
        return {
            "native_load": "REAL" if self.native_load is not None else "NULL",
            "wind_solar": "REAL" if self.wind_solar is not None else "NULL",
            "dam_as_mcpc": "REAL" if self.dam_as_mcpc is not None else "NULL",
            "int_gen_by_fuel": "REAL" if self.int_gen_by_fuel is not None else "NULL",
            "pvgrpp_forecast": "REAL" if self.pvgrpp_forecast is not None else "NULL",
        }


class ERCOTIngestPipeline:
    """Parse ERCOT raw files and persist to Parquet.

    Usage:
        pipeline = ERCOTIngestPipeline(raw_dir=Path("data/raw/uploads"),
                                        processed_dir=Path("data/processed"))
        store = pipeline.run(as_of_timestamp=datetime(2025, 5, 1, tzinfo=UTC))
    """

    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def run(
        self,
        as_of_timestamp: Optional[datetime] = None,
        skip_on_missing: bool = True,
    ) -> ERCOTParsedStore:
        """Run the full ingest pipeline.

        Args:
            as_of_timestamp: Walk-forward gate — rows after this are excluded.
            skip_on_missing: If True, log a warning and continue when a dataset
                is absent (NULL).  If False, raise MissingDataError.

        Returns:
            ERCOTParsedStore with all successfully parsed datasets.
        """
        store = ERCOTParsedStore()

        parsers = [
            ("native_load",    NativeLoadParser(),    "NativeLoad"),
            ("wind_solar",     WindSolarParser(),     "WindSolar"),
            ("dam_as_mcpc",    DAMASMCPCParser(),     "DAMASMCPC"),
            ("int_gen_by_fuel", IntGenByFuelParser(), "IntGenByFuel"),
            ("pvgrpp_forecast", PVGRPPForecastParser(), "PVGRPP"),
        ]

        for attr, parser, label in parsers:
            log = logger.bind(dataset=label)
            try:
                df = parser.parse(self.raw_dir, as_of_timestamp=as_of_timestamp)
                setattr(store, attr, df)
                log.info("ingest_ok", rows=len(df), tag="REAL")
                self._persist(df, label.lower())
            except MissingDataError as exc:
                log.warning("ingest_null", reason=str(exc), tag="NULL")
                if not skip_on_missing:
                    raise
            except Exception as exc:
                log.error("ingest_error", error=str(exc))
                if not skip_on_missing:
                    raise

        logger.info("ercot_pipeline_complete", tags=store.tag_summary())
        return store

    def _persist(self, df: pl.DataFrame, dataset_name: str) -> None:
        """Partition by year and write to Parquet under processed_dir."""
        base = self.processed_dir / dataset_name
        years = (
            df.with_columns(pl.col("interval_start_utc").dt.year().alias("_yr"))["_yr"]
            .unique()
            .to_list()
        )
        for year in sorted(years):
            subset = df.filter(pl.col("interval_start_utc").dt.year() == year)
            p = base / f"year={year}"
            p.mkdir(parents=True, exist_ok=True)
            path = p / "data.parquet"
            subset.write_parquet(path)
            logger.info("parquet_written", dataset=dataset_name, year=year, rows=len(subset))
