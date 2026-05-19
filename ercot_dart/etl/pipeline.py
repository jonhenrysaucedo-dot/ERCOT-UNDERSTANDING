"""
ETL Pipeline Orchestrator — Factory Method Pattern.

The ETLPipeline class is the top-level coordinator that:
  1. Invokes the ERCOTParserFactory to build all required parsers
  2. Runs each parser to produce clean DataFrames
  3. Hands the parsed DataFrames to the FeatureEngineer
  4. Persists the resulting feature matrix to disk (Parquet)
  5. Provides a cached fast-path for repeated runs

The pipeline is designed to be run nightly after ERCOT posts the previous
day's 60-day disclosure files (typically available by ~08:00 AM CPT the
following day), well before the 10:00 AM gate closure.

Usage:
    config = ETLConfig()
    pipeline = ETLPipeline(config)
    feature_matrix = pipeline.run()
"""

from __future__ import annotations

import hashlib
import time as _time
from pathlib import Path
from typing import Optional

import pandas as pd

from ercot_dart.config import ETLConfig, FeatureConfig
from ercot_dart.etl.features import FeatureEngineer
from ercot_dart.etl.parsers import ERCOTParserFactory
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Parse Result Container
# ---------------------------------------------------------------------------

class ParsedDataStore:
    """
    Lightweight container for all parsed ERCOT DataFrames.

    Centralises access so downstream components (feature engineer,
    backtester, compliance checker) all reference the same in-memory store
    without re-parsing disk files.
    """

    def __init__(self) -> None:
        self.dam_offers: pd.DataFrame = pd.DataFrame()
        self.dam_bids: pd.DataFrame = pd.DataFrame()
        self.dam_spp: pd.DataFrame = pd.DataFrame()
        self.rtm_spp: pd.DataFrame = pd.DataFrame()
        self.load_forecast: pd.DataFrame = pd.DataFrame()

    def is_complete(self) -> bool:
        return all(
            not df.empty
            for df in [self.dam_offers, self.dam_bids, self.dam_spp, self.rtm_spp]
        )

    def summary(self) -> dict[str, int]:
        return {
            "dam_offers_rows": len(self.dam_offers),
            "dam_bids_rows": len(self.dam_bids),
            "dam_spp_rows": len(self.dam_spp),
            "rtm_spp_rows": len(self.rtm_spp),
            "load_forecast_rows": len(self.load_forecast),
        }


# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------

class _CacheManager:
    """
    Parquet-backed caching for parsed DataFrames.

    Cache keys are SHA-256 hashes of the list of source file paths and
    their modification times, ensuring stale cache is automatically
    invalidated when new disclosure files are downloaded.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, parser_key: str, raw_dir: Path, glob: str) -> str:
        files = sorted(raw_dir.glob(glob))
        fingerprint = "".join(f"{p.name}:{p.stat().st_mtime}" for p in files)
        return hashlib.sha256(f"{parser_key}:{fingerprint}".encode()).hexdigest()[:16]

    def get(self, parser_key: str, raw_dir: Path, glob: str) -> Optional[pd.DataFrame]:
        key = self._key(parser_key, raw_dir, glob)
        path = self.cache_dir / f"{parser_key}_{key}.parquet"
        if path.exists():
            logger.info("Cache hit", extra={"parser": parser_key, "key": key})
            return pd.read_parquet(path)
        return None

    def put(self, parser_key: str, raw_dir: Path, glob: str, df: pd.DataFrame) -> None:
        key = self._key(parser_key, raw_dir, glob)
        path = self.cache_dir / f"{parser_key}_{key}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Cached parse result", extra={"parser": parser_key, "path": str(path)})


# ---------------------------------------------------------------------------
# Pipeline Step Interface (Strategy Pattern)
# ---------------------------------------------------------------------------

class _PipelineStep:
    """
    A single named step in the pipeline with timing and error capture.
    Wraps an arbitrary callable so the orchestrator can log uniformly.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, fn, *args, **kwargs):
        logger.info("Step starting", extra={"step": self.name})
        t0 = _time.monotonic()
        try:
            result = fn(*args, **kwargs)
            elapsed = _time.monotonic() - t0
            logger.info(
                "Step complete",
                extra={"step": self.name, "elapsed_s": round(elapsed, 2)},
            )
            return result
        except Exception as exc:
            elapsed = _time.monotonic() - t0
            logger.error(
                "Step failed",
                extra={"step": self.name, "elapsed_s": round(elapsed, 2), "error": str(exc)},
            )
            raise


# ---------------------------------------------------------------------------
# ETL Pipeline
# ---------------------------------------------------------------------------

class ETLPipeline:
    """
    Top-level ETL orchestrator.

    The pipeline is constructed once with an ETLConfig and can be run
    repeatedly (e.g., nightly cron). On each run it:
      1. Parses all ERCOT 60-day disclosure ZIPs from raw_dir
      2. Engineers the full feature matrix
      3. Writes feature_matrix.parquet to processed_dir
      4. Returns the feature matrix for downstream use

    External data (gas prices, weather forecasts) are injected as optional
    DataFrames so the pipeline remains testable without live API access.
    """

    _PARSER_GLOB_MAP = {
        "dam_offers":    "dam_offers_glob",
        "dam_bids":      "dam_bids_glob",
        "dam_spp":       "dam_spp_glob",
        "sced_spp":      "sced_spp_glob",
        "load_forecast": "load_forecast_glob",
    }

    def __init__(self, config: ETLConfig) -> None:
        self.config = config
        self.feature_engineer = FeatureEngineer(config.features)
        self._cache = _CacheManager(config.paths.cache_dir) if config.parser.cache_parsed else None
        self._store = ParsedDataStore()

    # -----------------------------------------------------------------------
    # Parsing layer
    # -----------------------------------------------------------------------

    def _parse_dataset(self, parser_key: str) -> pd.DataFrame:
        """
        Parse a single disclosure dataset, using cache if available.
        """
        glob_attr = self._PARSER_GLOB_MAP[parser_key]
        glob_pattern = getattr(self.config.parser, glob_attr)

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(parser_key, self.config.paths.raw_dir, glob_pattern)
            if cached is not None:
                return cached

        parser = ERCOTParserFactory.create(
            parser_key,
            self.config.parser,
            self.config.paths.raw_dir,
            self.config.timezone,
        )
        step = _PipelineStep(f"parse:{parser_key}")
        df = step.run(parser.parse)

        if self._cache is not None and not df.empty:
            self._cache.put(parser_key, self.config.paths.raw_dir, glob_pattern, df)

        return df

    def _run_parsing_layer(self) -> ParsedDataStore:
        store = ParsedDataStore()
        store.dam_offers   = self._parse_dataset("dam_offers")
        store.dam_bids     = self._parse_dataset("dam_bids")
        store.dam_spp      = self._parse_dataset("dam_spp")
        store.rtm_spp      = self._parse_dataset("sced_spp")
        store.load_forecast = self._parse_dataset("load_forecast")
        self._store = store

        logger.info("Parsing layer complete", extra=store.summary())
        return store

    # -----------------------------------------------------------------------
    # Feature engineering layer
    # -----------------------------------------------------------------------

    def _run_feature_layer(
        self,
        store: ParsedDataStore,
        gas_prices: Optional[pd.DataFrame] = None,
        weather: Optional[pd.DataFrame] = None,
        actual_load: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        fe = self.feature_engineer

        # DART spread (target)
        step = _PipelineStep("feature:dart_spread")
        dart = step.run(fe.compute_dart_spread, store.dam_spp, store.rtm_spp)

        # Shift factor proxy
        step = _PipelineStep("feature:shift_factors")
        sfp = step.run(fe.compute_shift_factor_proxy, store.dam_spp)

        # Temporal / Fourier
        step = _PipelineStep("feature:temporal")
        unique_ts = dart[["timestamp"]].drop_duplicates()["timestamp"]
        temporal = step.run(fe.compute_temporal_features, unique_ts)

        # Load forecast error (needs actual load to compute NLFE)
        load_feats = pd.DataFrame()
        if actual_load is not None and not store.load_forecast.empty:
            step = _PipelineStep("feature:load_forecast_error")
            load_feats = step.run(
                fe.compute_net_load_forecast_error,
                actual_load,
                store.load_forecast,
            )

        # Assemble
        step = _PipelineStep("feature:assemble_matrix")
        feature_matrix = step.run(
            fe.build_feature_matrix,
            dart=dart,
            offers=store.dam_offers,
            dam_spp=store.dam_spp,
            load_features=load_feats,
            temporal_features=temporal,
            shift_factor_features=sfp,
            gas_prices=gas_prices,
            weather=weather,
        )
        return feature_matrix

    # -----------------------------------------------------------------------
    # Persistence layer
    # -----------------------------------------------------------------------

    def _persist(self, df: pd.DataFrame, filename: str = "feature_matrix.parquet") -> Path:
        out_path = self.config.paths.processed_dir / filename
        df.to_parquet(out_path, index=False)
        logger.info("Feature matrix persisted", extra={"path": str(out_path), "rows": len(df)})
        return out_path

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        gas_prices: Optional[pd.DataFrame] = None,
        weather: Optional[pd.DataFrame] = None,
        actual_load: Optional[pd.DataFrame] = None,
        persist: bool = True,
    ) -> pd.DataFrame:
        """
        Execute the full ETL pipeline.

        Parameters
        ----------
        gas_prices : DataFrame with columns [date, gas_price_per_mmbtu]
        weather    : DataFrame with columns [timestamp, temp_f_north, ...]
        actual_load: DataFrame with columns [timestamp, load_actual_system_total]
        persist    : Write feature matrix to processed_dir as Parquet

        Returns
        -------
        feature_matrix : pd.DataFrame
        """
        logger.info("ETL pipeline starting")
        t0 = _time.monotonic()

        store = self._run_parsing_layer()

        if not store.is_complete():
            logger.warning(
                "Parsing layer incomplete — some datasets are empty. "
                "Feature matrix may have significant NaN coverage."
            )

        feature_matrix = self._run_feature_layer(store, gas_prices, weather, actual_load)

        if persist and not feature_matrix.empty:
            self._persist(feature_matrix)

        elapsed = _time.monotonic() - t0
        logger.info(
            "ETL pipeline complete",
            extra={
                "elapsed_s": round(elapsed, 2),
                "feature_rows": len(feature_matrix),
                "feature_cols": len(feature_matrix.columns),
            },
        )
        return feature_matrix

    def load_feature_matrix(self, filename: str = "feature_matrix.parquet") -> pd.DataFrame:
        """Load a previously persisted feature matrix from disk."""
        path = self.config.paths.processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Feature matrix not found at {path}. Run pipeline.run() first."
            )
        df = pd.read_parquet(path)
        logger.info("Loaded feature matrix", extra={"path": str(path), "rows": len(df)})
        return df

    @property
    def store(self) -> ParsedDataStore:
        """Access the parsed data store after pipeline.run() has been called."""
        return self._store


# ---------------------------------------------------------------------------
# Pipeline Factory
# ---------------------------------------------------------------------------

class ETLPipelineFactory:
    """
    Factory for constructing ETL pipelines with different configurations.

    Provides named constructors for common deployment profiles.
    """

    @staticmethod
    def create(config: Optional[ETLConfig] = None) -> ETLPipeline:
        """Create a pipeline with the given config, or default config."""
        return ETLPipeline(config or ETLConfig())

    @staticmethod
    def from_env() -> ETLPipeline:
        """
        Construct pipeline from environment variables.

        Required env vars:
            ERCOT_RAW_DIR       Path to raw disclosure ZIPs
            ERCOT_PROCESSED_DIR Path for processed Parquet output
            ERCOT_CACHE_DIR     Path for parse cache
        """
        config = ETLConfig()  # DataPaths reads from env in __post_init__
        return ETLPipeline(config)
