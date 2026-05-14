"""
ERCOT 60-Day Disclosure Report Parsers.

Each parser follows the Template Method pattern:
  - _discover_files()  -> list ZIP/CSV paths matching a glob
  - _read_zip()        -> stream CSV chunks from a ZIP archive
  - _parse_chunk()     -> validate schema and coerce dtypes
  - parse()            -> public entry point, returns a clean DataFrame

The ERCOTParserFactory.create() factory method instantiates the correct
parser subclass based on a string key, keeping the pipeline decoupled from
concrete parser implementations.

Supply/Demand Curve Reconstruction:
  The DAM offer and bid stacks arrive in a "wide" format with 10 MW/Price
  tier pairs per row. We melt these into a long-format "stacked" DataFrame
  with columns [timestamp, node, tier, mw, price] so that the full
  merit-order curve can be reconstructed by groupby-sorting.
"""

from __future__ import annotations

import io
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
import pandas as pd

from ercot_dart.config import (
    DAM_BID_COLS,
    DAM_OFFER_COLS,
    DAM_SPP_COLS,
    LOAD_FORECAST_COLS,
    NUM_OFFER_TIERS,
    PRICE_CAP,
    PRICE_FLOOR,
    SCED_SPP_COLS,
    ParserConfig,
)
from ercot_dart.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hour_ending_to_offset(he: pd.Series) -> pd.Series:
    """
    Convert ERCOT "HourEnding" (1-24) to a pandas Timedelta offset.
    HE=1 means the hour ending at 01:00, i.e. it STARTS at 00:00.
    We map to the interval START time: offset = (HE - 1) hours.
    """
    return pd.to_timedelta(he.astype(int) - 1, unit="h")


def _build_timestamp(date_col: pd.Series, he_col: pd.Series, tz: str) -> pd.Series:
    """
    Combine a date column and HourEnding column into a tz-aware Timestamp
    representing the START of each hourly interval.
    """
    base = pd.to_datetime(date_col)
    return (base + _hour_ending_to_offset(he_col)).dt.tz_localize(
        tz, ambiguous="infer", nonexistent="shift_forward"
    )


def _melt_tiers(
    df: pd.DataFrame,
    id_cols: list[str],
    n_tiers: int = NUM_OFFER_TIERS,
) -> pd.DataFrame:
    """
    Melt the wide 10-tier MW/Price columns into long format.

    Input:  columns [..., MW1, Price1, MW2, Price2, ..., MW10, Price10]
    Output: columns [..., tier, mw, price]

    The resulting DataFrame is the raw merit-order stack.
    Rows where MW is NaN or ≤ 0 are dropped (empty tiers).
    """
    mw_cols = [f"MW{i}" for i in range(1, n_tiers + 1)]
    px_cols = [f"Price{i}" for i in range(1, n_tiers + 1)]

    mw_long = df[id_cols + mw_cols].melt(
        id_vars=id_cols, var_name="tier", value_name="mw"
    )
    px_long = df[id_cols + px_cols].melt(
        id_vars=id_cols, var_name="_px_tier", value_name="price"
    )

    mw_long["tier"] = mw_long["tier"].str.extract(r"(\d+)").astype(int)
    px_long["tier"] = px_long["_px_tier"].str.extract(r"(\d+)").astype(int)

    merged = mw_long.merge(
        px_long[id_cols + ["tier", "price"]],
        on=id_cols + ["tier"],
    )
    return merged[merged["mw"] > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Abstract Base Parser
# ---------------------------------------------------------------------------

class BaseERCOTParser(ABC):
    """
    Template Method base class for all ERCOT disclosure report parsers.

    Subclasses implement _expected_columns, _parse_chunk(), and optionally
    override _post_process() for dataset-specific transformations.
    """

    _expected_columns: ClassVar[list[str]]

    def __init__(self, config: ParserConfig, raw_dir: Path, timezone: str) -> None:
        self.config = config
        self.raw_dir = raw_dir
        self.timezone = timezone

    # -- Template methods (must override) ------------------------------------

    @property
    @abstractmethod
    def file_glob(self) -> str:
        """Glob pattern relative to raw_dir for finding disclosure ZIPs."""

    @abstractmethod
    def _parse_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, coerce dtypes, return clean chunk."""

    # -- Template methods (may override) -------------------------------------

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dataset-level transforms applied after all chunks are concatenated."""
        return df

    # -- Concrete infrastructure --------------------------------------------

    def _discover_files(self) -> list[Path]:
        files = sorted(self.raw_dir.glob(self.file_glob))
        if not files:
            logger.warning(
                "No files matched glob",
                extra={"glob": self.file_glob, "raw_dir": str(self.raw_dir)},
            )
        return files

    def _read_zip(self, path: Path) -> Iterator[pd.DataFrame]:
        """Stream CSV chunks from a ZIP archive, handling nested ZIPs."""
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                # Some ERCOT ZIPs contain nested ZIPs
                inner_zips = [n for n in zf.namelist() if n.lower().endswith(".zip")]
                for inner_name in inner_zips:
                    inner_bytes = zf.read(inner_name)
                    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
                        csv_names = [
                            n for n in inner_zf.namelist() if n.lower().endswith(".csv")
                        ]
                        for csv_name in csv_names:
                            yield from self._stream_csv(inner_zf, csv_name)
            else:
                for csv_name in csv_names:
                    yield from self._stream_csv(zf, csv_name)

    def _stream_csv(
        self, zf: zipfile.ZipFile, csv_name: str
    ) -> Iterator[pd.DataFrame]:
        with zf.open(csv_name) as f:
            reader = pd.read_csv(
                f,
                encoding=self.config.encoding,
                chunksize=self.config.chunk_size,
                low_memory=False,
            )
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()
                yield chunk

    def _validate_schema(self, df: pd.DataFrame, source: str) -> None:
        missing = set(self._expected_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"[{self.__class__.__name__}] Missing columns in {source}: {missing}"
            )

    def parse(self) -> pd.DataFrame:
        """
        Discover, parse, and concatenate all matching disclosure files.
        Returns a single clean DataFrame sorted by timestamp.
        """
        files = self._discover_files()
        if not files:
            return pd.DataFrame()

        chunks: list[pd.DataFrame] = []
        for path in files:
            logger.info("Parsing file", extra={"file": path.name})
            try:
                for raw_chunk in self._read_zip(path):
                    clean = self._parse_chunk(raw_chunk)
                    if not clean.empty:
                        chunks.append(clean)
            except Exception as exc:
                logger.error(
                    "Failed to parse file",
                    extra={"file": path.name, "error": str(exc)},
                )
                raise

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)
        df = self._post_process(df)
        logger.info(
            "Parse complete",
            extra={"rows": len(df), "parser": self.__class__.__name__},
        )
        return df


# ---------------------------------------------------------------------------
# Concrete Parsers
# ---------------------------------------------------------------------------

class DAMEnergyOfferParser(BaseERCOTParser):
    """
    Parses 60d_DAM_EnergyOnlyOffers disclosure files.

    Each row represents one resource's submitted supply offer for a given
    delivery hour, expressed as up to 10 (price, MW) pairs forming a
    step-function supply curve segment.

    After melting, the output is the raw DAM supply stack — every offer tier
    that clears or would have cleared in merit order.
    """

    _expected_columns: ClassVar[list[str]] = DAM_OFFER_COLS

    @property
    def file_glob(self) -> str:
        return self.config.dam_offers_glob

    def _parse_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(chunk, "DAM_EnergyOnlyOffers")

        # Coerce tier columns to numeric, silencing non-numeric entries
        for i in range(1, NUM_OFFER_TIERS + 1):
            chunk[f"MW{i}"] = pd.to_numeric(chunk[f"MW{i}"], errors="coerce")
            chunk[f"Price{i}"] = pd.to_numeric(chunk[f"Price{i}"], errors="coerce").clip(
                lower=PRICE_FLOOR, upper=PRICE_CAP
            )

        chunk["timestamp"] = _build_timestamp(
            chunk["DeliveryDate"], chunk["HourEnding"], self.timezone
        )

        id_cols = ["timestamp", "SettlementPoint", "QSE", "ResourceName"]
        melted = _melt_tiers(chunk, id_cols)
        melted.rename(columns={"SettlementPoint": "node"}, inplace=True)
        return melted

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["timestamp", "node", "price"]).reset_index(drop=True)


class DAMEnergyBidParser(BaseERCOTParser):
    """
    Parses 60d_DAM_EnergyBids disclosure files.

    Demand bids are "price-willing-to-pay" curves — the demand side of the
    DAM merit order. Bids are sorted descending by price (highest willingness
    to pay first) to reconstruct the demand stack.
    """

    _expected_columns: ClassVar[list[str]] = DAM_BID_COLS

    @property
    def file_glob(self) -> str:
        return self.config.dam_bids_glob

    def _parse_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(chunk, "DAM_EnergyBids")

        for i in range(1, NUM_OFFER_TIERS + 1):
            chunk[f"MW{i}"] = pd.to_numeric(chunk[f"MW{i}"], errors="coerce")
            chunk[f"Price{i}"] = pd.to_numeric(chunk[f"Price{i}"], errors="coerce").clip(
                lower=PRICE_FLOOR, upper=PRICE_CAP
            )

        chunk["timestamp"] = _build_timestamp(
            chunk["DeliveryDate"], chunk["HourEnding"], self.timezone
        )

        id_cols = ["timestamp", "SettlementPoint", "QSE"]
        melted = _melt_tiers(chunk, id_cols)
        melted.rename(columns={"SettlementPoint": "node"}, inplace=True)
        return melted

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Demand stack: highest-priced bids set at top
        return df.sort_values(
            ["timestamp", "node", "price"], ascending=[True, True, False]
        ).reset_index(drop=True)


class DAMSettlementPriceParser(BaseERCOTParser):
    """
    Parses 60d_DAM_SPP (Settlement Point Price) files.

    These are the hourly DAM LMPs for each settlement point — the price at
    which virtual supply is settled in the Day-Ahead Market.
    """

    _expected_columns: ClassVar[list[str]] = DAM_SPP_COLS

    @property
    def file_glob(self) -> str:
        return self.config.dam_spp_glob

    def _parse_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(chunk, "DAM_SPP")

        chunk["SettlementPointPrice"] = pd.to_numeric(
            chunk["SettlementPointPrice"], errors="coerce"
        ).clip(lower=PRICE_FLOOR, upper=PRICE_CAP)

        chunk["timestamp"] = _build_timestamp(
            chunk["DeliveryDate"], chunk["HourEnding"], self.timezone
        )

        return chunk[["timestamp", "SettlementPoint", "SettlementPointType",
                       "SettlementPointPrice"]].rename(
            columns={
                "SettlementPoint": "node",
                "SettlementPointType": "node_type",
                "SettlementPointPrice": "dam_spp",
            }
        )

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(["timestamp", "node"]).reset_index(drop=True)


class SCEDSettlementPriceParser(BaseERCOTParser):
    """
    Parses 60d_SCED_SPP (Real-Time 15-min LMP) files.

    SCED runs every ~5 minutes but settlement occurs on 15-minute intervals.
    We resample to 15-min weighted-average LMPs, then compute the hourly
    RTM price as the arithmetic mean of the four 15-min intervals.

    This is the price at which virtual supply positions are unwound in
    the Real-Time Market.
    """

    _expected_columns: ClassVar[list[str]] = SCED_SPP_COLS

    @property
    def file_glob(self) -> str:
        return self.config.sced_spp_glob

    def _parse_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(chunk, "SCED_SPP")

        chunk["LMP"] = pd.to_numeric(chunk["LMP"], errors="coerce").clip(
            lower=PRICE_FLOOR, upper=PRICE_CAP
        )

        # SCEDTimestamp is "MM/DD/YYYY HH:MM:SS" in CPT
        chunk["timestamp"] = pd.to_datetime(
            chunk["SCEDTimestamp"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
        ).dt.tz_localize(self.timezone, ambiguous="infer", nonexistent="shift_forward")

        chunk = chunk.dropna(subset=["timestamp", "LMP"])
        chunk = chunk.rename(columns={"SettlementPoint": "node"})
        return chunk[["timestamp", "node", "LMP", "RepeatedHourFlag"]]

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 5-min SCED dispatches → 15-min settlement intervals,
        then aggregate to hourly RTM prices for DART spread computation.

        Step 1: Resample to 15-min mean (each 15-min interval = one settlement)
        Step 2: Resample 15-min → hourly mean  (4 intervals per hour)
        """
        df = df.sort_values(["node", "timestamp"])
        df = df.set_index("timestamp")

        rtm_15min = (
            df.groupby("node")["LMP"]
            .resample("15min")
            .mean()
            .reset_index()
            .rename(columns={"LMP": "rtm_15min_lmp"})
        )

        rtm_hourly = (
            rtm_15min.set_index("timestamp")
            .groupby("node")["rtm_15min_lmp"]
            .resample("1h")
            .mean()
            .reset_index()
            .rename(columns={"rtm_15min_lmp": "rtm_spp"})
        )

        return rtm_hourly.sort_values(["timestamp", "node"]).reset_index(drop=True)


class LoadForecastParser(BaseERCOTParser):
    """
    Parses ERCOT system and zonal load forecast files.

    These are the ERCOT-published day-ahead load forecasts — a key
    fundamental feature for the regime detection model. The Net Load
    Forecast Error (actual - forecast) is computed during feature engineering
    once actual metered data is available post-delivery.
    """

    _expected_columns: ClassVar[list[str]] = LOAD_FORECAST_COLS

    @property
    def file_glob(self) -> str:
        return self.config.load_forecast_glob

    def _parse_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(chunk, "LoadForecast")

        numeric_cols = ["SystemTotal", "North", "Houston", "South", "West"]
        for col in numeric_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk["timestamp"] = _build_timestamp(
            chunk["DeliveryDate"], chunk["HourEnding"], self.timezone
        )

        return chunk[["timestamp"] + numeric_cols].rename(
            columns={c: f"load_fcst_{c.lower()}" for c in numeric_cols}
        )

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merit-Order Curve Reconstructor
# ---------------------------------------------------------------------------

class MeritOrderReconstructor:
    """
    Reconstructs the supply and demand merit-order curves from parsed
    offer/bid stacks and computes market-clearing diagnostics.

    The supply curve is the monotonically increasing step function of
    cumulative MW vs. offer price. The demand curve is monotonically
    decreasing. Their intersection defines the theoretical clearing price.

    These reconstructed curves are used in Phase 4 to simulate the
    market impact of injecting our virtual bids.
    """

    @staticmethod
    def build_supply_curve(
        offers: pd.DataFrame, timestamp: pd.Timestamp, node: str
    ) -> pd.DataFrame:
        """
        Returns cumulative supply curve for a specific (timestamp, node).

        Columns: [price, mw, cumulative_mw]
        Sorted ascending by price (merit order).
        """
        mask = (offers["timestamp"] == timestamp) & (offers["node"] == node)
        stack = (
            offers[mask][["price", "mw"]]
            .sort_values("price")
            .reset_index(drop=True)
        )
        stack["cumulative_mw"] = stack["mw"].cumsum()
        return stack

    @staticmethod
    def build_demand_curve(
        bids: pd.DataFrame, timestamp: pd.Timestamp, node: str
    ) -> pd.DataFrame:
        """
        Returns cumulative demand curve for a specific (timestamp, node).

        Columns: [price, mw, cumulative_mw]
        Sorted descending by price (highest willingness-to-pay first).
        """
        mask = (bids["timestamp"] == timestamp) & (bids["node"] == node)
        stack = (
            bids[mask][["price", "mw"]]
            .sort_values("price", ascending=False)
            .reset_index(drop=True)
        )
        stack["cumulative_mw"] = stack["mw"].cumsum()
        return stack

    @staticmethod
    def estimate_clearing_price(
        supply: pd.DataFrame, demand: pd.DataFrame
    ) -> dict[str, float]:
        """
        Estimate the theoretical clearing price and volume where
        cumulative supply MW >= cumulative demand MW.

        Returns a dict with keys: clearing_price, clearing_mw,
        supply_surplus_mw, demand_surplus_mw.
        """
        if supply.empty or demand.empty:
            return {
                "clearing_price": np.nan,
                "clearing_mw": np.nan,
                "supply_surplus_mw": np.nan,
                "demand_surplus_mw": np.nan,
            }

        total_demand = demand["mw"].sum()
        # Find the first supply tier where cumulative MW exceeds total demand
        clearing_row = supply[supply["cumulative_mw"] >= total_demand]

        if clearing_row.empty:
            # Demand exceeds all available supply — scarcity pricing
            clearing_price = PRICE_CAP
            clearing_mw = supply["cumulative_mw"].iloc[-1]
        else:
            clearing_price = float(clearing_row["price"].iloc[0])
            clearing_mw = float(clearing_row["cumulative_mw"].iloc[0])

        supply_surplus = max(0.0, clearing_mw - total_demand)
        demand_surplus = max(0.0, total_demand - clearing_mw)

        return {
            "clearing_price": clearing_price,
            "clearing_mw": clearing_mw,
            "supply_surplus_mw": supply_surplus,
            "demand_surplus_mw": demand_surplus,
        }


# ---------------------------------------------------------------------------
# Parser Factory
# ---------------------------------------------------------------------------

_PARSER_REGISTRY: dict[str, type[BaseERCOTParser]] = {
    "dam_offers": DAMEnergyOfferParser,
    "dam_bids": DAMEnergyBidParser,
    "dam_spp": DAMSettlementPriceParser,
    "sced_spp": SCEDSettlementPriceParser,
    "load_forecast": LoadForecastParser,
}


class ERCOTParserFactory:
    """
    Factory for instantiating ERCOT disclosure report parsers.

    Usage:
        parser = ERCOTParserFactory.create("dam_offers", config, paths, tz)
        df = parser.parse()

    The factory decouples the pipeline orchestrator from concrete parser
    classes, making it straightforward to register new report types.
    """

    @staticmethod
    def create(
        parser_key: str,
        config: ParserConfig,
        raw_dir: Path,
        timezone: str,
    ) -> BaseERCOTParser:
        if parser_key not in _PARSER_REGISTRY:
            raise KeyError(
                f"Unknown parser key '{parser_key}'. "
                f"Available: {sorted(_PARSER_REGISTRY)}"
            )
        cls = _PARSER_REGISTRY[parser_key]
        return cls(config, raw_dir, timezone)

    @staticmethod
    def register(key: str, cls: type[BaseERCOTParser]) -> None:
        """Register a custom parser under a new key."""
        _PARSER_REGISTRY[key] = cls

    @staticmethod
    def available_parsers() -> list[str]:
        return sorted(_PARSER_REGISTRY.keys())
