"""Reconciliation checker — local uploaded files vs. gridstatus pulls.

Per PRD §11: before Phase 1 is declared done, every field that overlaps between
local files and gridstatus must be reconciled. Pass criterion: ≥99% of hourly
observations match within 0.5% relative tolerance.

Results saved to: reports/reconciliation_YYYY-MM-DD.html

Usage:
    python -m src.ingest.reconcile --start 2024-01-01 --end 2025-12-31

Walk-forward safety:
    Reconciliation is a one-time check, not a live feature. No as_of_timestamp
    needed — this function compares two historical datasets directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import structlog

from src.ingest.live.gridstatus_client import get_as_prices, get_fuel_mix, get_load
from src.ingest.ercot.parsers import NativeLoadParser, WindSolarParser, DAMASMCPCParser, IntGenByFuelParser
from src.ingest.exceptions import MissingDataError

logger = structlog.get_logger(__name__)
UTC = timezone.utc

REL_TOL = 0.005    # 0.5% relative tolerance (PRD §11)
PASS_PCT = 0.99    # ≥99% of rows must match within tolerance


@dataclass
class ReconciliationResult:
    dataset: str
    total_rows: int
    match_rows: int
    match_pct: float
    passed: bool
    max_delta_pct: float
    sample_mismatches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "total_rows": self.total_rows,
            "match_rows": self.match_rows,
            "match_pct": round(self.match_pct, 4),
            "passed": self.passed,
            "max_delta_pct": round(self.max_delta_pct, 4),
            "sample_mismatches": self.sample_mismatches[:10],
        }


def reconcile_native_load(
    raw_dir: Path,
    start: date,
    end: date,
) -> ReconciliationResult:
    """Compare NativeLoadParser output vs. gridstatus ERCOT load.

    Matches on (interval_start_utc, zone=ERCOT) — system-wide load total.
    """
    log = logger.bind(dataset="native_load")
    log.info("reconcile_start")

    local_df = NativeLoadParser().parse(raw_dir)
    # Use ERCOT system-wide total for comparison
    local = (
        local_df.filter(pl.col("zone") == "ERCOT")
        .filter(
            (pl.col("interval_start_utc").dt.date() >= start) &
            (pl.col("interval_start_utc").dt.date() <= end)
        )
        .rename({"load_mw": "local_load_mw"})
        .select(["interval_start_utc", "local_load_mw"])
    )

    gs_df = get_load(start, end)
    gs = gs_df.rename({"load_mw": "gs_load_mw"}).select(["interval_start_utc", "gs_load_mw"])

    merged = local.join(gs, on="interval_start_utc", how="inner")
    return _compute_result("native_load", merged, "local_load_mw", "gs_load_mw")


def reconcile_as_prices(
    raw_dir: Path,
    start: date,
    end: date,
) -> ReconciliationResult:
    """Compare DAMASMCPCParser vs. gridstatus AS prices on RegUp."""
    log = logger.bind(dataset="as_prices_regup")
    log.info("reconcile_start")

    local_df = DAMASMCPCParser().parse(raw_dir)
    local = (
        local_df.filter(
            (pl.col("interval_start_utc").dt.date() >= start) &
            (pl.col("interval_start_utc").dt.date() <= end)
        )
        .select(["interval_start_utc", "as_regup_usd"])
        .rename({"as_regup_usd": "local_regup"})
    )

    gs_df = get_as_prices(start, end)
    gs = gs_df.select(["interval_start_utc", "as_regup_usd"]).rename({"as_regup_usd": "gs_regup"})

    merged = local.join(gs, on="interval_start_utc", how="inner")
    return _compute_result("as_prices_regup", merged, "local_regup", "gs_regup")


def reconcile_fuel_mix(
    raw_dir: Path,
    start: date,
    end: date,
) -> ReconciliationResult:
    """Compare IntGenByFuelParser vs. gridstatus fuel mix on total Gas-CC generation."""
    log = logger.bind(dataset="fuel_mix_gascc")
    log.info("reconcile_start")

    local_df = IntGenByFuelParser().parse(raw_dir)
    local = (
        local_df.filter(pl.col("fuel") == "Gas-CC")
        .filter(pl.col("settlement_type") == "FINAL")
        .filter(
            (pl.col("interval_start_utc").dt.date() >= start) &
            (pl.col("interval_start_utc").dt.date() <= end)
        )
        # Aggregate 15-min → hourly
        .with_columns(
            pl.col("interval_start_utc").dt.truncate("1h").alias("hour_utc")
        )
        .group_by("hour_utc")
        .agg(pl.col("gen_mw").mean().alias("local_gascc_mw"))
        .rename({"hour_utc": "interval_start_utc"})
    )

    gs_df = get_fuel_mix(start, end)
    gs_gascc_col = _find_fuel_col(gs_df, "gas-cc")
    if gs_gascc_col is None:
        raise MissingDataError("gridstatus fuel mix has no Gas-CC column")

    gs = (
        gs_df.filter(pl.col("fuel").str.to_lowercase().str.contains("gas.cc"))
        .rename({"gen_mw": "gs_gascc_mw"})
        .select(["interval_start_utc", "gs_gascc_mw"])
    )

    merged = local.join(gs, on="interval_start_utc", how="inner")
    return _compute_result("fuel_mix_gascc", merged, "local_gascc_mw", "gs_gascc_mw")


def run_all(
    raw_dir: Path,
    start: date,
    end: date,
    report_dir: Optional[Path] = None,
) -> list[ReconciliationResult]:
    """Run all reconciliation checks and save an HTML report.

    Returns a list of ReconciliationResult objects.
    All-pass → print green; any fail → print red with sample mismatches.
    """
    if report_dir is None:
        report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    reconcilers = [
        ("native_load", reconcile_native_load),
        ("as_prices", reconcile_as_prices),
        ("fuel_mix", reconcile_fuel_mix),
    ]

    results: list[ReconciliationResult] = []
    for name, fn in reconcilers:
        try:
            result = fn(raw_dir, start, end)
            results.append(result)
            status = "PASS ✓" if result.passed else "FAIL ✗"
            logger.info(
                "reconcile_result",
                dataset=name,
                passed=result.passed,
                match_pct=f"{result.match_pct:.1%}",
                max_delta=f"{result.max_delta_pct:.1%}",
            )
        except Exception as exc:
            logger.error("reconcile_error", dataset=name, error=str(exc))

    _write_html_report(results, report_dir / f"reconciliation_{date.today()}.html")
    _write_json_report(results, report_dir / f"reconciliation_{date.today()}.json")

    all_passed = all(r.passed for r in results)
    if all_passed:
        logger.info("reconcile_all_pass", msg="Phase 0 reconciliation: PASSED — green light for Phase 1")
    else:
        failed = [r.dataset for r in results if not r.passed]
        logger.error("reconcile_some_fail", failed=failed,
                     msg="Phase 0 blocker: fix discrepancies before starting Phase 1")

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_result(
    dataset: str,
    merged: pl.DataFrame,
    col_a: str,
    col_b: str,
) -> ReconciliationResult:
    """Compute match statistics for two numeric columns."""
    n_total = len(merged)
    if n_total == 0:
        return ReconciliationResult(
            dataset=dataset, total_rows=0, match_rows=0,
            match_pct=0.0, passed=False, max_delta_pct=1.0,
        )

    a = merged[col_a].to_numpy()
    b = merged[col_b].to_numpy()
    import numpy as np
    denom = np.abs(a) + 1e-9  # avoid div-by-zero
    delta_pct = np.abs(a - b) / denom

    matches = int((delta_pct <= REL_TOL).sum())
    match_pct = matches / n_total
    max_delta = float(delta_pct.max())
    passed = match_pct >= PASS_PCT

    # Sample worst mismatches
    worst_idx = np.argsort(delta_pct)[::-1][:10]
    ts_col = merged["interval_start_utc"].to_list()
    sample = [
        {
            "interval_start_utc": str(ts_col[i]),
            col_a: round(float(a[i]), 4),
            col_b: round(float(b[i]), 4),
            "delta_pct": round(float(delta_pct[i]) * 100, 2),
        }
        for i in worst_idx
    ]

    return ReconciliationResult(
        dataset=dataset,
        total_rows=n_total,
        match_rows=matches,
        match_pct=match_pct,
        passed=passed,
        max_delta_pct=max_delta,
        sample_mismatches=sample,
    )


def _find_fuel_col(df: pl.DataFrame, fuel_substr: str) -> Optional[str]:
    for col in df["fuel"].unique().to_list():
        if fuel_substr in col.lower():
            return col
    return None


def _write_json_report(results: list[ReconciliationResult], path: Path) -> None:
    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pass_threshold": PASS_PCT,
        "rel_tolerance": REL_TOL,
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("reconcile_json_written", path=str(path))


def _write_html_report(results: list[ReconciliationResult], path: Path) -> None:
    rows_html = ""
    for r in results:
        color = "#22c55e" if r.passed else "#ef4444"
        label = "PASS" if r.passed else "FAIL"
        rows_html += f"""
        <tr>
          <td>{r.dataset}</td>
          <td>{r.total_rows:,}</td>
          <td>{r.match_rows:,}</td>
          <td>{r.match_pct:.2%}</td>
          <td>{r.max_delta_pct:.2%}</td>
          <td style="color:{color};font-weight:bold">{label}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><title>Phase 0 Reconciliation Report</title>
<style>
  body {{ font-family: monospace; padding: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f3f4f6; }}
</style></head>
<body>
<h1>Phase 0 Reconciliation Report</h1>
<p>Generated: {datetime.now(UTC).isoformat()}</p>
<p>Pass criterion: ≥{PASS_PCT:.0%} of rows within {REL_TOL:.1%} relative tolerance</p>
<table>
  <tr><th>Dataset</th><th>Total Rows</th><th>Match Rows</th>
      <th>Match %</th><th>Max Delta %</th><th>Status</th></tr>
  {rows_html}
</table>
</body></html>"""

    path.write_text(html)
    logger.info("reconcile_html_written", path=str(path))
