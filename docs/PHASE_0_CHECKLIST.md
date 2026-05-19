# Phase 0 — Data Acquisition Checklist (revised)

**Goal:** Get all training data into `data/raw/` so Phase 1 ingestion can start.
**Node:** `RN_QTUM_SLR` (West Zone solar, 160 MW PV) — confirmed
**Estimated time:** 2–5 days, no QSE involvement required, no paid API needed

---

## What changed from the first draft

The first version of this checklist routed everything through your QSE. **It was wrong.** Almost every dataset we need is classified by ERCOT as `Audience: Public, Security Classification: Public` and is accessible either via:

1. The **`gridstatus`** open-source Python library (free, no API key, scrapes ERCOT public reports)
2. The **ERCOT Public Reports archive** at `https://data.ercot.com/data-product-archive/` (free, anonymous CSV/zip downloads)
3. **Iowa State ASOS** for weather (free, no auth)

The hosted version at `gridstatus.io` is paid (rate limits on free tier), but we are *not* using the hosted version. The OSS library does it all.

---

## Step 1 — Install the toolchain

```bash
pip install gridstatus pyarrow polars pandas
```

That's it for data acquisition. No credentials, no portals, no QSE emails.

---

## Step 2 — Pull historical LMPs (the gap in your local data)

This is the only dataset you cannot reconstruct from your uploaded files. Run:

```python
import gridstatus
import pandas as pd

ercot = gridstatus.Ercot()

# DAM hourly SPPs
dam_spp = ercot.get_spp(
    start='2023-01-01',
    end='2026-05-17',
    market='DAY_AHEAD_HOURLY',
    locations=['RN_QTUM_SLR', 'HB_WEST', 'HB_HUBAVG'],
)
dam_spp.to_parquet('data/raw/gridstatus/dam_spp.parquet')

# RTM 15-min SPPs
rtm_spp = ercot.get_spp(
    start='2023-01-01',
    end='2026-05-17',
    market='REAL_TIME_15_MIN',
    locations=['RN_QTUM_SLR', 'HB_WEST', 'HB_HUBAVG'],
)
rtm_spp.to_parquet('data/raw/gridstatus/rtm_spp.parquet')
```

**Heads-up on `RN_QTUM_SLR` specifically:** if the node was only recently energized, gridstatus may return empty results for early date ranges. That's not an error — it's a reflection of when the node started settling. The hubs (`HB_WEST`, `HB_HUBAVG`) will have full 2023+ history and serve as your proxy until the node has 90+ days of price history.

Runtime: ~5 minutes for hubs; varies for the node depending on its history depth.

---

## Step 3 — Reconcile gridstatus against your local files

This is the step that justifies the hybrid architecture. For each field your local files cover, pull the same date range from gridstatus and confirm they match.

```python
# Example reconciliation: load by zone, Jan 2024
gs_load = ercot.get_load(start='2024-01-01', end='2024-02-01')
local_load = pd.read_csv('data/raw/local/Native_Load_2024.csv')
# Normalize timestamps, merge on (timestamp, zone), compute |delta|/value
# Save delta report to reports/reconciliation_load_2024-01.html
```

Reconcile these five datasets:
- Native Load by zone
- Hourly Wind Generation actual
- Hourly Solar Generation actual
- DAM AS prices (RegUp, RegDn, RRS, NSpin, ECRS)
- Fuel mix (Coal, Gas, Gas-CC, Wind, Solar, Nuclear, WSL)

**Pass criterion:** ≥99% of rows match within 0.5% relative tolerance.

**If reconciliation fails:** investigate before proceeding. Common causes are timezone mismatches (CT vs UTC), HE vs HS conventions, ERCOT post-publication revisions, or gridstatus parser bugs (rare but real). Don't paper over discrepancies — they will surface later as inference-time prediction errors.

**If reconciliation passes:** the two sources are interchangeable. Train on local, infer on gridstatus, sleep at night.

---

## Step 4 — Pull 60-Day Disclosure for the backtester

This is the bid-stack data needed by M9 for slippage modeling. Pulled as full zip bundles from the ERCOT public archive — **no login**, despite the misleading `MIS LOG IN` link at the top of ERCOT's page (that's for posting reports, not consuming them).

```bash
# The archive URL pattern:
# https://data.ercot.com/data-product-archive/NP3-966-ER

# Or use the gridstatus equivalent:
python -c "
import gridstatus
ercot = gridstatus.Ercot()
df = ercot.get_60_day_dam_disclosure(start='2024-01-01', end='2024-01-31')
df.to_parquet('data/raw/gridstatus/dam_disclosure_2024_01.parquet')
"
```

**Size warning:** ~5 GB per year of bundles. Plan for ~10 GB total for 2024-2025. Pull in monthly chunks if disk space is constrained.

If you only want the bid awards (not the full ESR/AS/PTP data), there are sub-endpoints — see `gridstatus.Ercot` source code for the filtered methods.

---

## Step 5 — Weather observations

Iowa State ASOS archive, free, no auth. Three West Texas stations cover the geography of `RN_QTUM_SLR`:

- **MAF** — Midland-Odessa
- **LBB** — Lubbock
- **SJT** — San Angelo

Plus stations for the other ERCOT zones if you want a full-state feature set: **DFW** (Dallas), **HOU** (Houston), **AUS** (Austin), **SAT** (San Antonio).

```python
import pandas as pd

def fetch_asos(station: str, start: str, end: str) -> pd.DataFrame:
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station}&data=tmpf&year1={start[:4]}&month1={start[5:7]}&day1={start[8:10]}"
        f"&year2={end[:4]}&month2={end[5:7]}&day2={end[8:10]}"
        f"&tz=UTC&format=onlycomma&latlon=no&missing=M&trace=T"
    )
    return pd.read_csv(url, parse_dates=['valid'])

for sta in ['MAF', 'LBB', 'SJT', 'DFW', 'HOU', 'AUS', 'SAT']:
    df = fetch_asos(sta, '2023-01-01', '2026-05-18')
    df.to_parquet(f'data/raw/weather/{sta}.parquet')
```

---

## Step 6 — Sanity checks (don't skip)

Before declaring Phase 0 done, run all four:

1. **DART spread sanity.** Compute `RT - DA` at `HB_WEST` for one mid-2024 day. Reasonable range: roughly -$50 to +$200/MWh, most hours within ±$15. Persistent four-digit spreads = broken join.
2. **PV forecast vs actual sanity.** For a sunny mid-day hour in summer 2025, PVGRPP forecast and actual hourly solar output should be within ~10% statewide. 50%+ deviation = timezone or HE/HS bug.
3. **AS price sanity.** RegUp MCPC should run roughly 3–10× ECRS MCPC under normal conditions, converging during scarcity. ECRS persistently higher = column order swap.
4. **Disclosure file completeness.** For each delivery day in the 60-Day data, the bundle should have ~16 files. Days with <5 files = incomplete extraction.

All four pass → Phase 0 done → Claude Code can start Phase 1.

---

## What you do NOT need (and what I told you incorrectly last time)

- ❌ ERCOT MIS Market Participant credentials — public reports cover everything
- ❌ QSE intervention to pull data — public reports cover everything
- ❌ Paid `gridstatus.io` hosted API — the open-source library covers everything
- ❌ Pulling `NP4-160-SG` Settlement Points List specifically — the node name is confirmed; the list is useful as reference but not blocking

The ERCOT API portal at `developer.ercot.com` requires a free API key for some newer endpoints, but it's not required for any v1 data needs. If we hit an endpoint gridstatus doesn't cover, sign up there — it's free, 5-minute self-service.

---

## After Phase 0, you should have

```
data/raw/
├── local/                          # Your uploaded files (immutable)
│   ├── load/
│   ├── wind_solar_actual/
│   ├── as_prices/
│   ├── fuel_mix/
│   └── solar_forecast/
├── gridstatus/                     # LMP gap fill + 60-day disclosure
│   ├── dam_spp.parquet
│   ├── rtm_spp.parquet
│   └── dam_disclosure_YYYY_MM.parquet
└── weather/
    ├── MAF.parquet
    ├── LBB.parquet
    └── ... (etc.)
```

Plus a `reports/reconciliation_*.html` set showing local vs gridstatus matched within tolerance.

That's the green light for Phase 1.
