# Data Inventory — `quantum_dart`

**Last updated:** 2026-05-19
**Tag legend:** `[REAL]` = real ERCOT settlement / observation data · `[NULL]` = missing · `[SYNTHETIC]` = simulated (never permitted for prices per CLAUDE.md §131)

---

## Have (in `data/raw/uploads/`)

| Dataset | Years | ERCOT Report | Tag | Notes |
|---|---|---|---|---|
| Installed Generation by Fuel | 2025, 2026 | (mfg report) | `[REAL]` | `IntGenbyFuel*.xlsx` |
| Hourly Wind/Solar Output | 2023, 2024, 2025 | NP4-732-CD-style | `[REAL]` | duplicates present (different upload hashes, same content); de-dup at ingest |
| Native Load by Zone | 2023, 2024, 2025, 2026 | NP6-345-CD | `[REAL]` | `Native_Load_*.xlsx` |
| DAM AS MCPC (Ancillary Service capacity prices) | 2023, 2024, 2025, 2026 | NP4-188-CD | `[REAL]` | `DAMASMCPC_*.csv` |
| Hourly STPPF / PVGRPP Solar Forecast | May 2026 snapshots only | NP4-743-CD | `[REAL]` | 3 daily files; rolling forecasts |

## Missing (must acquire in Phase 0 — all free, no credentials needed)

| Dataset | Years needed | Source | Tag | Action |
|---|---|---|---|---|
| **DAM Settlement Point Prices** | 2023–2026 | gridstatus | `[NULL]` | **BLOCKER** — DART = RTM − DAM |
| **RTM Settlement Point Prices (15-min)** | 2023–2026 | gridstatus | `[NULL]` | **BLOCKER** |
| 60-Day DAM Disclosure | 2024–2025 | ERCOT public archive NP3-966-ER | `[NULL]` | M9 backtest slippage model |
| 60-Day SCED Disclosure | 2024–2025 | ERCOT public archive NP3-965-ER | `[NULL]` | M9 backtest |
| STWPF / WGRPP Wind Forecast | 2023–2026 | gridstatus | `[NULL]` | `wind_delta` feature |
| Weather — West Texas temperature | 2023–2026 | Iowa State ASOS (MAF, LBB, SJT) | `[NULL]` | `src/ingest/live/asos_weather.py` ready |
| Natural gas spot (Henry Hub) | 2023–2026 | EIA API (free key) | `[NULL]` | `src/ingest/external/gas_prices.py` ready |

## Target node — CONFIRMED

| Field | Value | Status |
|---|---|---|
| ERCOT settlement point | `RN_QTUM_SLR` | ✅ **Confirmed 2026-05-18** |
| Settlement-point type | Resource Node (`RN_` prefix) | Confirmed |
| Resource type | Solar PV (`_SLR` suffix per ERCOT naming) | Confirmed |
| Installed nameplate | **160 MW PV** | ✅ **Confirmed 2026-05-18** |
| Zone | ERCOT West Zone (West Texas) | Confirmed |
| Reference hub | `HB_WEST` (proxy if < 90d settlement history) | Config |
| Note | QTUM branding is shared with IP Quantum BESS battery (queue 26INR0309, Haskell County) — that is a separate settlement point and separate workstream |

## Data sourcing strategy (from PRD §11)

### LMP gap-fill (training) — gridstatus OSS library
- **Library:** `gridstatus` (open-source, free, no API key)
- **DAM SPP:** `ercot.get_spp(market='DAY_AHEAD_HOURLY', locations=['RN_QTUM_SLR', 'HB_WEST', 'HB_HUBAVG'])`
- **RTM SPP:** `ercot.get_spp(market='REAL_TIME_15_MIN', ...)`
- **Cached to:** `data/raw/gridstatus/dam_spp/`, `data/raw/gridstatus/rtm_spp/`
- **Module:** `src/ingest/live/gridstatus_client.py`

### 60-Day Disclosures (backtester) — ERCOT public archive
- **URL:** `https://data.ercot.com/data-product-archive/NP3-966-ER`
- **Size:** ~5 GB/year — pull in monthly chunks
- **No login required** (the "MIS LOG IN" link on ERCOT's page is for posting, not consuming)

### Weather — Iowa State ASOS (West Texas stations for RN_QTUM_SLR)
- **URL:** `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
- **Stations:** MAF (Midland, primary), LBB (Lubbock), SJT (San Angelo), plus DFW/HOU/AUS/SAT for zone features
- **No auth required**
- **Module:** `src/ingest/live/asos_weather.py`

### Natural gas spot — EIA Open Data API v2 (fuel cost proxy)
- **Endpoint:** `https://api.eia.gov/v2/natural-gas/pri/fut/data/`
- **Series:** `RNGWHHD` (Henry Hub spot, daily, $/MMBtu)
- **Module:** `src/ingest/external/gas_prices.py`
- **ERCOT basis:** approximate HSC = Henry Hub − $0.10/MMBtu

---

## Compliance status

All ingested local data is `[REAL]`. No `[SYNTHETIC]` price data exists in this project. Per CLAUDE.md §3, any feature that depends on a `[NULL]` upstream input propagates `[NULL]` and the daily runner skips that hour rather than fabricating values.

`RN_QTUM_SLR` price history depth is unknown until a gridstatus pull confirms how far back the node appears in ERCOT settlements. The backtest proxy falls back to `HB_WEST` until ≥90 days of node-specific settlement data are available.
