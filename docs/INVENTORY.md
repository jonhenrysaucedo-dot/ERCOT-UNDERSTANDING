# Data Inventory — `quantum_dart`

**Last updated:** 2026-05-18
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

## Missing (per PRD §M1 — must acquire in Phase 0)

| Dataset | Years needed | ERCOT Report | Tag | Action |
|---|---|---|---|---|
| DAM Settlement Point Prices | 2023–2026 | NP4-190-CD | `[NULL]` | **BLOCKER for target variable** — DART = RTM − DAM |
| RTM Settlement Point Prices (15-min) | 2023–2026 | NP6-905-CD | `[NULL]` | **BLOCKER for target variable** |
| 60-Day DAM Disclosure (EnergyOnlyOffers, EnergyBidAwards) | 2023–2026 | NP3-966-ER | `[NULL]` | Required for bid-stack slippage backtest (M9) |
| 60-Day SCED Disclosure | 2023–2026 | NP3-965-ER | `[NULL]` | Required for backtest |
| STWPF / WGRPP Wind Forecast | 2023–2026 | NP4-742-CD | `[NULL]` | Required for `wind_delta` feature |
| Weather temperature by zone | 2023–2026 | external | `[NULL]` | Plan: Open-Meteo historical archive (free, ERA5) |
| Natural gas spot (Henry Hub / HSC) | 2023–2026 | external | `[NULL]` | Plan: EIA Open Data API (free key) |

## Target node

| Field | Value | Status |
|---|---|---|
| Working label | `QUANTUM_ESR` | `[NULL]` — provisional |
| Best public match | IP Quantum BESS / Solace Storage (321.79 MW BESS, Haskell County TX, queue 26INR0309) | Unconfirmed |
| Zone | ERCOT North (adjacent West) | Confirmed by geography |
| Confirm via | `NP4-160-SG` Settlement Points List, ERCOT MIS | **PRD §10 open question #1** |

## Source plan for external data

### Natural gas (recommended: EIA Open Data API v2)

- **Endpoint:** `https://api.eia.gov/v2/natural-gas/pri/fut/data/`
- **Series:** `RNGWHHD` (Henry Hub spot, daily, $/MMBtu)
- **Latency:** T-1 business day
- **Key:** free signup at `eia.gov/opendata`
- **Fallback:** `https://www.eia.gov/dnav/ng/hist/rngwhhdd.csv` (no auth)
- **ERCOT basis:** approximate HSC = Henry Hub − $0.10/MMBtu (constant; refine when Platts data available)

### Temperature (recommended: Open-Meteo Historical Archive)

- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Variables:** `temperature_2m` (hourly, °C → convert to °F)
- **Source:** ERA5 reanalysis (NASA/ECMWF), 30-km grid
- **Latency:** T-1 day
- **Key:** none required
- **Zone mapping (lat, lon):**
  | ERCOT zone | Anchor city | Lat | Lon |
  |---|---|---|---|
  | North | Dallas (KDFW) | 32.90 | -97.04 |
  | Houston | Houston (KIAH) | 29.98 | -95.36 |
  | South | San Antonio (KSAT) | 29.53 | -98.47 |
  | West | Midland (KMAF) | 31.94 | -102.20 |
  | Coast | Corpus Christi (KCRP) | 27.77 | -97.50 |

---

## Compliance status

All ingested data here is `[REAL]`. No `[SYNTHETIC]` price data exists in this project. Per CLAUDE.md §136, any future feature that depends on a `[NULL]` upstream input must propagate `[NULL]` and the daily runner skips that hour rather than fabricating values.
