# PRD — ERCOT DART Trading Strategy for `QUANTUM_ESR` (West Texas)

**Owner:** Jonathan
**Status:** Draft v1.0 — Phase 0 not started
**Last updated:** 2026-05-18
**Reference design:** `NOTERBOOK_LLM_1.docx` (in same `/docs/` folder) — this PRD operationalizes that whitepaper

---

## 0. TL;DR

Build a Python system that submits Day-Ahead Market virtual bids and PTP Obligations for a single ERCOT resource node (a battery storage facility associated with IP Quantum BESS, exact ERCOT-registered settlement-point name **TBD**, see §10). Bids are sized by a Bayesian-Kelly framework and structured as 10-tier limit curves — never as price-taker blocks.

The system is **not** a live execution engine in v1. v1 produces a daily CSV of bids that a human (or QSE) reviews and submits manually. Automated submission is out of scope.

---

## 1. Problem & background

ERCOT DART (Day-Ahead minus Real-Time price) spreads are the primary alpha vector for financial market participants holding physical renewables and batteries. The standard "price taker" approach — offering INC supply at the price floor or DEC demand at the cap — guarantees clearance but locks the trader into whatever DAM price prints, which is increasingly poor as solar-suppressed midday prices have collapsed margins.

The whitepaper proposes a fix: rank each operating hour by a Composite Score (directional conviction × spread magnitude × fundamental alignment), size positions using a continuous Kelly criterion over an ML-forecasted distribution, then translate the Kelly MW into a 10-tier limit-price bid stack. The Composite Kelly approach beats Distributionally Robust Optimization (DRO/CVaR) on operational practicality because it produces explainable, tier-by-tier outputs that map directly to ERCOT's API submission schema.

This PRD operationalizes that design for a single node and a single trading day at a time.

---

## 2. Goals (must-have)

1. **G1** — Ingest, clean, and align ERCOT historical data into a single tidy time-series store keyed by `(settlement_point, interval_start_utc)`.
2. **G2** — Produce a probabilistic forecast of the DART spread at the target node for each of the 24 operating hours of the next day, with explicit 10/50/90 percentiles.
3. **G3** — Output a daily `bids.csv` containing a 10-tier limit-price/quantity curve for each hour, separated into virtual energy bids (INC/DEC) and PTP Obligation bids where applicable.
4. **G4** — Provide an audit trail: every bid must be traceable to (a) the feature values that drove it, (b) the regime state, (c) the Kelly fraction before and after Half-Kelly damping, (d) the covariance haircut.
5. **G5** — Backtest the strategy against historical 60-Day Disclosure data with realistic bid-stack-aware slippage modeling.

## 2.1 Non-goals (explicit out-of-scope for v1)

- ❌ Automated bid submission to ERCOT (manual paste by QSE only)
- ❌ Multi-node portfolio optimization (one node at a time)
- ❌ Real-time intraday re-bidding (DAM-only workflow)
- ❌ Ancillary service capacity bidding (energy + PTP only)
- ❌ Battery state-of-charge co-optimization (treat physical battery as fixed reference; financial trades only)
- ❌ DRO/CVaR alternative implementation (compared and rejected in whitepaper)

---

## 3. Target node

| Field | Value |
|---|---|
| Provisional name | `QUANTUM_ESR` (working label, **NOT verified**) |
| Best public match | IP Quantum BESS, LLC — "Solace Storage" (321.79 MW BESS, Haskell County, TX, ERCOT queue `26INR0309`, COD projected Jun 2026) |
| Geographic zone | ERCOT North Zone (Haskell County borders the West Zone) |
| Settlement-point type | Resource Node (`RN`) |
| **Action required (Phase 0)** | Confirm the ERCOT-registered settlement-point name by pulling the `NP4-160-SG` Settlement Points List from the ERCOT MIS. Until confirmed, all code must reference the node via the env var `TARGET_NODE`, never a hard-coded string. |

---

## 4. Functional requirements (modules)

### M1 — Data ingestion (`src/ingest/`)
Pull and normalize:
| Source | Status | ERCOT report ID |
|---|---|---|
| Native Load by Zone | ✅ have 2023–partial 2026 | `np6-345-cd` |
| Hourly Wind/Solar Actual | ✅ have 2023–2025 | varies |
| DAM AS MCPC | ✅ have 2024–partial 2026 | `np4-188-cd` |
| Interval Gen by Fuel (15-min) | ✅ have 2007–2026 | `np4-732-cd` |
| Hourly PV Forecast (PVGRPP) | ✅ have May 2026 snapshots only | `np4-743-cd` |
| **DAM Settlement Point Prices** | ❌ MISSING | `np4-190-cd` |
| **RTM Settlement Point Prices (15-min)** | ❌ MISSING | `np6-905-cd` |
| **STWPF / WGRPP Wind Forecast** | ❌ MISSING | `np4-742-cd` |
| **60-Day DAM Disclosure (EnergyOnlyOffers, EnergyBidAwards)** | ❌ MISSING | `np3-966-er` |
| **60-Day SCED Disclosure** | ❌ MISSING | `np3-965-er` |
| **Weather (temperature by zone)** | ❌ MISSING | external (NOAA / Iowa State ASOS) |

**Requirement:** All ingested data is rewritten to a single canonical Parquet store under `data/processed/`, partitioned by year, with `interval_start_utc` (timezone-aware UTC) as the index. No raw timestamps in CT/CST/CDT past the ingestion layer.

**DST handling:** The ERCOT `Repeated Hour Flag` column must be honored — the fall-back hour repeats and the spring-forward hour is skipped. Document the convention in `src/ingest/timestamps.md`.

### M2 — Feature engineering (`src/features/`)
Outputs an hourly feature matrix at the target node.

| Feature | Formula | Notes |
|---|---|---|
| `dart_spread_t` | `RTM_LMP_15min` time-weight-averaged to hour − `DAM_LMP_hourly` | Target variable |
| `net_load` | `ERCOT_LOAD` − `WIND_GEN` − `SOLAR_GEN` | Built from existing data |
| `nlfe` (Net Load Forecast Error) | DA net load forecast − RT net load (intra-day update) | Requires forecast pull |
| `solar_delta` | PVGRPP day-ahead − actual PV output | Critical at sunset |
| `wind_delta` | STWPF day-ahead − actual wind output | Requires forecast pull |
| `thermal_share` | (Coal + Gas + Gas-CC + Nuclear) ÷ Total Gen | From fuel_mix; HMM regime feature |
| `temp_hinge_hot` | `max(0, temp − 90°F)` | Piecewise activation |
| `temp_hinge_cold` | `max(0, 30°F − temp)` | Piecewise activation |
| `hour_sin`, `hour_cos` | Fourier hour-of-day encoding | k=1,2 harmonics |
| `dow_sin`, `dow_cos` | Fourier day-of-week | k=1 |
| `as_total_capacity` | RegUp + RegDn + RRS + NSpin + ECRS (DAM MCPC) | RTC+B opportunity-cost proxy |
| `ecrs_premium` | ECRS MCPC ÷ (RegUp MCPC) | Detects ECRS-distortion regime (2023–2025) |

**Critical rule:** Every feature must respect the DAM submission deadline (10:00 CT on prior day). No feature may use information unavailable at that moment. Walk-forward validation only — enforced in code by an `as_of_timestamp` argument that all feature builders accept.

### M3 — Regime detection (`src/models/hmm.py`)
- Library: `hmmlearn.hmm.GaussianHMM`
- States: 3 — `NORMAL`, `SCARCITY`, `NEGATIVE_CONGESTION`
- Inputs: standardized DART spread, thermal_share, nlfe, solar_delta, wind_delta
- Training window: 2 years rolling, refit weekly
- **Output:** per-hour state probability vector P(s ∈ {0,1,2}) for the next operating day

### M4 — Volatility (`src/models/garch.py`)
- Library: `arch` package — `arch_model(mean='Constant', vol='GARCH', p=1, q=1)`
- Variant: Markov-Switching GARCH — variance conditionally scaled by HMM state probability
- Output: hour-specific σ²_t for the DART spread

### M5 — Bayesian forecaster (`src/models/bayesian_nuts.py`)
- Library: `PyMC` with `pm.sample(..., nuts_sampler='nutpie')` (10× faster than default NUTS)
- Model: linear regression on the feature matrix, weakly-informative priors (Normal(0, 1) on standardized coefficients, HalfNormal(1) on σ)
- Sampling: 4 chains, 2000 draws, 1000 tune
- **Output:** posterior trace → P(R) for each hour. Extract q10, q50, q90 percentiles.

### M6 — Composite Score (`src/scoring/composite.py`)
For each hour h:
- `directional_conviction_h` = P(spread > 0) for INC, P(spread < 0) for DEC, computed from posterior trace
- `spread_magnitude_h` = |E[spread]| ÷ historical σ(spread) at this hour
- `fundamental_alignment_h` = ∈ [0, 1] from outage schedule + transmission heuristics (binary in v1: 1 if no major export-restricting outage, 0 otherwise)
- `composite_h` = `w₁ · directional_conviction + w₂ · normalize(spread_magnitude) + w₃ · fundamental_alignment`
- Weights (`w₁=0.5, w₂=0.3, w₃=0.2`) configurable in `config/scoring.yaml`

### M7 — Continuous Kelly sizer (`src/sizing/kelly.py`)
- Formula: `f* = argmax_f ∫ log(1 + f·R) · P(R) dR`
- Solver: 1D numerical optimization over f ∈ [0, 0.5] (Half-Kelly cap)
- Penalty: if posterior credible interval width / |E[spread]| > 1.0 (high uncertainty), multiply f* by 0.5
- Covariance haircut: if multiple hours are highly correlated in posterior, scale f* down by (1 - max_pairwise_correlation)
- Output: MW allocation per hour, must respect portfolio max-position constraint from `config/risk.yaml`

### M8 — 10-tier bid generator (`src/execution/tier_generator.py`)
Input: Kelly MW total for hour h, posterior distribution
Output: 10 (price, quantity) pairs, monotonically increasing in price for INC and decreasing for DEC.
- Tiers 1–3 (Base, ~30% of volume): prices spread tightly around q50 of DAM expected price
- Tiers 4–8 (Scaling, ~50%): prices spread between q50 and q90 (for INC) of posterior
- Tiers 9–10 (Tail, ~20%): extreme limit prices beyond q90 — these only clear if DAM diverges anomalously

Constraint: ERCOT requires monotonic price/quantity pairs. Validate before write.

### M9 — Backtester (`src/backtest/`)
- Library: `vectorbt` for vectorized P&L
- Bid-stack reconstruction: from 60-Day DAM Disclosure (when available), rebuild the historical supply/demand curve at each settlement point.
- Slippage model: inject our 10-tier curve into the historical stack and recompute clearing price (exogenous quantity shock).
- Per-tier clearance reporting: which of the 10 tiers cleared, which didn't, and at what MCP.
- Walk-forward: rolling 1-year train / 1-month test, 12 folds minimum across 2024–2025.
- **Regime-aware metrics:** Report Sharpe, Sortino, Calmar, max DD, hit rate separately for each HMM state.

### M10 — Daily bid runner (`src/runners/daily.py`)
- CLI entrypoint: `python -m runners.daily --as-of 2026-05-19`
- Pulls latest features, runs M3 → M4 → M5 → M6 → M7 → M8
- Writes `output/bids_YYYYMMDD.csv` + `output/audit_YYYYMMDD.json`
- Logs every input feature value, the regime probabilities, Kelly fraction pre/post damping, and the tier curve.

---

## 5. Non-functional requirements

| Requirement | Spec |
|---|---|
| **Reproducibility** | Every run pins `numpy`, `pymc`, `arch`, `hmmlearn`, `vectorbt` to exact versions in `pyproject.toml`. RNG seed is logged. |
| **Data freshness** | Daily run must complete in < 30 min on a single 16-core CPU. PyMC sampling is the bottleneck; use `nutpie` and parallel chains. |
| **Audit trail** | All decisions logged to a SQLite `audit.db` with foreign keys: `runs → features → predictions → bids → settlements`. |
| **Testing** | Every module in `src/` has matching `tests/`. Minimum 70% coverage on `src/features/` and `src/sizing/` (correctness matters most there). |
| **Compliance tagging** | Per Jonathan's `alpha-signal-validator` skill: every dataset must be tagged `[REAL]`, `[SYNTHETIC]`, or `[NULL]`. **No synthetic price data is permitted in this strategy**, period — backtests must use only real historical ERCOT settlements. |
| **Secrets** | ERCOT MIS credentials and weather API keys live in `.env`, never committed. `.env.example` documents required vars. |

---

## 6. Architecture (target file layout)

```
quantum_dart/
├── CLAUDE.md                  # Claude Code's project guide (see separate file)
├── README.md                  # Human-readable summary
├── pyproject.toml             # Pinned deps; uses uv or poetry
├── .env.example
├── config/
│   ├── scoring.yaml           # Composite Score weights
│   ├── risk.yaml              # Max position MW, Half-Kelly multiplier
│   └── nodes.yaml             # Target node name + zone + neighboring nodes
├── data/
│   ├── raw/                   # Original ERCOT downloads (immutable)
│   ├── processed/             # Canonical Parquet store
│   └── external/              # Weather data
├── src/
│   ├── ingest/
│   ├── features/
│   ├── models/                # hmm.py, garch.py, bayesian_nuts.py
│   ├── scoring/
│   ├── sizing/
│   ├── execution/
│   ├── backtest/
│   └── runners/
├── tests/
├── notebooks/                 # Exploratory only; not in CI
└── output/
    ├── bids_YYYYMMDD.csv
    └── audit_YYYYMMDD.json
```

---

## 7. Phased delivery

| Phase | Scope | Acceptance | Estimated effort |
|---|---|---|---|
| **0. Confirm & acquire** | Verify ERCOT-registered node name; pull missing data sources (LMP, 60-Day Disclosure, wind forecast, weather) | `data/raw/` contains all sources in §M1; `config/nodes.yaml` has confirmed node name | 1–2 weeks (bottleneck: ERCOT MIS approval if not already in hand) |
| **1. Ingest** | Build M1; produce a clean Parquet store | `pytest tests/ingest/` green; one DataFrame per source, joinable on `(node, interval_start_utc)` | 1 week |
| **2. Features** | Build M2; all features computable from historical data | Feature matrix for 2024–2025 produced; walk-forward boundary respected (tested) | 1 week |
| **3. Models** | Build M3 (HMM), M4 (GARCH), M5 (PyMC) | Each model trains on 2024 data and produces sensible predictions on Jan 2025 holdout. Posterior credible intervals contain truth ≥85% of the time | 2–3 weeks |
| **4. Scoring + Sizing** | Build M6 (Composite), M7 (Kelly), M8 (tier gen) | For a known-good test day, the generated bid stack is monotonic, sums to the Kelly MW, and respects the risk-config caps | 1 week |
| **5. Backtest** | Build M9 against 60-Day Disclosure | Walk-forward 2024–2025 backtest produces a tearsheet (use `backtest-report-generator` skill). Reports Sharpe by regime. **Demonstrates positive net-of-slippage P&L on out-of-sample folds, or returns honest negative result.** | 2 weeks |
| **6. Daily runner** | Build M10 | Cron-scheduled, produces bids.csv at 08:00 CT, ready for QSE review by 09:00 CT (DAM submission deadline is 10:00 CT) | 1 week |
| **7. Paper trade** | Run daily for 30+ days, log every "would have submitted" without actually submitting | Realized backtest P&L vs paper P&L match within 10%; manual reviewer reports the rationale was clear | 30 days |
| **8. Live (separate PRD)** | Out of scope for this document | — | — |

---

## 8. Risks & failure modes

1. **Node never gets confirmed or never energizes.** Quantum BESS / Solace's COD was projected June 2026. If physical interconnection slips, there's no resource node to trade. Mitigation: framework should work for any West/North zone resource node — make the target node a config parameter, not a hard-coded constant.
2. **60-Day Disclosure unavailable.** Without it, the backtest can only use top-of-book DAM prices and no slippage model. The strategy is still runnable as a forward paper trade, but historical alpha cannot be validated rigorously. Mitigation: document this as a known gap and proceed with paper trading only until disclosure data is in hand.
3. **RTC+B regime change invalidates pre-Dec-2025 training data.** The whitepaper flags this explicitly. Mitigation: HMM training window starts no earlier than Dec 2025 once 6+ months of post-RTC+B data exists. Until then, use 2023–2025 with a regime-aware indicator.
4. **PyMC sampling too slow for 24-hour daily run.** Mitigation: `nutpie` backend; if still too slow, fall back to variational inference (`pm.fit()`) which trades calibration for speed.
5. **Catastrophic tail event invalidates Kelly distribution assumption.** Winter Storm Uri-class events make the posterior structurally wrong. Mitigation: hard MW caps in `config/risk.yaml` that override Kelly output. The Composite Kelly framework is *not* a substitute for absolute position limits.
6. **Walk-forward leakage from joins.** Most common bug. Mitigation: every feature accepts an `as_of_timestamp` and tests assert no future data was joined.

---

## 9. Success metrics (review at end of paper trade phase)

- **Primary:** Sharpe ratio (gross of slippage, then net) ≥ 1.5 on the paper trade period
- **Secondary:** Max drawdown < 15% of allocated capital
- **Tertiary:** Tier-clearance rate — proportion of generated tiers that actually clear in DAM. Target: 30–70%. Higher means tiers are priced too aggressively; lower means the model is mis-calibrated.

---

## 10. Open questions (must resolve before Phase 1)

1. **What is the exact ERCOT-registered settlement-point name for the target node?** Action: Jonathan or QSE pulls `NP4-160-SG` from ERCOT MIS, filters by `RESOURCE_NAME` containing "QUANTUM" or "QTUM" or "SOLACE" or "IPQ", and confirms.
2. **What is the trader's capital allocation for this strategy?** Drives the `max_position_mw` in `config/risk.yaml`.
3. **Does the QSE support 10-tier curve trades on virtuals, or only block trades?** If only blocks, M8 degrades to a 1-tier output and a different sizing logic is needed.
4. **Is the target node already energized as of the run date?** A valid-but-not-energized node will fail ERCOT pre-trade validation.
5. **Is the trader running ancillary service co-bids on this node?** If yes, M10's bid generator must reserve battery capacity for AS, reducing the financial trade size. Out of scope for v1 per §2.1, but flag for v2.
