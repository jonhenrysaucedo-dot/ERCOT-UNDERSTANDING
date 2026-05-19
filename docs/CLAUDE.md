# CLAUDE.md — Project guidance for Claude Code

This file is **the first thing Claude Code reads** when working in this repository. Update it whenever conventions change.

---

## What this project is

A daily-DAM bidding system for a single ERCOT solar resource node — **`RN_QTUM_SLR`** in the West Zone. It produces a 10-tier limit-price bid curve for each of the 24 operating hours of the next day, sized by a Bayesian-Kelly framework over a probabilistic DART-spread forecast, with a solar-curtailment hedge overlay that adds INC virtuals when the posterior implies RT will go deeply negative.

**The system does not submit bids to ERCOT.** It writes a CSV that a human/QSE reviews and submits. Do not add any submission logic.

---

## Required reading (in this order)

1. **`docs/PRD.md`** — the product requirements. This is the authoritative scope document. Anything in this file (`CLAUDE.md`) that contradicts the PRD is a bug — fix `CLAUDE.md`.
2. **`docs/NOTERBOOK_LLM_1.docx`** — the design whitepaper. Explains *why* the architecture is what it is.
3. **`docs/INVENTORY.md`** — what data has been ingested and what's still missing.
4. **The relevant module's `README.md`** before editing it.

---

## Tech stack — non-negotiable choices

These are pinned in `pyproject.toml`. Do not swap them for "modern alternatives" without explicit approval — they were chosen deliberately to match the whitepaper.

| Layer | Library | Why this and not alternatives |
|---|---|---|
| Regime detection | `hmmlearn` | Whitepaper-specified. `pomegranate` is faster but has API churn |
| Volatility | `arch` | Whitepaper-specified. Standard GARCH implementation; `statsmodels` ARCH is buggier |
| Bayesian inference | `PyMC` + `nutpie` sampler | Whitepaper-specified. `numpyro` is faster but the team's mental model is PyMC |
| Backtesting | `vectorbt` | Whitepaper-specified. Vectorized speed matters for walk-forward |
| ERCOT data feed | `gridstatus` (open-source) | Free, no API key. Pulls DAM/RTM SPPs, AS, load, wind/solar forecast, fuel mix from ERCOT public reports. Used for both training (LMP gap-fill) and inference (everything) |
| Data store | Parquet via `pyarrow` | Columnar + range queries; no databases in v1 |
| Dataframes | `polars` for ingestion, `pandas` for modeling | Polars is faster for joins; PyMC ergonomics still favor pandas |
| Config | YAML via `pydantic-settings` | Type-checked configs prevent silent misconfiguration |
| Logging | `structlog` (JSON) | The audit trail in §5 of the PRD requires structured logs |

Python: 3.11+. Use `uv` for env management (much faster than poetry on this repo size).

---

## Repo layout (canonical)

```
quantum_dart/
├── CLAUDE.md                  # YOU ARE HERE
├── README.md                  # External-facing summary
├── docs/                      # PRD, whitepaper, inventory
├── pyproject.toml
├── uv.lock
├── .env.example
├── config/                    # YAML configs only — no .py files here
│   ├── scoring.yaml
│   ├── risk.yaml
│   └── nodes.yaml
├── data/
│   ├── raw/                   # IMMUTABLE — never edit files here
│   ├── processed/             # Parquet store, regenerable from raw/
│   └── external/              # Weather, third-party
├── src/
│   ├── ingest/                # ERCOT MIS pulls, parsers
│   ├── features/              # Feature engineering
│   ├── models/                # hmm.py, garch.py, bayesian_nuts.py
│   ├── scoring/               # composite.py
│   ├── sizing/                # kelly.py
│   ├── execution/             # tier_generator.py
│   ├── backtest/              # vectorbt harness, bid-stack reconstruction
│   └── runners/               # daily.py CLI entry point
├── tests/                     # mirrors src/ structure
├── notebooks/                 # exploratory ONLY, never CI
└── output/                    # bids and audit JSON per run date
```

**Rule:** new code goes in the module that matches its purpose. If you can't decide between two modules, the code probably needs to be split.

---

## Commands

```bash
# Setup (one time)
uv sync

# Run the daily bid generator for tomorrow
uv run python -m src.runners.daily --as-of $(date -d 'tomorrow' +%Y-%m-%d)

# Run a backtest fold
uv run python -m src.backtest.run --train-start 2024-01-01 --train-end 2024-12-31 --test-start 2025-01-01 --test-end 2025-01-31

# Tests
uv run pytest                           # all tests
uv run pytest tests/features/ -v        # one module
uv run pytest -k "kelly" -v             # by keyword
uv run pytest --cov=src --cov-report=term-missing

# Lint + type-check
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/

# Re-ingest a year of data
uv run python -m src.ingest.ercot --year 2025
```

---

## Code conventions

1. **No magic numbers in code.** If a value isn't obvious from context, put it in a YAML config or a named constant. The threshold "credible interval width > 1.0" is a config knob, not a literal.
2. **Every function that touches time has a timezone-aware UTC contract.** Naive datetimes raise. ERCOT's CT timestamps get converted at the ingest boundary and never re-leak.
3. **Every feature builder accepts an `as_of_timestamp` argument.** This enforces walk-forward at the type level. A function without it cannot be in `src/features/`.
4. **No silent failures.** If data is missing, raise a typed exception (`MissingDataError`, `StaleDataError`). Do not return empty DataFrames or NaN-fill.
5. **Pydantic models for all data interfaces.** Functions between modules pass typed `BaseModel` instances, not loose dicts.
6. **Docstrings are mandatory on public functions.** Numpy-style. Include a `Walk-forward safety:` line explaining what the function does and doesn't see.
7. **Random seeds are explicit.** Every random source (numpy, PyMC, sklearn, hmmlearn) reads `seed` from `config/runtime.yaml`. RNG state is logged with every run.

---

## Data conventions

1. **Canonical timestamp:** `interval_start_utc` (column name). Type `datetime64[ns, UTC]`. Hour-ending data is converted to hour-starting at ingest.
2. **Canonical node key:** `settlement_point` (column name). String, uppercase, matches ERCOT MIS exactly.
3. **Filename convention for raw ERCOT downloads:** preserve the original ERCOT filename. Do not rename.
4. **Parquet partitioning:** `data/processed/<dataset>/year=<YYYY>/month=<MM>/data.parquet`
5. **Schema migrations:** changes to processed Parquet require a `data/processed/SCHEMA.md` entry and a re-ingest script.
6. **Settlement Type semantics for fuel mix data:** `FINAL` is true once-and-for-all. `INITIAL` is preliminary and *will* be overwritten on re-ingest. Tag and version both.
7. **Training data lives in `data/raw/local/`** (user's uploaded files, immutable). **Live/inference data lives in `data/cache/gridstatus/`** (fetched on demand, can be deleted and re-pulled). Never mix them in the same Parquet file without an explicit `source` column.
8. **Every fetched gridstatus result is cached.** Use `joblib` or `diskcache` keyed by (method, start, end, locations). Reruns of the daily script must not re-pull data already in cache. Cache invalidation is manual via `python -m src.ingest.live.invalidate_cache`.
9. **Vintage tracking for inference data is mandatory.** Every gridstatus call records the `publish_time` (or pull timestamp if unavailable) into a sidecar JSON. This is what proves walk-forward compliance.

---

## Compliance rules (override everything else)

These come from Jonathan's `alpha-signal-validator` skill. Violations are blocking.

1. **Every dataset is tagged `[REAL]`, `[SYNTHETIC]`, or `[NULL]`.** Tag stored as column metadata in the Parquet file.
2. **No synthetic price data is permitted in this strategy.** Backtests use only real, settled ERCOT prices. Simulated stress scenarios may exist in `notebooks/` for exploration but never feed into model training or risk metrics.
3. **`[NULL]` is a valid answer.** If data is missing and no real substitute exists, the model output is `[NULL]` and the daily runner skips that hour rather than fabricating a number.
4. **Compliance tags propagate.** A feature derived from `[REAL]` and `[NULL]` inputs is `[NULL]`. Don't silently downgrade.

---

## Things NOT to do (in order of severity)

1. **Do not write code that submits bids to ERCOT.** v1 is human-in-the-loop. Submission is an entirely separate workstream with separate governance.
2. **Do not hard-code the target node name.** Read it from `config/nodes.yaml`. The current node is `RN_QTUM_SLR`, but the framework must remain node-agnostic for v2 multi-node expansion.
3. **Do not bypass the walk-forward boundary.** Any feature that uses RTM data to predict the same day's DAM is a bug, even if it improves metrics. Especially if it improves metrics.
4. **Do not use synthetic prices in backtests.** See compliance rules.
5. **Do not skip the Half-Kelly damping.** Full-Kelly sizing is mathematically correct for known distributions and operationally catastrophic for forecasted ones. The 0.5 multiplier is non-negotiable.
6. **Do not edit files in `data/raw/`.** They're the audit trail. Reprocess from raw, don't mutate raw.
7. **Do not commit `.env` or any file containing ERCOT MIS credentials.** `.gitignore` is already set; verify before pushing.
8. **Do not refactor `src/models/bayesian_nuts.py` to use a different sampler without re-running the calibration tests.** PyMC + nutpie is the validated stack.
9. **Do not use `pandas.read_excel` on the 2007–2015 fuel-mix files without `engine='xlrd'`.** They're legacy `.xls` and openpyxl rejects them.

---

## Where context lives when you can't figure something out

- **Why is the architecture this way?** → `docs/NOTERBOOK_LLM_1.docx`
- **What's the scope of v1?** → `docs/PRD.md`
- **What ERCOT data exists locally?** → `docs/INVENTORY.md`
- **What's the column schema for X?** → `data/processed/SCHEMA.md`
- **Why does a test exist?** → test docstring; if missing, that's a bug, add one
- **What did the last daily run produce?** → `output/audit_YYYYMMDD.json`

---

## When in doubt

- Prefer the boring option. This is a money-handling system; novelty is risk.
- If the PRD is silent, ask before assuming. Open a question at the bottom of `docs/PRD.md` §10.
- Walk-forward correctness > model accuracy > code elegance.
- Honest negative results > optimistic positive results that leaked future data.
