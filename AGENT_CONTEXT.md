# Project Context

## Project Goal Summary
- Build a production-quality Python analytics project for end-of-day dark pool ETF analysis (Option B inference).
- Generate daily summary tables (estimated bought/sold, buy ratio, total off-exchange volume) and historical buy-ratio plots.
- Persist all raw and derived data in DuckDB and save outputs locally.

## Current Capabilities
- Full implementation complete with performance optimizations.
- Parallel Polygon API fetching with aggregates fallback (when trades API returns 403).
- Proper connection management and consistent buy-ratio formulas.
- FINRA direct API key authentication (no OAuth token needed).
- Multi-date fetch modes: "single", "daily" (last N trading days), "weekly" (last N Fridays).
- FINRA tier tracking via `tierIdentifier` field (T1=NMS Tier 1, T2=NMS Tier 2, OTC).

## Where Things Live
- `main.py` — Root entry point (can run from project root)
- `darkpool_analysis/` — Package directory
- `darkpool_analysis/__init__.py` — Package init (v0.1.0)
- `darkpool_analysis/orchestrator.py` — Main orchestration logic
- `darkpool_analysis/config.py` — Configuration loader from .env
- `darkpool_analysis/db.py` — DuckDB connection and upsert helpers
- `darkpool_analysis/fetch_finra.py` — FINRA OTC data fetcher
- `darkpool_analysis/fetch_polygon_equity.py` — Polygon trades fetcher (parallel)
- `darkpool_analysis/infer_buy_sell.py` — Lit trade classification (NBBO/TICK)
- `darkpool_analysis/analytics.py` — Darkpool estimates and summary builders
- `darkpool_analysis/plotter.py` — Buy ratio time-series plots
- `darkpool_analysis/data/darkpool.duckdb` — Persistent storage
- `darkpool_analysis/output/tables/` — CSV exports
- `darkpool_analysis/output/plots/` — PNG plots
- `.env` — Environment secrets (API keys only, at project root)
- `darkpool_analysis/config.py` — All default configuration (URLs, tickers, dates)
- `darkpool_analysis/requirements.txt` — Python dependencies

## Execution Flow (per scheduler tick)
- Load config and environment variables (tickers, date ranges, MIN_LIT_VOLUME, RTH filter).
- Fetch FINRA OTC weekly data; abort on failure; write `finra_otc_volume_raw`.
- Fetch Polygon equity trades for eligible tickers; warn on partial failure; write `equity_trades_raw`.
- Classify lit trades and compute lit buy ratios; write `equity_lit_directional_flow`.
- Apply lit ratios to FINRA volume; write `darkpool_estimated_flow`.
- Build `darkpool_daily_summary`, export tables, and generate plots from DuckDB only.

## Data and Storage Details
- DuckDB is the single source of truth: `darkpool_analysis/data/darkpool.duckdb`.
- Raw tables: `finra_otc_volume_raw` (includes `source` column for tier: T1/T2/OTC), `equity_trades_raw`.
- Derived tables: `equity_lit_directional_flow`, `darkpool_estimated_flow`, `darkpool_daily_summary`.
- Store float64 values; round only for display/export.

## External Integrations
- FINRA OTC Transparency (ATS + non-ATS) equity volume; authoritative weekly dataset.
- Polygon equity trade prints for lit-market proxy (NBBO if available; otherwise TICK rule).
- Options flow is out of scope; no inferred trade direction from options or short-sale files.

## Deployment and Operations
- Local setup: `pip install -r darkpool_analysis/requirements.txt`, configure `.env` with API keys, then run from root.
- Run from root: `python main.py`
- Run from subdirectory: `cd darkpool_analysis && python orchestrator.py`
- Daily run after market close; weekly run Sunday after FINRA settlement.
- Set `POLYGON_MAX_WORKERS` env var to control parallel API threads (default: 4).

## Configuration Structure
- **`.env` (project root)**: Only secrets (POLYGON_API_KEY, FINRA_API_KEY, FINRA_API_SECRET).
- **`config.py`**: All defaults (DEFAULT_TICKERS, DEFAULT_TARGET_DATE, DEFAULT_FETCH_MODE, URLs, etc.).
- **Fetch modes**: `FETCH_MODE` controls both date selection and snapshot behavior:
  - `"single"`: Process only target_date, snapshot keyed by target_date.
  - `"daily"`: Process last N trading days (BACKFILL_COUNT), snapshot keyed by target_date.
  - `"weekly"`: Process last N Fridays (BACKFILL_COUNT), snapshot keyed by FINRA week.
- Polygon trades API requires paid tier; code auto-falls back to aggregates (minute bars) on 403.
- DEFAULT_TARGET_DATE should be set to a trading day (not weekend/holiday).

## Known Gaps and Watchouts
- SPXW is not an equity ticker and must be excluded from FINRA volume tables.
- If lit volume is below MIN_LIT_VOLUME, set buy ratio to NULL and persist coverage metadata.
- Daily outputs must reference the most recent settled FINRA week.

## Tips for Future Contributors
- Keep the mandatory disclaimer in README and code comments: FINRA does not publish trade direction; buy/sell values are inferred estimates.
- Avoid hidden assumptions; document inference logic and coverage limits explicitly.
- Update `AGENT_CHANGELOG.md` after each session with the required checkbox format.
