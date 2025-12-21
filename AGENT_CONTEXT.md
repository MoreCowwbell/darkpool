# Project Context

## Project Goal Summary
- Build a production-quality Python analytics project for end-of-day dark pool ETF analysis (Option B inference).
- Generate daily summary tables (estimated bought/sold, buy ratio, total off-exchange volume) and historical buy-ratio plots.
- Persist all raw and derived data in DuckDB and save outputs locally.

## Current Capabilities
- Requirements and plan captured; implementation is pending.

## Where Things Live (intended layout)
- `darkpool_analysis/`
- `darkpool_analysis/orchestrator.py`
- `darkpool_analysis/config.py`
- `darkpool_analysis/db.py`
- `darkpool_analysis/fetch_finra.py`
- `darkpool_analysis/fetch_polygon_equity.py`
- `darkpool_analysis/infer_buy_sell.py`
- `darkpool_analysis/analytics.py`
- `darkpool_analysis/plotter.py`
- `darkpool_analysis/data/darkpool.duckdb`
- `darkpool_analysis/output/tables/`
- `darkpool_analysis/output/plots/`
- `darkpool_analysis/.env`
- `darkpool_analysis/README.md`

## Execution Flow (per scheduler tick)
- Load config and environment variables (tickers, date ranges, MIN_LIT_VOLUME, RTH filter).
- Fetch FINRA OTC weekly data; abort on failure; write `finra_otc_volume_raw`.
- Fetch Polygon equity trades for eligible tickers; warn on partial failure; write `equity_trades_raw`.
- Classify lit trades and compute lit buy ratios; write `equity_lit_directional_flow`.
- Apply lit ratios to FINRA volume; write `darkpool_estimated_flow`.
- Build `darkpool_daily_summary`, export tables, and generate plots from DuckDB only.

## Data and Storage Details
- DuckDB is the single source of truth: `darkpool_analysis/data/darkpool.duckdb`.
- Raw tables: `finra_otc_volume_raw`, `equity_trades_raw`.
- Derived tables: `equity_lit_directional_flow`, `darkpool_estimated_flow`, `darkpool_daily_summary`.
- Store float64 values; round only for display/export.

## External Integrations
- FINRA OTC Transparency (ATS + non-ATS) equity volume; authoritative weekly dataset.
- Polygon equity trade prints for lit-market proxy (NBBO if available; otherwise TICK rule).
- Options flow is out of scope; no inferred trade direction from options or short-sale files.

## Deployment and Operations
- Local setup: `python -m pip install -r requirements.txt`, copy `.env`, then run `python orchestrator.py`.
- Daily run after market close; weekly run Sunday after FINRA settlement.

## Known Gaps and Watchouts
- SPXW is not an equity ticker and must be excluded from FINRA volume tables.
- If lit volume is below MIN_LIT_VOLUME, set buy ratio to NULL and persist coverage metadata.
- Daily outputs must reference the most recent settled FINRA week.

## Tips for Future Contributors
- Keep the mandatory disclaimer in README and code comments: FINRA does not publish trade direction; buy/sell values are inferred estimates.
- Avoid hidden assumptions; document inference logic and coverage limits explicitly.
- Update `AGENT_CHANGELOG.md` after each session with the required checkbox format.
