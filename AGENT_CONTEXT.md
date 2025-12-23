# Project Context

## Project Goal Summary
- Build a production-quality Python analytics project for end-of-day dark pool ETF analysis (Option B inference).
- Generate daily summary tables (estimated bought/sold, buy ratio, total off-exchange volume) and historical log-ratio plots.
- Persist all raw and derived data in DuckDB and save outputs locally.

## Current Capabilities
- Full implementation complete with performance optimizations.
- Parallel Polygon API fetching with aggregates fallback (when trades API returns 403).
- Proper connection management and consistent buy-ratio formulas.
- FINRA direct API key authentication (X-API-KEY/X-API-SECRET headers; OAuth optional).
- Multi-date fetch modes: "single", "daily" (last N trading days), "weekly" (last N Fridays).
- FINRA tier tracking via `tierIdentifier` field (T1=NMS Tier 1, T2=NMS Tier 2, OTC).
- Dark-theme HTML/PNG table renderer with signal annotations (BOT/SELL).
- Log-ratio time-series plots with volume-scaled markers.

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
- `darkpool_analysis/plotter.py` — Log-ratio time-series plots (CLI: `python plotter.py`)
- `darkpool_analysis/table_renderer.py` — HTML/PNG table generator (CLI: `python table_renderer.py`)
- `darkpool_analysis/data/darkpool.duckdb` — Persistent storage
- `darkpool_analysis/output/tables/` — CSV and HTML/PNG exports
- `darkpool_analysis/output/plots/` — PNG plots
- `.env` — Environment secrets (API keys only, at project root)
- `darkpool_analysis/requirements.txt` — Python dependencies

## Execution Flow (per scheduler tick)
1. Load config and environment variables (tickers, date ranges, MIN_LIT_VOLUME, RTH filter).
2. For each target date:
   - Fetch FINRA OTC weekly data; abort on failure; write `finra_otc_volume_raw`.
   - Fetch Polygon equity trades for eligible tickers; warn on partial failure; write `equity_trades_raw`.
   - Classify lit trades and compute lit buy ratios; write `equity_lit_directional_flow`.
   - Apply lit ratios to FINRA volume; write `darkpool_estimated_flow`.
   - Build `darkpool_daily_summary`; optionally export CSVs.
3. After all dates processed:
   - Generate log-ratio plots from DuckDB.
   - Render combined HTML/PNG table for all processed dates.

## Data and Storage Details
- DuckDB is the single source of truth: `darkpool_analysis/data/darkpool.duckdb`.
- Raw tables: `finra_otc_volume_raw` (includes `source` column for tier: T1/T2/OTC), `equity_trades_raw`.
- Derived tables: `equity_lit_directional_flow`, `darkpool_estimated_flow`, `darkpool_daily_summary`.
- Store float64 values; round only for display/export.

## External Integrations
- FINRA OTC Transparency (ATS + non-ATS) equity volume; authoritative weekly dataset.
- Polygon equity trade prints for lit-market proxy (NBBO if available; otherwise TICK rule).
- Options flow is out of scope; no inferred trade direction from options or short-sale files.

## Data Sources Explained

### FINRA OTC Data
FINRA publishes **volume only** — no buy/sell direction:
- `off_exchange_volume`: Total shares traded off-exchange
- `trade_count`: Number of trades
- `tierIdentifier`: Market tier (T1, T2, OTC)
- `week_start_date`: Weekly aggregation period

### Polygon Equity Trades
Polygon provides lit-market trade data used to infer direction:
- `symbol`: Ticker symbol
- `timestamp`: Trade execution time (nanoseconds)
- `price`: Execution price
- `size`: Shares traded
- `bid`: Best bid at trade time (NBBO)
- `ask`: Best ask at trade time (NBBO)

## Inference Methodology (Option B)

We use **Option B: Lit Market Proxy** — applying lit-market buy/sell ratios to dark pool volume.

### Why Option B?
| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Option A: Options Flow** | Infer from options activity | Captures institutional intent | Complex; options ≠ equity |
| **Option B: Lit Market Proxy** | Apply lit buy/sell ratio | Simple; same-asset inference | Assumes dark ≈ lit behavior |
| **Option C: Short Sale Data** | Use short volume as sell proxy | Direct short data | Shorts ≠ all sells |

### Classification Methods
1. **NBBO** (preferred): Compare trade price to bid/ask quotes
   - `price >= ask` → BUY (buyer lifting the offer)
   - `price <= bid` → SELL (seller hitting the bid)
   - else → NEUTRAL (excluded)

2. **TICK** (fallback): Compare to previous trade price
   - `price > prev_price` → BUY
   - `price < prev_price` → SELL
   - else → NEUTRAL (excluded)

NBBO is used when all trades have valid bid/ask; TICK is used if any are missing.

## Polygon API Subscription Tiers

| Tier | Trades API | Aggregates API | Classification |
|------|------------|----------------|----------------|
| **Free** | 403 Forbidden | Available | TICK (minute bars) |
| **Paid** | Available | Available | NBBO (tick-level) |

When trades API returns 403, code auto-falls back to minute aggregates:
- Uses OHLC bars: `close > open` → BUY, `close < open` → SELL
- Less accurate since each bar contains multiple trades in both directions

## Trading Hours Consideration

**Current setting**: RTH 09:30-16:15 ET (15-min buffer for closing auction)

### The Mismatch
- FINRA reports **total daily volume** (all hours, including pre/post market)
- We classify lit trades during **RTH only** (~90% of volume)
- Assumption: Extended hours direction ≈ RTH direction

### Extended Hours Trade-off
| Approach | Pro | Con |
|----------|-----|-----|
| RTH only (current) | Better NBBO quality, tight spreads | Misses ~10% of volume |
| Include extended hours | Captures all volume | Wider spreads, noisier classification |

Extended hours typically have wider spreads and lower liquidity, making NBBO classification less reliable.

## Deployment and Operations
- Local setup: `pip install -r darkpool_analysis/requirements.txt`, configure `.env` with API keys, then run from root.
- Run from root: `python main.py`
- Run from subdirectory: `cd darkpool_analysis && python orchestrator.py`
- CLI tools: `python plotter.py --symbols TQQQ,SPY` or `python table_renderer.py --dates 2025-12-20`
- Daily run after market close; weekly run Sunday after FINRA settlement.
- Set `POLYGON_MAX_WORKERS` env var to control parallel API threads (default: 4).

## Configuration Structure
- **`.env` (project root)**: Only secrets (POLYGON_API_KEY, FINRA_API_KEY, FINRA_API_SECRET).
- **`config.py`**: All defaults including:
  - `DEFAULT_TICKERS = ["TQQQ"]`
  - `DEFAULT_TARGET_DATE = "2025-12-22"` (update as needed)
  - `DEFAULT_FETCH_MODE = "daily"`
  - `DEFAULT_BACKFILL_COUNT = 10`
  - `DEFAULT_RTH_START = "09:30"`, `DEFAULT_RTH_END = "16:15"`
  - `DEFAULT_MIN_LIT_VOLUME = 10000`
  - `DEFAULT_EXPORT_CSV = False`
- **Fetch modes** (`FETCH_MODE` env var or DEFAULT_FETCH_MODE):
  - `"single"`: Process only target_date, snapshot keyed by target_date.
  - `"daily"`: Process last N trading days (BACKFILL_COUNT), snapshot keyed by target_date.
  - `"weekly"`: Process last N Fridays (BACKFILL_COUNT), snapshot keyed by FINRA week.
- Polygon trades API requires paid tier; code auto-falls back to aggregates (minute bars) on 403.
- DEFAULT_TARGET_DATE should be set to a trading day (not weekend/holiday).

## Known Gaps and Watchouts
- SPXW is not an equity ticker and must be excluded from FINRA volume tables (via `EXCLUDED_FINRA_TICKERS`).
- If lit volume is below MIN_LIT_VOLUME, set buy ratio to NULL and persist coverage metadata.
- Daily outputs must reference the most recent settled FINRA week.
- PNG table rendering requires one of: Playwright (recommended), imgkit, or Selenium.

## Tips for Future Contributors
- Keep the mandatory disclaimer in README and code comments: FINRA does not publish trade direction; buy/sell values are inferred estimates.
- Avoid hidden assumptions; document inference logic and coverage limits explicitly.
- Update `AGENT_CHANGELOG.md` after each session with the required checkbox format.
- The plotter and table_renderer modules have CLI entry points for standalone use.
