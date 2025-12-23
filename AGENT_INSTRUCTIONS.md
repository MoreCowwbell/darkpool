# Agent Instructions for This Repository

## Mission and Starting Point
- Always read `AGENT_CONTEXT.md` before coding.
- Implement the Dark Pool ETF Analysis (Option B) plan in `darkpool_analysis/` without inventing data sources.

## Core Requirements (non-negotiable)
- Dark pool = FINRA OTC (ATS + non-ATS) equity volume only.
- FINRA does not publish direction; all buy/sell values are inferred estimates.
- Use Option B inference only; options flow is out of scope; no short-sale files.
- Exclude `SPXW` from FINRA volume tables (via `EXCLUDED_FINRA_TICKERS` in config.py).
- DuckDB is the only persistence layer; store float64 and round only for display.
- All paths must be local and relative; no notebooks; no global state.

## Option B Inference Rules
- Regular Trading Hours only (09:30-16:15 ET); exclude extended hours.
- Classification (NBBO) — used when all trades have valid bid/ask:
  - `price >= ask` => BUY
  - `price <= bid` => SELL
  - else => NEUTRAL
- Fallback (TICK) — used if any bid/ask is missing:
  - `price > prev_price` => BUY
  - `price < prev_price` => SELL
  - else => NEUTRAL
- Exclude NEUTRAL trades from volume calculations.
- If `lit_buy_volume + lit_sell_volume < MIN_LIT_VOLUME`, set `lit_buy_ratio` to NULL and persist coverage metadata.

## Ratio Formulas
- `lit_buy_ratio` = buy_volume / (buy_volume + sell_volume) — percentage (0-1 scale), used internally for volume estimation
- `buy_ratio` (in summary) = estimated_bought / estimated_sold — ratio scale (1.0 = neutral, >1 = net buying, <1 = net selling)

## Signal Thresholds
- `BOT` signal: buy_ratio > 1.25 (or log_ratio > ln(1.25) ≈ +0.223)
- `SELL` signal: buy_ratio < 0.80 (or log_ratio < ln(0.80) ≈ -0.223)

## Required Tables (must materialize in db.py)
- `finra_otc_volume_raw`: symbol, week_start_date, off_exchange_volume, trade_count, source (tier: T1/T2/OTC)
- `equity_trades_raw`: symbol, timestamp, price, size, bid, ask
- `equity_lit_directional_flow`: symbol, date, lit_buy_volume, lit_sell_volume, lit_buy_ratio, classification_method, lit_coverage_pct
- `darkpool_estimated_flow`: symbol, date, finra_off_exchange_volume, estimated_dark_buy_volume, estimated_dark_sell_volume, applied_lit_buy_ratio, inference_version
- `darkpool_daily_summary`: date, symbol, estimated_bought, estimated_sold, buy_ratio, total_off_exchange_volume, finra_period_type

## Outputs
- Save tables to `darkpool_analysis/output/tables/` (CSV if EXPORT_CSV=True, HTML/PNG always).
- Save plots to `darkpool_analysis/output/plots/`.
- Plot log ratio per ticker with thresholds (BOT > +0.223, SELL < -0.223); annotate crossings.
- Plots and tables read from DuckDB only.

## Error Handling
- FINRA failure: abort run with RuntimeError.
- Polygon trades 403: auto-fallback to aggregates (minute bars) with log message.
- Polygon complete failure: proceed with warning; persist empty lit flow.
- Always persist coverage metadata when inference is limited.

## Configuration Structure
- **`.env` (project root)**: Only API secrets (POLYGON_API_KEY, FINRA_API_KEY, FINRA_API_SECRET).
- **`config.py`**: All defaults including DEFAULT_TICKERS, DEFAULT_TARGET_DATE, DEFAULT_FETCH_MODE, API URLs.
- **Fetch modes** (`FETCH_MODE` env var or DEFAULT_FETCH_MODE):
  - `"single"`: Process only target_date.
  - `"daily"`: Process last BACKFILL_COUNT trading days.
  - `"weekly"`: Process last BACKFILL_COUNT Fridays, snapshot keyed by FINRA week.
- FINRA uses direct X-API-KEY/X-API-SECRET headers (no OAuth token flow by default).
- FINRA data includes `tierIdentifier` field (T1=NMS Tier 1, T2=NMS Tier 2, OTC) stored as `source` column.
- Set DEFAULT_TARGET_DATE to a trading day; weekends/holidays will have no Polygon data.

## Code Style, Layout, and Dependencies
- Target Python 3.11+ with explicit imports.
- Use relative imports within the package (with `try/except ImportError` fallback for direct execution).
- Keep `orchestrator.py` thin; delegate API, persistence, analytics, and plotting to dedicated modules.
- Keep `requirements.txt` aligned with actual imports: duckdb, matplotlib, pandas, python-dotenv, pytz, requests.
- Optional PNG rendering: playwright (recommended), imgkit, selenium.
- Use `ThreadPoolExecutor` for parallel API fetching; control via `POLYGON_MAX_WORKERS` env var (default: 4).
- Always wrap DuckDB connections in `try/finally` to ensure `conn.close()` on exit.

## Testing, Validation, and Logging
- Keep console logging informative (`logging` module) so Windows Task Scheduler logs are useful during failures.
- Use `logging.info()` for normal operations, `logging.warning()` for partial failures, `logging.error()` for fatal issues.

## Documentation and Runbooks
- Keep `README.md` synchronized with the module layout and deployment steps.
- CLI usage:
  - `python main.py` (from root)
  - `python orchestrator.py` (from darkpool_analysis/)
  - `python plotter.py --symbols TQQQ,SPY` (standalone plots)
  - `python table_renderer.py --dates 2025-12-20` (standalone tables)
- Document new env vars or config toggles in both `README.md` and `AGENT_CONTEXT.md`.
- Mandatory disclaimer must appear in README and code:
  "FINRA does not publish trade direction for off-exchange volume. Buy/Sell values are inferred estimates derived from lit-market equity trades and applied proportionally to FINRA OTC volume."

## AGENT_CHANGELOG Expectations
- After each work session, append a dated section to `AGENT_CHANGELOG.md`.
- Use checkboxes `[x]` for completed tasks and `⧗` prefix for in-progress items.
- Keep entries concise and archive/summarize older blocks instead of letting the file grow forever.

## Output Rules: Snippets vs. Full Files
- Default to returning only the modified snippet when the change is localized to a single block or function.
- Return the entire file only when edits touch multiple sections, adjust imports/class definitions, or when global context is critical for correctness.
- Keep full-file outputs to a few hundred lines.
- When uncertain, provide the snippet and ask if the user needs the full file.

## Environment and Safety
- Assume `python -m pip install -r requirements.txt` prepares the environment.
- Mention hidden assumptions or scheduling edge cases in responses so the next agent can continue seamlessly.

## PATH and Directories
- Always use relative paths whenever possible.
- Use `Path(__file__).resolve().parent` for package-relative paths.
- Project root `.env` is loaded via `load_dotenv(project_root / ".env")` in config.py.
