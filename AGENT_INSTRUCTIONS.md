# Agent Instructions for This Repository

## Mission and Starting Point
- Always read `AGENT_CONTEXT.md` before coding.
- Implement the Dark Pool ETF Analysis (Option B) plan in `darkpool_analysis/` without inventing data sources.

## Core Requirements (non-negotiable)
- Dark pool = FINRA OTC (ATS + non-ATS) equity volume only.
- FINRA does not publish direction; all buy/sell values are inferred estimates.
- Use Option B inference only; options flow is out of scope; no short-sale files.
- Exclude `SPXW` from FINRA volume tables; skip any ticker without FINRA equity data gracefully.
- DuckDB is the only persistence layer; store float64 and round only for display.
- All paths must be local and relative; no notebooks; no global state.

## Option B Inference Rules
- Regular Trading Hours only (09:30-16:00 ET); exclude extended hours.
- Classification (NBBO):
  - `price >= ask` => BUY
  - `price <= bid` => SELL
  - else => NEUTRAL
- Fallback (TICK) if NBBO unavailable:
  - `price > prev_price` => BUY
  - `price < prev_price` => SELL
  - else => NEUTRAL
- Exclude NEUTRAL trades.
- If `lit_buy_volume + lit_sell_volume < MIN_LIT_VOLUME`, set `lit_buy_ratio` to NULL and persist coverage metadata.

## Required Tables (must materialize)
- `finra_otc_volume_raw`: symbol, trade_date or week_start_date, off_exchange_volume, trade_count (if available)
- `equity_trades_raw`: symbol, timestamp, price, size, bid (if available), ask (if available)
- `equity_lit_directional_flow`: symbol, date, lit_buy_volume, lit_sell_volume, lit_buy_ratio, classification_method, lit_coverage_pct
- `darkpool_estimated_flow`: symbol, date, finra_off_exchange_volume, estimated_dark_buy_volume, estimated_dark_sell_volume, applied_lit_buy_ratio, inference_version
- `darkpool_daily_summary`: date, symbol, estimated_bought, estimated_sold, buy_ratio, total_off_exchange_volume, finra_period_type

## Outputs
- Save tables to `darkpool_analysis/output/tables/` and plots to `darkpool_analysis/output/plots/`.
- Plot buy ratio per ticker with thresholds (BOT > 1.25, SELL < 0.80); annotate only.
- Plots and tables read from DuckDB only.

## Error Handling
- FINRA failure: abort run.
- Polygon trades 403: auto-fallback to aggregates (minute bars) with log message.
- Polygon complete failure: proceed with warning; persist empty lit flow.
- Always persist coverage metadata when inference is limited.

## Configuration Structure
- **`.env` (project root)**: Only API secrets (POLYGON_API_KEY, FINRA_API_KEY, FINRA_API_SECRET).
- **`config.py`**: All defaults including DEFAULT_TICKERS, DEFAULT_TARGET_DATE, API URLs.
- FINRA uses direct X-API-KEY/X-API-SECRET headers (no OAuth token flow).
- Set DEFAULT_TARGET_DATE to a trading day; weekends/holidays will have no Polygon data.

## Code Style, Layout, and Dependencies
- Target Python 3.11+ with explicit imports.
- Use relative imports within the package (with `try/except ImportError` fallback for direct execution).
- Keep `orchestrator.py` thin; delegate API, persistence, analytics, and plotting to dedicated modules.
- Keep `requirements.txt` aligned with actual imports (requests, duckdb, matplotlib, pytz, python-dotenv, pandas, etc.).
- Use `ThreadPoolExecutor` for parallel API fetching; control via `POLYGON_MAX_WORKERS` env var.
- Always wrap DuckDB connections in `try/finally` to ensure `conn.close()` on exit.

## Testing, Validation, and Logging
- Keep console logging informative (`print` or `logging`) so Windows Task Scheduler logs are useful during failures.

## Documentation and Runbooks
- Keep `README.md` synchronized with the module layout and deployment steps.
- CLI usage: `python main.py` (from root) or `python orchestrator.py` (from darkpool_analysis/).
- Document new env vars or config toggles in both `README.md` and `AGENT_CONTEXT.md`.
- Mandatory disclaimer must appear in README and code:
  "FINRA does not publish trade direction for off-exchange volume. Buy/Sell values are inferred estimates derived from lit-market equity trades and applied proportionally to FINRA OTC volume."

## AGENT_CHANGELOG Expectations
- After each work session, append a dated section to `AGENT_CHANGELOG.md`. Use checkboxes for completed tasks and prefix in-progress bullets with the required marker. Keep entries concise and archive/summarize older blocks instead of letting the file grow forever.

## Output Rules: Snippets vs. Full Files
- Default to returning only the modified snippet when the change is localized to a single block or function.
- Return the entire file only when edits touch multiple sections, adjust imports/class definitions, or when global context is critical for correctness. Keep full-file outputs to a few hundred lines.
- When uncertain, provide the snippet and ask if the user needs the full file.

## Environment and Safety
- Assume `python -m pip install -r requirements.txt` prepares the environment.
- Mention hidden assumptions or scheduling edge cases in responses so the next agent can continue seamlessly.

## PATH and directories
- Always use relative paths whenever possible.
