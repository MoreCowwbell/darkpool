# Agent Instructions for This Repository

## Mission and Starting Point
- Always read AGENT_CONTEXT.md before coding.
- Implement the FINRA + Polygon institutional accumulation/distribution pipeline with explicit data provenance.

## Core Requirements (non-negotiable)
- Dark pool anchor = FINRA OTC weekly summary volume only.
- FINRA does not publish direction; all buy/sell values are inferred estimates.
- FINRA daily short sale data is a separate pressure metric and must never replace OTC weekly volume.
- Price context (close, return, z-score) must come from Polygon aggregates, not short sale files.
- Use Option B inference only; options flow and short-sale direction files are out of scope.
- Exclude SPXW from FINRA volume tables (EXCLUDED_FINRA_TICKERS in config.py).
- DuckDB is the only persistence layer; store float64 and round only for display.
- All paths must be local and relative.

## Phase Sequencing
- Do not implement Phase C visualization until database_check.ipynb and daily table outputs are correct.

## Option B Inference Rules
- Regular Trading Hours only (09:30-16:15 ET); exclude extended hours.
- Classification (NBBO) used when all trades have valid bid/ask:
  - price >= ask => BUY
  - price <= bid => SELL
  - else => NEUTRAL
- Fallback (TICK) used if any bid/ask is missing:
  - price > prev_price => BUY
  - price < prev_price => SELL
  - else => NEUTRAL
- Exclude NEUTRAL trades from volume calculations.
- If lit_buy_volume + lit_sell_volume < MIN_LIT_VOLUME, set lit_buy_ratio to NULL and persist coverage metadata.

## Ratio Formulas
- lit_buy_ratio = buy_volume / (buy_volume + sell_volume)
- log_buy_sell = ln(buy_volume / sell_volume) when both > 0, else NULL
- short_ratio = (short_volume + short_exempt_volume) / denominator_value when denominator_value > 0
- short_ratio_denominator_type = FINRA_TOTAL or POLYGON_TOTAL
- short_ratio_denominator_value = numeric denominator used

## Signal Thresholds (Phase C)
- BOT signal: log_buy_sell > ln(1.25) (~ +0.223)
- SELL signal: log_buy_sell < ln(0.80) (~ -0.223)

## Required Tables (must materialize in db.py)
- finra_otc_weekly_raw: symbol, week_start_date, off_exchange_volume, trade_count, tier/participant metadata
- finra_short_daily_raw: symbol, trade_date, short_volume, short_exempt_volume, total_volume, market, source_file
- polygon_equity_trades_raw: symbol, timestamp, price, size, bid, ask
- polygon_daily_agg_raw: symbol, trade_date, open, high, low, close, vwap, volume
- lit_direction_daily: symbol, date, lit_buy_volume, lit_sell_volume, lit_buy_ratio, log_buy_sell, classification_method, coverage stats, inference_version
- daily_metrics: symbol, date, log_buy_sell, short_ratio, short_ratio_denominator_type/value, price context, pressure_context_label, data_quality, provenance flags, inference_version
- index_constituent_short_agg_daily: index_symbol, trade_date, agg_short_ratio, coverage stats, index_return, interpretation_label, inference_version, data_quality
- composite_signal (Phase C only)

## Outputs
- Save tables to darkpool_analysis/output/tables/ (CSV if EXPORT_CSV=True, HTML/PNG always).
- Save plots to darkpool_analysis/output/plots/ (Phase C only).
- Tables and plots read from DuckDB only.

## Data Quality and Provenance
- data_quality = OTC_ANCHORED when the OTC week covers the target date; else PRE_OTC.
- Persist has_otc, has_short, has_lit flags for each daily_metrics row.
- Persist short_ratio_source to avoid double counting when consolidated files are present.
- If short_ratio uses Polygon denominator, label short_ratio_denominator_type = POLYGON_TOTAL.

## Error Handling
- FINRA OTC weekly failure: abort run with RuntimeError.
- FINRA short sale failure: log warning and continue (short_ratio becomes NULL).
- Polygon trades 403: auto-fallback to aggregates (minute bars) with log message.
- Polygon complete failure: proceed with warning; persist empty lit flow.
- Polygon daily aggregates failure: log warning and continue (price context becomes NULL).
- Always persist coverage metadata when inference is limited.

## Configuration Structure
- .env (project root): only API secrets (POLYGON_API_KEY, FINRA_API_KEY, FINRA_API_SECRET).
- config.py: defaults for tickers, dates, URLs, and pipeline behavior.
- Fetch modes (FETCH_MODE): "single", "daily", or "weekly".
- FINRA_SHORT_SALE_* env vars control short sale ingestion (file or API).
- FINRA_OTC_* env vars control weekly OTC ingestion.
- POLYGON_DAILY_AGG_* env vars control daily aggregates ingestion.

## Documentation and Runbooks
- Keep README.md synchronized with module layout and deployment steps.
- CLI usage:
  - python main.py (from project root)
  - python orchestrator.py (from darkpool_analysis/)
  - python table_renderer.py --dates 2025-12-20 (standalone tables)
- database_check.ipynb is required for validation before plotting.
- Mandatory disclaimer must appear in README and code comments:
  "FINRA does not publish trade direction for off-exchange volume. Buy/Sell values are inferred estimates derived from lit-market equity trades and applied proportionally to FINRA OTC volume."

## Code Style, Layout, and Dependencies
- Target Python 3.11+ with explicit imports.
- Use relative imports within the package (with try/except ImportError fallback for direct execution).
- Keep orchestrator.py thin; delegate API, persistence, analytics, and rendering to dedicated modules.
- Keep requirements.txt aligned with actual imports: duckdb, matplotlib, pandas, python-dotenv, pytz, requests.
- Optional PNG rendering: playwright (recommended), imgkit, selenium.
- Use ThreadPoolExecutor for parallel API fetching; control via POLYGON_MAX_WORKERS env var (default: 4).
- Always wrap DuckDB connections in try/finally to ensure conn.close() on exit.

## Testing, Validation, and Logging
- Keep console logging informative (logging module) so scheduler logs are useful during failures.
- Use logging.info() for normal operations, logging.warning() for partial failures, logging.error() for fatal issues.

## AGENT_CHANGELOG Expectations
- After each work session, append a dated section to AGENT_CHANGELOG.md.
- Use checkboxes [x] for completed tasks and a "ƒ-" prefix for in-progress items.
- Keep entries concise and archive/summarize older blocks when needed.

## Output Rules: Snippets vs. Full Files
- Default to returning only the modified snippet when the change is localized to a single block or function.
- Return the entire file only when edits touch multiple sections, adjust imports/class definitions, or when global context is critical.
- Keep full-file outputs to a few hundred lines.
- When uncertain, provide the snippet and ask if the user needs the full file.

## Environment and Safety
- Assume python -m pip install -r requirements.txt prepares the environment.
- Mention hidden assumptions or scheduling edge cases in responses so the next agent can continue seamlessly.

## PATH and Directories
- Always use relative paths whenever possible.
- Use Path(__file__).resolve().parent for package-relative paths.
- Project root .env is loaded via load_dotenv(project_root / ".env") in config.py.
