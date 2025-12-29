# Agent Changelog

This file tracks completed tasks and in-progress work across sessions.  
Each entry should be short, bullet-based, and written immediately after a task is completed.

Format:
- Use checkboxes for completed tasks.
- Use a "⧗" prefix for tasks that are in progress (not yet finished).
- Keep each bullet to one line.
- Keep total log size manageable (older items can be summarized or archived periodically).

---

## Example of logs:
## 2025-01-XX — Session Summary (Agent Name)
- [x] Short description of completed task.
- [x] Another completed task.
- ⧗ In-progress task that should be resumed next session.

## 2025-01-XX — Session Summary (Agent Name)
- [x] Completed adjustment to function X.
- ⧗ Started refactor of Y but not finished.

## 2025-01-14 — Session Summary (Claude Code)
- [x] Added timeout support to Polygon request helper.
- [x] Fixed missing import in historical.py.
- ⧗ Started rewriting intraday_stream.py state machine (continue next session).

---
## Start of Changelog below:
## 2025-12-27 Session Summary (Codex)
- [x] Added FINRA CDN full-list short sale scanner with new raw/metrics tables and scoring.
- [x] Added scanner orchestration, renderer outputs to output/scanner/YYYY-MM-DD, and -scanner CLI flag.
- [x] Updated config/env and README/AGENT_CONTEXT docs for scanner settings and outputs.
## 2025-12-21 Session Summary (Codex)
- [x] Updated `darkpool/AGENT_CONTEXT.md` to reflect the Option B dark pool ETF analysis plan and execution flow.
- [x] Updated `darkpool/AGENT_INSTRUCTIONS.md` with inference rules, table requirements, outputs, and error handling.
- [x] Logged this session entry in `darkpool/AGENT_CHANGELOG.md`.
## 2025-12-21 Session Summary (Codex)
- [x] Created `darkpool_analysis/` project structure with config and DuckDB scaffolding.
- [x] Implemented FINRA/Polygon ingestion, lit inference, analytics, and plotting modules.
- [x] Added `darkpool_analysis/README.md`, `darkpool_analysis/requirements.txt`, and `darkpool_analysis/.env` template.
- [x] Wired `darkpool_analysis/orchestrator.py` end-to-end and exported tables/plots.

## 2025-12-21 Session Summary (Claude Code)
- [x] Added root `main.py` entry point for running from project root.
- [x] Created `darkpool_analysis/__init__.py` for proper package structure.
- [x] Fixed imports to use relative imports with `try/except ImportError` fallback.
- [x] Fixed DuckDB connection leak by wrapping in `try/finally` with `conn.close()`.
- [x] Added parallel Polygon API fetching using `ThreadPoolExecutor` (configurable via `POLYGON_MAX_WORKERS`).
- [x] Fixed `iterrows()` performance in plotter using vectorized mask operations.
- [x] Fixed `buy_ratio` formula inconsistency: now `bought/(bought+sold)` matching `lit_buy_ratio`.
- [x] Updated `AGENT_CONTEXT.md`, `AGENT_INSTRUCTIONS.md`, and `AGENT_CHANGELOG.md`.

## 2025-12-21 Session Summary (Claude Code) - Debugging & Config Cleanup
- [x] Consolidated `.env` files: moved secrets to project root, deleted `darkpool_analysis/.env`.
- [x] Moved non-secret config (URLs, methods, tickers) to `config.py` constants.
- [x] Fixed FINRA auth: set `DEFAULT_FINRA_TOKEN_URL = ""` to use direct API key auth (no OAuth).
- [x] Added `issueSymbolIdentifier` to FINRA column resolution candidates.
- [x] Fixed DuckDB connection leak in orchestrator: moved `plot_buy_ratio_series` after `conn.close()`.
- [x] Added Polygon aggregates fallback when trades API returns 403.
- [x] Simplified DEFAULT_TICKERS to just NVDA for initial testing.
- [x] Added `DEFAULT_TARGET_DATE = "2025-12-19"` to handle weekend runs.
- [x] Successfully ran end-to-end analysis with plot output.

## 2025-12-21 Session Summary (Claude Code) - Fetch Mode & FINRA Tier Tracking
- [x] Consolidated `RUN_MODE` and `FETCH_MODE` into single `FETCH_MODE` setting.
- [x] Added multi-date fetch modes: "single", "daily" (last N trading days), "weekly" (last N Fridays).
- [x] Renamed `DEFAULT_BACKFILL_FRIDAYS` to `DEFAULT_BACKFILL_COUNT`.
- [x] Added `_get_past_trading_days()` helper for daily mode.
- [x] Attempted multi-endpoint FINRA approach (ats, nms_tier1, nms_tier2, otc) - discovered 401 errors.
- [x] Researched FINRA API: single endpoint returns all tiers via `tierIdentifier` field (T1, T2, OTC).
- [x] Reverted to single FINRA endpoint, now extracting tier from `tierIdentifier` as `source` column.
- [x] Added `source` column to `finra_otc_volume_raw` table for tier tracking.
- [x] Updated upsert key to `["symbol", "week_start_date", "source"]` for proper deduplication.
- [x] Added symbol filtering to FINRA API request (`orFilters`) to fetch only configured tickers.
- [x] Changed `buy_ratio` formula from percentage (bought/(bought+sold)) to ratio (bought/sold).
- [x] Added `EXPORT_CSV` config option (default: False) to control CSV table exports.
- [x] Created dark-theme table renderer (`table_renderer.py`) with HTML/PNG output.
- [x] Improved `plotter.py` with dark mode styling, 300 DPI, colored thresholds, and better annotations.

## 2025-12-21 Session Summary (Claude Code) - Log Ratio Visualization
- [x] Updated `plotter.py` to use symmetric log-ratio: y = ln(Buy/Sell) instead of raw buy_ratio.
- [x] Added log-space thresholds: LOG_BOT_THRESHOLD = ln(1.25) ≈ +0.223, LOG_SELL_THRESHOLD = ln(0.80) ≈ -0.223.
- [x] Implemented volume-scaled marker sizes using scatter plot (MARKER_SIZE_MIN=30, MARKER_SIZE_MAX=300).
- [x] Added helper functions: `_compute_log_ratio()` for safe log calculation, `_scale_marker_sizes()` for normalization.
- [x] Uses `darkpool_daily_summary` table for `estimated_bought`, `estimated_sold`, `total_off_exchange_volume`.
- [x] Updated y-axis label to "log(Buy / Sell)" and title to include "Log".
- [x] Output filename changed from `*_buy_ratio.png` to `*_log_ratio.png`.

## 2025-12-23 Session Summary (Claude Code) - Documentation Audit & Update
- [x] Analyzed all 11 Python source files in darkpool_analysis package.
- [x] Updated `AGENT_CONTEXT.md` with accurate current state:
  - Corrected DEFAULT_RTH_END from "16:00" to "16:15" (matches config.py).
  - Added CLI entry points for plotter.py and table_renderer.py.
  - Added PNG rendering dependency note (playwright/imgkit/selenium).
  - Updated execution flow to show multi-date processing loop.
- [x] Updated `AGENT_INSTRUCTIONS.md` with accurate rules:
  - Added Signal Thresholds section (BOT > 1.25, SELL < 0.80).
  - Fixed RTH end time to 16:15.
  - Added CLI usage examples for all entry points.
  - Clarified NBBO vs TICK classification conditions.
  - Added logging level guidance.
- [x] Logged this session entry in `AGENT_CHANGELOG.md`.

## 2025-12-23 Session Summary (Claude Code) - Context Enrichment
- [x] Added "Data Sources Explained" section to `AGENT_CONTEXT.md`:
  - FINRA OTC data fields (volume only, no direction).
  - Polygon equity trades fields (tick-level with NBBO).
- [x] Added "Inference Methodology (Option B)" section:
  - Comparison of Option A (options flow), Option B (lit proxy), Option C (short sale).
  - Detailed NBBO vs TICK classification logic.
- [x] Added "Polygon API Subscription Tiers" section:
  - Free tier (403 on trades, falls back to aggregates).
  - Paid tier (tick-level trades with NBBO).
- [x] Added "Trading Hours Consideration" section:
  - RTH vs extended hours trade-offs.
  - FINRA all-hours vs Polygon RTH mismatch explanation.

## 2025-12-23 Session Summary (Claude Code) - Plot Mode & Sell Ratio
- [x] Added `PLOT_MODE` config setting ("log" or "absolute") to `config.py`.
- [x] Added `sell_ratio` column to `darkpool_daily_summary` table schema in `db.py`.
- [x] Added `sell_ratio = sold / bought` computation in `analytics.py`.
- [x] Updated `plotter.py` to support both Log and Absolute plot modes:
  - Absolute mode: Y-axis 0.0 to 2.0+ (dynamic expansion), neutral line at 1.0.
  - Log mode: unchanged symmetric Y-axis around 0.
  - Mode-dependent thresholds, labels, and output filenames (`*_abs_ratio.png` vs `*_log_ratio.png`).
  - Added `--mode` CLI argument.
- [x] Updated `table_renderer.py`:
  - Added Sell Ratio column next to Buy Ratio.
  - Changed signal labels from "BOT"/"SELL" to "Accumulation"/"Distribution".
  - Added `_get_sell_ratio_color()` helper with inverse color logic.
- [x] Updated `orchestrator.py` to pass `config.plot_mode` to plotter.
- [x] Added smooth B-spline interpolation to plot line (replaces straight line connectors).
  - Uses `scipy.interpolate.make_interp_spline` with k=3 (cubic spline).
  - Falls back to straight lines if fewer than 4 data points.
  - Added `scipy` to `requirements.txt`.

## 2025-12-23 Session Summary (Codex)
- [x] Rewrote `finra_ticker_check.ipynb` with a clean ticker selector plus FINRA presence and Polygon snapshot checks.

## 2025-12-23 Session Summary (Codex)
- [x] Updated FINRA debug cell to select the latest week on or before `DATE` and print coverage range.

## 2025-12-24 Session Summary (Codex)
- [x] Updated AGENT docs and project README for the short-sale + price-context pipeline and new table names.
- [x] Refactored schema to separate raw tables and new derived tables (daily_metrics, index_constituent_short_agg_daily).
- [x] Implemented FINRA OTC weekly and daily short-sale fetchers plus Polygon daily aggregates ingestion.
- [x] Reworked analytics to compute log(B/S), short pressure metrics, price context, and pressure labels.
- [x] Rebuilt orchestrator to run Phase A/B pipeline and render daily metrics tables.
- [x] Added database_check.ipynb validation notebook and sample SPX constituent file.
- [x] Removed deprecated plotter files and trimmed requirements to active dependencies.

## 2025-12-24 Session Summary (Claude Code) - FINRA API Fixes
- [x] Fixed FINRA OAuth 2.0 authentication:
  - Set `DEFAULT_FINRA_TOKEN_URL` to FINRA Identity Platform FIP endpoint.
  - Updated `_build_finra_headers()` to use HTTP Basic Auth for token request.
  - Removed duplicate `data=` param from token request (grant_type already in URL).
- [x] Fixed FINRA short sale API field name mapping:
  - Updated date filter to use `tradeReportDate` instead of `tradeDate`.
  - Updated normalizer to handle API column names (securitiesInformationProcessorSymbolIdentifier, shortParQuantity, etc.).
- [x] Fixed pandas merge_asof dtype error:
  - Converted `date` and `week_start_date` columns to datetime64 before merge.
- [x] Fixed KeyError 'symbol' after merge_asof:
  - Dropped `symbol` column from `weekly` dataframe before merge to prevent symbol_x/symbol_y duplication.
- [x] Fixed FutureWarning for isin with datetime64:
  - Converted to `.dt.date` before comparison in analytics.py.
- [x] Updated finra_ticker_check.ipynb for refactored module structure.
- [x] Updated all project documentation (README, TODOS, AGENT_* files).

## 2025-12-24 Session Summary (Claude Code) - Phase C Plotter
- [x] Created `plotter.py` module with 3-panel dark-theme visualization:
  - Panel 1: log(Buy/Sell) with accumulation/distribution threshold lines.
  - Panel 2: Short ratio z-score with high/low pressure thresholds.
  - Panel 3: OTC off-exchange volume bars with data quality coloring.
- [x] Added matplotlib to requirements.txt.
- [x] Wired `render_metrics_plots()` into orchestrator after table rendering.
- [x] Updated TODOS.md: marked C2 as complete.
- [x] Updated README.md and AGENT_CONTEXT.md with plotter documentation.

## 2025-12-24 Session Summary (Claude Code) - FINRA domainFilters Fix
- [x] Fixed FINRA Short Sale API symbol filtering:
  - Root cause: `orFilters` is NOT a valid FINRA Query API filter type (API ignores it).
  - Fix: Changed to `domainFilters` with `fieldValues` array (correct syntax for multiple values).
  - API now returns only requested symbols instead of first 5000 rows alphabetically.
  - Reference: https://developer.finra.org/docs
- [x] Rolling z-score crash fixes already applied in previous session.

## 2025-12-24 Session Summary (Claude Code) - Smooth Line Interpolation
- [x] Added PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) smoothing to `plotter.py`:
  - Added `numpy` and `scipy.interpolate.PchipInterpolator` imports.
  - Created `_plot_smooth_line()` helper function for reusable smooth curve plotting.
  - Panel 1 (log_buy_sell): Now uses smooth PCHIP curve instead of linear segments.
  - Panel 2 (short_ratio_z): Now uses smooth PCHIP curve instead of linear segments.
  - Uses 150 interpolation points when 3+ data points available.
  - Falls back to linear plot for fewer than 3 data points.
- [x] Updated `requirements.txt`: Added `numpy` and `scipy` dependencies.

## 2025-12-24 Session Summary (Codex)
- [x] Fixed FINRA domainFilters payloads to use the required `values` list for OTC and short sale API calls.
- [x] Normalized daily metrics date typing to avoid mixed Timestamp/date sort failures.
- [x] Updated AGENT docs with the corrected domainFilters example.
## 2025-12-25 Session Summary (Codex)
- [x] Updated the daily metrics table to group by ticker with average rows, add inferred buy/sell and volume fields, and include summary/definition panels.
- [x] Added a short-only plot mode alongside layered plots and wired the orchestrator to generate both outputs.
- [x] Documented plot modes in `darkpool_analysis/README.md`.
## 2025-12-25 Session Summary (Codex)
- [x] Reworked daily metrics to compute short-sale buy ratios, lit ratios, and OTC weekly proxy ratios with consolidated short totals and OTC status.
- [x] Rebuilt table layout with short/lit/OTC sections, updated summaries/definitions, and added OTC week/status columns.
- [x] Updated plots to show short sale buy ratio, lit buy/log ratios, OTC buy/sell with a decision strip, and set orchestrator to layered mode.
## 2025-12-25 Session Summary (Codex)
- [x] Updated `darkpool/AGENT_CONTEXT.md` with consolidated short totals, new table layout, OTC status, and layered plots.
- [x] Updated `darkpool/AGENT_INSTRUCTIONS.md` with short ratio formulas and OTC status provenance.
- [x] Updated `darkpool/darkpool_analysis/TODOS.md` with revised plot description and table/metric revision items.
## 2025-12-25 Session Summary (Codex)
- [x] Added configurable table styling in `darkpool/darkpool_analysis/config.py` (DEFAULT_TABLE_STYLE).
- [x] Restyled `darkpool/darkpool_analysis/table_renderer.py` for grouped readability, muted signals, zoning, and status glyphs.
- [x] Documented table styling knobs in `darkpool/darkpool_analysis/README.md`.
## 2025-12-25 Session Summary (Codex)
- [x] Added plot styling helpers for axis bounds, centerlines, and legend styling in `darkpool/darkpool_analysis/plotter.py`.
- [x] Applied per-panel legends, subtle gridlines, and reduced marker/bar dominance for layered and short-only plots.
## 2025-12-25 Session Summary (Codex)
- [x] Added short buy/sell ratio and z-score to analytics and DuckDB schema.
- [x] Switched tables and plots to display Short Sale Buy/Sell Ratio with updated definitions.
## 2025-12-25 Session Summary (Codex)
- [x] Added BOT/SELL threshold lines and dashed neutral midline to the short-sale ratio panels.
## 2025-12-25 Session Summary (Codex)
- [x] Added column group headers for short/lit/OTC sections and emphasized AVG rows in the table.
- [x] Colored pressure text for Accumulating/Distribution in the table.
## 2025-12-25 Session Summary (Codex)
- [x] Reorganized table summary panel into aligned short/lit/OTC grids with larger summary fonts.

## 2025-12-26 Session Summary (Claude Code)
- [x] Added `POLYGON_TRADES_MODE` config option to control data granularity for lit inference.
- [x] Three modes supported: "tick" (accurate/slow), "minute" (faster/less accurate), "daily" (skip lit inference).
- [x] Updated `fetch_polygon_equity.py` to branch on mode.
- [x] Added detailed docstrings explaining `DEFAULT_POLYGON_TRADES_MODE` and `DEFAULT_INFERENCE_VERSION`.
- [x] Updated orchestrator to log the polygon trades mode at startup.
- [x] Updated AGENT_CONTEXT.md and AGENT_INSTRUCTIONS.md with new config option documentation.
- [x] Updated plotter.py: dates now shown on ALL panels (top, middle, bottom), every day labeled (no skipping).
- [x] Implemented Polygon data caching to avoid re-fetching:
  - Added `polygon_ingestion_state` table to track fetched (symbol, trade_date, data_source) combinations.
  - Added `data_source` column to `polygon_equity_trades_raw` to distinguish tick vs minute data.
  - Added `fetch_timestamp` column to `polygon_daily_agg_raw`.
  - Added `SKIP_CACHED` config option (default: True).
  - Updated `fetch_polygon_equity.py` and `fetch_polygon_agg.py` to check cache before fetching.
  - Updated orchestrator to pass conn to fetchers and log cache hit/miss stats.
- [x] Fixed KeyError 'trade_date' when cache hits all symbols:
  - `fetch_polygon_agg.py` was returning empty DataFrame without columns on cache hit.
  - Changed to return DataFrame with proper column structure (`empty_df` with all expected columns).
  - Fixed `_concat_frames` in orchestrator.py to preserve column structure when all frames are empty.
  - Uses `df.head(0)` to return empty DataFrame with columns instead of `pd.DataFrame()`.
  - Allows downstream analytics.py to safely access columns on empty DataFrame.
- [x] Fixed FutureWarning about DataFrame concatenation with empty/all-NA entries:
  - Added filter to exclude all-NA DataFrames before concat.
  - Added early return for single-frame case to avoid unnecessary concat.
- [x] Fixed lit direction data being overwritten with NA when trades cached:
  - Root cause: `compute_lit_directional_flow` returned placeholder rows with NA values when trades_df was empty.
  - These placeholder rows were upserting and overwriting real lit_direction_daily data.
  - Fix: Only compute and upsert lit direction when `trades_cache_stats["fetched"] > 0`.
  - Now read lit_direction_daily from database for all dates (includes cached + newly computed).


## 2025-12-28 Session Summary (Codex)
- [x] Updated plotter styling: larger agreement markers, thinner panel 1 line, added agreement markers/legend to panel 2, standardized Y-axis label styling, and clarified OTC panel title/legend.


## 2025-12-28 Session Summary (Codex)
- [x] Updated plot width scaling to use 12 inches per 50-day bucket for long date ranges.


## 2025-12-28 Session Summary (Codex)
- [x] Aligned panel 1/2 markers to valid data points and forced consistent x-limits/ticks across all panels (including OTC).

## 2025-12-28 Session Summary (Codex)
- [x] Extended FINRA OTC API date range to cover the full target-date span and wired the range into orchestrator calls.

## 2025-12-28 Session Summary (Codex)
- [x] Added standalone OHLCV price chart renderer with accumulation signal markers and dark-theme styling.
- [x] Wired price chart rendering into config/orchestrator and documented new outputs/options.
- [x] Added right-side padding on OHLCV charts to prevent last bar truncation.
- [x] Inverted price-chart signal placement (red above, green below) and increased marker spacing.
- [x] Added ticker group selection via TICKERS_TYPE and defined group lists in config.
