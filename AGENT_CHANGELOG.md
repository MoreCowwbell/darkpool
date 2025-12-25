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
