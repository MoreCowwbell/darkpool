# Project Context

## Project Goal Summary
- Build a production-quality Python analytics pipeline to infer institutional accumulation vs distribution for liquid ETFs.
- Keep FINRA OTC weekly volume as the authoritative off-exchange anchor.
- Add FINRA daily short sale volume as a separate, timely pressure signal (never a replacement for OTC weekly).
- Use Polygon lit-market trade inference (Option B) to compute a directional proxy (log(Buy/Sell)).
- Add price context (daily aggregates, returns, z-scores) alongside short-pressure metrics.
- Aggregate constituent-level short pressure to index-level totals with coverage stats.
- Persist raw and derived datasets in DuckDB and render daily tables for DEFAULT_TICKERS.

## Current Capabilities
- Phase A/B pipeline with separate ingestion tables and daily metrics with provenance flags.
- FINRA OTC weekly fetcher (API or file) with week selection.
- FINRA daily short sale ingestion (file or API) with OAuth 2.0 authentication and domainFilters symbol filtering.
- Polygon trades fetcher with aggregates fallback on 403.
- Polygon daily aggregates ingestion for price context.
- Lit inference using NBBO/TICK and log(Buy/Sell).
- Table renderer for daily outputs (HTML/PNG) with pressure context labels.
- Multi-panel plotter (Phase C): log_buy_sell, short_ratio_z, OTC volume per ticker with PCHIP smooth curves.
- Index-level constituent aggregation for short pressure.
- Pipeline validated and running end-to-end (2025-12-24).

## Where Things Live
- main.py - root entry point
- darkpool_analysis/orchestrator.py - pipeline driver
- darkpool_analysis/config.py - config/env loader
- darkpool_analysis/db.py - DuckDB schema and upsert
- darkpool_analysis/fetch_finra_otc.py - OTC weekly ingestion
- darkpool_analysis/fetch_finra_short.py - short sale daily ingestion
- darkpool_analysis/fetch_polygon_equity.py - Polygon trades fetcher
- darkpool_analysis/fetch_polygon_agg.py - Polygon daily aggregates (price/volume)
- darkpool_analysis/infer_buy_sell.py - lit-direction classification
- darkpool_analysis/analytics.py - lit_direction_daily + daily_metrics + index aggregation
- darkpool_analysis/table_renderer.py - daily table outputs (HTML/PNG)
- darkpool_analysis/plotter.py - multi-panel metric plots (Phase C)
- darkpool_analysis/data/darkpool.duckdb - persistent storage
- darkpool_analysis/data/constituents/spx_sample.csv - sample constituent list (replace with full coverage)
- database_check.ipynb - validation notebook
- darkpool_analysis/output/tables - table outputs
- darkpool_analysis/output/plots - plots (Phase C)

## Execution Flow (per target date)
1. Load config and env vars.
2. Fetch FINRA OTC weekly data (latest week <= target_date) and store finra_otc_weekly_raw.
3. Fetch FINRA daily short sale data for target_date and store finra_short_daily_raw.
4. Fetch Polygon trades (or aggregates fallback) and store polygon_equity_trades_raw.
5. Fetch Polygon daily aggregates for price context and store polygon_daily_agg_raw.
6. Compute lit_direction_daily (buy/sell volumes, ratios, log_buy_sell).
7. Build daily_metrics by combining lit_direction_daily, short ratio, price context, and OTC weekly volume.
8. Aggregate constituent short pressure to index-level daily totals.
9. Render daily table outputs (HTML/PNG). (Phase C plots later.)

## Data and Storage Details
- DuckDB is the single source of truth.
- Raw tables:
  - finra_otc_weekly_raw: symbol, week_start_date, off_exchange_volume, trade_count, tier/participant metadata
  - finra_short_daily_raw: symbol, trade_date, short_volume, short_exempt_volume, total_volume, market, source_file
  - polygon_equity_trades_raw: symbol, timestamp, price, size, bid, ask
  - polygon_daily_agg_raw: symbol, trade_date, open, high, low, close, vwap, volume
- Derived tables:
  - lit_direction_daily: symbol, date, lit_buy_volume, lit_sell_volume, lit_buy_ratio, log_buy_sell, classification_method, coverage, inference_version
  - daily_metrics: symbol, date, log_buy_sell, short_ratio, short_ratio_denominator_type/value, price/return context, pressure_context_label, data_quality, provenance flags, inference_version
  - index_constituent_short_agg_daily: index_symbol, trade_date, agg_short_ratio, coverage stats, index_return, interpretation_label
  - composite_signal (Phase C only)

## Data Sources Explained
- FINRA OTC weekly summary: authoritative off-exchange volume, published weekly with delay.
- FINRA daily short sale: short + short-exempt volume and total volume per facility during RTH; used as pressure signal only.
- Polygon trades: lit-market proxy for direction; NBBO when bid/ask present, TICK otherwise.
- Polygon daily aggregates: price/volume context and return metrics.

## Data Quality Flags
- OTC_ANCHORED: the OTC week covers the target date week.
- PRE_OTC: the OTC week is stale relative to the target date.

## Configuration Notes
- .env holds API secrets only (POLYGON_API_KEY, FINRA_API_KEY, FINRA_API_SECRET).
- config.py defines defaults (tickers, dates, RTH window, fetch mode, API URLs).
- FINRA_SHORT_SALE_* env vars control daily short sale ingestion.
- FINRA_OTC_* env vars control weekly OTC ingestion.
- POLYGON_* env vars control Polygon ingestion.

## FINRA API Authentication
- FINRA Query API requires OAuth 2.0 client_credentials flow.
- Token endpoint: `https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials`
- Use HTTP Basic Auth with FINRA_API_KEY:FINRA_API_SECRET to obtain Bearer token.
- Short sale API uses different field names than file format (tradeReportDate, securitiesInformationProcessorSymbolIdentifier, etc.).

## FINRA Query API Filtering
- Use `domainFilters` for multiple symbol filtering (NOT `orFilters` which is invalid).
- Use `dateRangeFilters` for date range filtering.
- Use `compareFilters` for single value comparisons.
- Example: `{"domainFilters": [{"fieldName": "securitiesInformationProcessorSymbolIdentifier", "fieldValues": ["RGTI", "AMZN"]}]}`

## Known Gaps and Watchouts
- Short sale "TotalVolume" is facility-specific, not total market volume.
- If total_volume is missing, short_ratio uses Polygon total volume and is marked as a proxy.
- OTC weekly data is delayed; daily metrics must flag PRE_OTC when stale.
- Some tickers may lack FINRA OTC coverage; keep rows with has_otc false.
- Index aggregation requires a maintained constituent list; static lists may drift over time.

## Tips for Future Contributors
- Keep the mandatory disclaimer in README and code comments.
- Do not imply observed dark-pool direction; all buy/sell values are inferred.
- Use the database_check.ipynb notebook to validate schema and data before plotting.
