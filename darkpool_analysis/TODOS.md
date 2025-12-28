# FINRA + Polygon Accumulation/Distribution Roadmap

## Phase A - Separate ingestion + independent outputs
- [x] A1: Update config/db schema for new raw and derived tables.
- [x] A2: Implement FINRA OTC weekly ingestion (fetch_finra_otc.py).
- [x] A3: Implement FINRA daily short sale ingestion (fetch_finra_short.py).
- [x] A4: Implement Polygon daily aggregates ingestion (fetch_polygon_agg.py).
- [x] A5: Confirm Polygon trades ingestion path (fetch_polygon_equity.py).
- [x] A6: Build lit_direction_daily + daily_metrics derivations.
- [x] A7: Add price context metrics (close, return, z-score, range).
- [x] A8: Add short pressure metrics (short_ratio + z-score, explicit denominator).

## Phase B - Data provenance + anchor labeling
- [x] B1: Add data_quality flag (OTC_ANCHORED vs PRE_OTC).
- [x] B2: Add has_otc / has_short / has_lit / has_price flags and short_ratio_source.
- [x] B3: Add pressure_context_label derived from return + short_ratio z-score.
- [x] B4: Update table renderer to include provenance columns.

## Phase B2 - Constituent aggregation
- [x] B2a: Add index_constituent_short_agg_daily table.
- [x] B2b: Implement constituent aggregation with coverage stats and index proxy returns.

## Validation
- [x] V1: Create database_check.ipynb for schema + provenance checks.
- [x] V2: Align daily table renderer to daily_metrics outputs.

## Bug Fixes (2025-12-24)
- [x] F1: Fix FINRA OAuth 2.0 authentication (set correct FIP token URL).
- [x] F2: Fix FINRA short sale API field name mapping (tradeReportDate, securitiesInformationProcessorSymbolIdentifier, etc.).
- [x] F3: Fix pandas merge_asof dtype error (convert date columns to datetime64).
- [x] F4: Fix KeyError 'symbol' after merge_asof (drop duplicate column before merge).
- [x] F5: Fix FutureWarning for isin with datetime64 (convert to .dt.date before comparison).

## Phase C - Composite interpretation + visualization
- [ ] C1: Implement composite_signal table/view with inference_version.
- [x] C2: Add multi-panel plotter (short sale buy ratio, lit buy + log ratios, OTC buy/sell + decision strip).

## Table + Metric Revisions (2025-12-25)
- [x] T1: Consolidate per-venue short sale rows into one daily total for ratios.
- [x] T2: Add short/lit/OTC buy/sell volumes, buy ratios, and z-scores to daily_metrics.
- [x] T3: Add OTC status and OTC week used fields for table/plot display.
- [x] T4: Rebuild table layout into Short Sale / Lit / OTC sections with ticker averages and summary panel.


## faster code
- [] user toggle to feth LIT data as OHLCV vs 1 min
- [] refactor and identify onsolete part of code
- [] figure out how to leverage OTC weelly and plot it porperly

## overla taks
- [] Work on how to best visualize panel 3 (OTC weekly) so that this plot is more intuitive and useful to look at.
- [] if the OTC wekkly report is stable, don't carry it past it's know week.
- [] Generate a merge Plot  and table for multiple tickers (key sector indicators), to get a bit more of a macro shift view
- [] I built an automated daily scanner that reports the top 50 tickers whose z-score suddenly blow up (to catch tickers we would not usually look at from the 10,000 Finra daily list. I already have a foundation in place, but I need to improve the mathematical logic.
- [] Same scanner, if filtered for volume or market cap and tracked as a plot over time, could also give us a macro signal of the entire market.
- [] Add figure expansion on width based on number of days being plotted
- [] Add price chart at the bottom
- [] don't overwrite tables

