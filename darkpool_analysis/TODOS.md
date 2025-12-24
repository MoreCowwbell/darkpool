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

## Phase C - Composite interpretation + visualization
- [ ] C1: Implement composite_signal table/view with inference_version.
- [ ] C2: Add multi-panel plotter (log_buy_sell + short ratio + OTC anchor).
