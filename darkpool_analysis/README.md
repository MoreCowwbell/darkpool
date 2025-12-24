# Institutional Accumulation / Distribution (FINRA + Polygon)

Production-quality Python analytics for end-of-day institutional accumulation/distribution analysis using:
- FINRA OTC weekly summary (authoritative off-exchange anchor)
- FINRA daily short sale volume (timely pressure signal)
- Polygon lit-market trades (directional proxy)
- Polygon daily aggregates (price context)

Mandatory disclaimer:
"FINRA does not publish trade direction for off-exchange volume. Buy/Sell values are inferred estimates derived from lit-market equity trades and applied proportionally to FINRA OTC volume."

## Objectives
- Ingest and persist raw datasets in DuckDB (no mixing).
- Compute daily metrics with explicit provenance flags.
- Render daily tables for configured tickers.
- Phase C (later): composite interpretation layer and multi-panel plots.

Phase C plotting is intentionally not implemented until database validation passes.

## Data Sources
- FINRA OTC weekly summary (ATS + non-ATS off-exchange volume, delayed).
- FINRA Reg SHO daily short sale volume files (RTH only, facility-specific).
- Polygon equity trades for lit-market directional inference (NBBO if available, TICK fallback).
- Polygon daily aggregates for close/return context.

## Project Layout
```
darkpool_analysis/
├── orchestrator.py
├── config.py
├── db.py
├── fetch_finra_otc.py
├── fetch_finra_short.py
├── fetch_polygon_equity.py
├── fetch_polygon_agg.py
├── infer_buy_sell.py
├── analytics.py
├── table_renderer.py
├── data/
│   └── darkpool.duckdb
└── output/
    ├── tables/
    └── plots/
```

## Setup
```
python -m pip install -r requirements.txt
```

## Configuration (Environment Variables)
Core:
- TICKERS (comma-separated)
- FETCH_MODE ("single", "daily", "weekly")
- TARGET_DATE (YYYY-MM-DD)
- BACKFILL_COUNT (int)
- MIN_LIT_VOLUME (default 10000)
- MARKET_TZ (default US/Eastern)
- RTH_START, RTH_END (default 09:30, 16:15)
- INFERENCE_VERSION (e.g., PhaseA_v1)
- EXPORT_CSV (true/false)

Polygon:
- POLYGON_API_KEY (required for API fetch)
- POLYGON_BASE_URL (default https://api.polygon.io)
- POLYGON_TRADES_FILE (optional local CSV/JSON override)

FINRA OTC weekly:
- FINRA_OTC_URL (API endpoint)
- FINRA_OTC_FILE (optional local CSV/JSON override)
- FINRA_API_KEY, FINRA_API_SECRET (optional headers or token auth)
- FINRA_TOKEN_URL (optional OAuth token endpoint)
- FINRA_REQUEST_METHOD (GET or POST)
- FINRA_REQUEST_JSON (JSON string for POST body)
- FINRA_REQUEST_PARAMS (JSON string for query params)
- FINRA_DATE_FIELD, FINRA_SYMBOL_FIELD, FINRA_VOLUME_FIELD, FINRA_TRADE_COUNT_FIELD (optional column overrides)

FINRA daily short sale:
- FINRA_SHORT_SALE_URL (API endpoint, optional)
- FINRA_SHORT_SALE_FILE (optional single file override)
- FINRA_SHORT_SALE_DIR (optional directory of daily files)

Polygon aggregates:
- POLYGON_DAILY_AGG_FILE (optional single file override)
- POLYGON_DAILY_AGG_DIR (optional directory of daily files)

Index constituents:
- INDEX_CONSTITUENTS_DIR (default data/constituents)
- INDEX_CONSTITUENTS_FILE (optional single file override)
- INDEX_PROXY_MAP (JSON map, e.g., {"SPX":"SPY"})

## Run
```
python orchestrator.py
```

## Validation (Required Before Plotting)
Open and run:
- database_check.ipynb

## Outputs
- DuckDB database: darkpool_analysis/data/darkpool.duckdb
- Tables: darkpool_analysis/output/tables/
- Plots: darkpool_analysis/output/plots/ (Phase C only)

## Notes on Inference
- Lit trades are classified NBBO-first, TICK fallback.
- log(Buy/Sell) is computed only when both buy and sell volumes are > 0.
- Short ratio uses (ShortVolume + ShortExemptVolume) / TotalVolume when available; otherwise uses Polygon daily volume as the denominator and flags it as POLYGON_TOTAL.
- OTC weekly data is delayed; daily metrics are labeled OTC_ANCHORED or PRE_OTC accordingly.
- Price context is sourced from Polygon daily aggregates only.
- Constituent aggregation requires a maintained index list; see data/constituents/spx_sample.csv as a format example.
