# Dark Pool ETF Analysis (Option B)

Production-quality Python analytics for end-of-day dark pool ETF analysis using FINRA OTC volume and lit-market trade inference.

Mandatory disclaimer:
"FINRA does not publish trade direction for off-exchange volume. Buy/Sell values are inferred estimates derived from lit-market equity trades and applied proportionally to FINRA OTC volume."

## Objective
- Generate a daily summary table with estimated buy/sell volumes, buy ratio, and total off-exchange volume.
- Generate historical buy-ratio plots per ticker with threshold annotations.
- Persist all raw and derived data in DuckDB as the single source of truth.
- Save tables and plots locally for portability.

## Data Sources
- FINRA OTC Transparency (ATS + non-ATS) equity volume (weekly authoritative).
- Polygon equity trades for lit-market proxy (bid/ask if available; otherwise tick rule).

## Assumptions and Constraints
- Dark pool = FINRA OTC equity volume only.
- FINRA does not publish trade direction.
- Options flow is out of scope.
- SPXW is excluded (no FINRA equity volume).
- Regular trading hours only (09:30-16:00 ET).
- DuckDB is the only persistence layer; values stored as float64.

## Project Layout
```
darkpool_analysis/
├── orchestrator.py
├── config.py
├── db.py
├── fetch_finra.py
├── fetch_polygon_equity.py
├── infer_buy_sell.py
├── analytics.py
├── plotter.py
├── data/
│   └── darkpool.duckdb
├── output/
│   ├── tables/
│   └── plots/
├── .env
└── README.md
```

## Setup
```
python -m pip install -r requirements.txt
```

Copy or edit `darkpool_analysis/.env` with your API credentials and settings.

## Configuration (Environment Variables)
Core:
- `TICKERS` (default `TQQQ,SPY,SPXW`)
- `RUN_MODE` (`daily` or `weekly`)
- `TARGET_DATE` (YYYY-MM-DD; defaults to today in US/Eastern)
- `MIN_LIT_VOLUME` (default `10000`)
- `MARKET_TZ` (default `US/Eastern`)
- `RTH_START`, `RTH_END` (default `09:30`, `16:00`)
- `INFERENCE_VERSION` (default `OptionB_v1`)

Polygon:
- `POLYGON_API_KEY` (required for API fetch)
- `POLYGON_BASE_URL` (default `https://api.polygon.io`)
- `POLYGON_TRADES_FILE` (optional local CSV/JSON override)

FINRA:
- `FINRA_OTC_URL` (required for API fetch)
- `FINRA_OTC_FILE` (optional local CSV/JSON override)
- `FINRA_API_KEY`, `FINRA_API_SECRET` (optional headers or token auth)
- `FINRA_TOKEN_URL` (optional OAuth token endpoint)
- `FINRA_REQUEST_METHOD` (`GET` or `POST`)
- `FINRA_REQUEST_JSON` (JSON string for POST body)
- `FINRA_REQUEST_PARAMS` (JSON string for query params)
- `FINRA_DATE_FIELD`, `FINRA_SYMBOL_FIELD`, `FINRA_VOLUME_FIELD`, `FINRA_TRADE_COUNT_FIELD` (optional column overrides)

## Run
```
python orchestrator.py
```

## Outputs
- DuckDB database: `darkpool_analysis/data/darkpool.duckdb`
- Tables: `darkpool_analysis/output/tables/`
- Plots: `darkpool_analysis/output/plots/`

## Notes on Inference
- If bid/ask fields are present for all trades in a symbol/day, NBBO classification is used.
- If any bid/ask is missing, the tick rule is used for that symbol/day.
- Trades classified as neutral are excluded.
- If lit volume is below `MIN_LIT_VOLUME`, `lit_buy_ratio` is set to NULL and coverage metadata is persisted.

## Extending to New Tickers
1. Add symbols to `TICKERS` in `.env`.
2. Ensure FINRA data includes the symbol; otherwise it will be skipped.
3. Run `python orchestrator.py` to refresh tables and plots.
