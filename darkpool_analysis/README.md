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
- Generate multi-panel plots for pressure analysis (Phase C).

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
├── plotter.py
├── plotter_chart.py
├── data/
│   └── darkpool.duckdb
└── output/
    ├── tables/
    └── plots/
    └── price_charts/
```

## Setup
```
python -m pip install -r requirements.txt
```

## Configuration (Environment Variables)
Core:
- TICKERS (comma-separated)
- TICKERS_TYPE (DEFAULT, SECTOR_CORE, GLOBAL_MACRO, COMMODITIES)
- FETCH_MODE ("single", "daily", "weekly")
- TARGET_DATE (YYYY-MM-DD)
- BACKFILL_COUNT (int)
- MIN_LIT_VOLUME (default 10000)
- MARKET_TZ (default US/Eastern)
- RTH_START, RTH_END (default 09:30, 16:15)
- INFERENCE_VERSION (e.g., PhaseA_v1)
- EXPORT_CSV (true/false)
- RENDER_PRICE_CHARTS (true/false)
- PRICE_BAR_TIMEFRAME (daily, weekly, monthly)

Polygon:
- POLYGON_API_KEY (required for API fetch)
- POLYGON_BASE_URL (default https://api.polygon.io)
- POLYGON_TRADES_FILE (optional local CSV/JSON override)

FINRA OTC weekly:
- FINRA_OTC_URL (API endpoint)
- FINRA_OTC_FILE (optional local CSV/JSON override)
- FINRA_API_KEY, FINRA_API_SECRET (required for OAuth 2.0 authentication)
- FINRA_TOKEN_URL (OAuth token endpoint, defaults to FINRA Identity Platform FIP)
- FINRA_REQUEST_METHOD (GET or POST)
- FINRA_REQUEST_JSON (JSON string for POST body)
- FINRA_REQUEST_PARAMS (JSON string for query params)
- FINRA_DATE_FIELD, FINRA_SYMBOL_FIELD, FINRA_VOLUME_FIELD, FINRA_TRADE_COUNT_FIELD (optional column overrides)

FINRA daily short sale:
- FINRA_SHORT_SALE_URL (API endpoint, optional)
- FINRA_SHORT_SALE_FILE (optional single file override)
- FINRA_SHORT_SALE_DIR (optional directory of daily files)
- FINRA_CDN_URL (scanner CDN base, default https://cdn.finra.org/equity/regsho/daily)

Scanner:
- SCANNER_LOOKBACK_DAYS (default 90)
- SCANNER_TREND_DAYS (default 3)
- SCANNER_TOP_N (default 50)
- SCANNER_OUTLIER_Z (default 2.0)
- SCANNER_VOLUME_Z (default 2.0)
- SCANNER_RATIO_Z (default 2.0)
- SCANNER_EXPORT_FULL (true/false, default true)
- SCANNER_INFERENCE_VERSION (default Scanner_v1)

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
python main.py -scanner
```

## Validation (Required Before Plotting)
Open and run:
- database_check.ipynb

## Outputs
- DuckDB database: darkpool_analysis/data/darkpool.duckdb
- Tables: darkpool_analysis/output/tables/ (HTML/PNG)
- Plots: darkpool_analysis/output/plots/ (multi-panel PNG per ticker)
- Price charts: darkpool_analysis/output/price_charts/ (OHLCV PNG per ticker)
- Scanner: darkpool_analysis/output/scanner/YYYY-MM-DD/
  - scanner_metrics.csv (all symbols)
  - scanner_top.csv (top N by scanner score)
  - scanner_top.png (bar chart of top N)
  - scanner_scores.png (score distribution histogram)

## Scanner Notes
- Scanner uses FINRA CDN short sale data only (no Polygon tick/minute data).
- Results are written to scanner_daily_metrics and ranked by trend/outlier/volume z-scores.

## Table Styling
Table presentation is controlled by `DEFAULT_TABLE_STYLE` in `darkpool_analysis/config.py`.
Key knobs:
- `mode`: "scan" (compact) or "analysis" (roomier)
- `base_font_size`, `header_font_size`, `numeric_font_scale`
- `row_padding_y`, `row_padding_x`, `group_alt_strength`
- `neutral_text_opacity`, `signal_opacity_strong`, `signal_opacity_muted`
- `group_separator_px`
- `palette` (muted green/red and neutral tones)

## Plot Modes
- layered (default): 4-panel visualization with short sale buy ratio, lit flow imbalance, OTC participation, and accumulation score.
- short_only: short ratio, short sale volume, close price.
- both: render layered and short_only together.

Example:
```
python plotter.py --dates 2025-12-20 --mode short_only
python plotter_chart.py --dates 2025-12-20 --timeframe daily
```

## How to Read the Plot (Layered Mode)

**Panel 1: Short Sale Buy/Sell Ratio (~50% height)**
- Primary signal: Short volume / (Total volume - Short volume)
- Thresholds: >1.25 = Bullish (BOT), <0.75 = Bearish (SELL), ~1.0 = Neutral
- Agreement markers: Green ▲ = short+lit both bullish, Red ▼ = both bearish, Yellow ◆ = divergence

**Panel 2: Lit Flow Imbalance (~20% height)**
- Confirmation signal: (LitBuy - LitSell) / (LitBuy + LitSell)
- Range: [-1, +1], centered at 0
- Thresholds: ±0.1 (weak signal), ±0.2 (strong signal)
- NULL if lit volume < MIN_LIT_VOLUME (insufficient data)

**Panel 3: OTC Participation (~15% height)**
- Institutional activity intensity: OTC_weekly_volume / Weekly_total_volume
- Color by z-score: Cyan (elevated), Yellow (normal), Gray (low/stale)
- Week-over-week delta bars: Green = rising, Red = falling
- NOTE: OTC direction is NOT observable; this measures participation only

**Panel 4: Accumulation Score (~10% height)**
- Composite score: 0-100 scale (Red <30 → Gray 50 → Green >70)
- Weights: 55% short z-score, 30% lit z-score, 15% price z-score
- Intensity modulation: High OTC participation amplifies signal, low OTC dampens
- Confidence bar: White bar below shows data quality (higher = more reliable)

## Key Metric Formulas

**Short Sale Buy/Sell Ratio:**
```
short_buy_volume = short_volume (from FINRA)
short_sell_volume = total_volume - short_volume
short_buy_sell_ratio = short_buy_volume / short_sell_volume
```

**Lit Flow Imbalance (bounded [-1, +1]):**
```
lit_flow_imbalance = (lit_buy - lit_sell) / (lit_buy + lit_sell)
NULL if (lit_buy + lit_sell) < MIN_LIT_VOLUME
```

**OTC Participation Rate (proxy):**
```
otc_participation_rate = otc_weekly_volume / weekly_total_volume
⚠️ FINRA OTC = all-hours, Polygon = RTH-biased; ratio is a proxy
```

**Accumulation Score (0-100):**
```
raw_score = 0.55 * tanh(short_z * 0.5) + 0.30 * tanh(lit_z * 0.5) + 0.15 * tanh(price_z * 0.3)
intensity_scale = 0.7 + 0.6 * sigmoid(otc_participation_z)  # range [0.7, 1.3]
accumulation_score = clip(raw_score * intensity_scale, -1, 1)
display_score = (accumulation_score + 1) * 50  # map to 0-100
```

**Confidence Score (0.25-1.0):**
```
staleness_penalty: Anchored=1.0, Staled=0.7, None=0.5
coverage_penalty: all data=1.0, missing one=0.8, missing 2+=0.5
confidence = staleness_penalty * coverage_penalty * (0.9 if lit_flow_imbalance NULL else 1.0)
```

## Notes on Inference
- Lit trades are classified NBBO-first, TICK fallback.
- log(Buy/Sell) is computed only when both buy and sell volumes are > 0.
- Short Sale Buy Ratio uses ShortVolume / ShortSellVolume (short-exempt excluded); if total volume is missing, Polygon daily volume is used to derive the sell volume and flagged as POLYGON_TOTAL.
- OTC weekly data is delayed; OTC status is Anchored, Staled, or None based on the week used.
- Price context is sourced from Polygon daily aggregates only.
- Constituent aggregation requires a maintained index list; see data/constituents/spx_sample.csv as a format example.

## FINRA API Notes
- FINRA Query API requires OAuth 2.0 authentication via the FINRA Identity Platform (FIP).
- Token endpoint: `https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials`
- Short sale API field names differ from file format:
  - `tradeReportDate` (not `Date`)
  - `securitiesInformationProcessorSymbolIdentifier` (not `Symbol`)
  - `shortParQuantity` (not `ShortVolume`)
  - `shortExemptParQuantity` (not `ShortExemptVolume`)
  - `totalParQuantity` (not `TotalVolume`)
  - `marketCode` (not `Market`)
