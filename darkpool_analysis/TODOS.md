
# Visualization improvement
- [ ] user toggle to have the plot in log scale and absolute (absolute start axis at 0, absolute is the value in the datbase.


# Tbale improvment
- [ ] Sell ratio

# architecture
- [ ] can we ingest entire table and just querry them first to sdecide which 'volume' is worth analyzing

# Code Logic
- [ ] Examine if we are inputting the right data 
- [ ] does having tick data really better
- [ ] delta normalizaiton (see older GPT note on that topic)


# macro signaling
- [ ] can we assmeble the right set of ticker to have a 'macro' signal on sector rotation or bullish/bearish overall trend
- [ ] Can the buy and sell code also be used on no darkpool data on all stock, and converted into an invered buy/sell put call?

---

# FINRA Data Enhancements (Dec 2025)

## Phase 1: Add FINRA Week Visibility Column
**Priority:** High | **Complexity:** Low

**Goal:** Add transparency to show when data is "actual" (FINRA+Polygon same week) vs "estimated" (stale FINRA + current Polygon).

**Background:**
- FINRA OTC data has 2-4 week publication delay
- Currently we apply stale FINRA weekly volume to current Polygon daily ratios
- Users should see which FINRA week underlies each daily estimate

**Implementation:**
- [x] Add `finra_week_used DATE` column to `darkpool_daily_summary` table in `db.py`
- [x] Add `finra_week_used DATE` column to `darkpool_estimated_flow` table in `db.py`
- [x] Modify `build_darkpool_estimates()` in `analytics.py` to accept and store `finra_week` parameter
- [x] Modify `build_daily_summary()` in `analytics.py` to propagate `finra_week_used`
- [x] Update `orchestrator.py` to pass `finra_week` through the pipeline
- [x] Add `data_quality` column to table renderer (shows "ACTUAL" if same week, "ESTIMATED (N weeks stale)" if older)
- [x] Update table HTML template to display FINRA week and quality flag

**Output Example:**
```
date       | symbol | buy_ratio | finra_week_used | data_quality
2025-12-22 | AMZN   | 1.38      | 2025-12-01      | ESTIMATED (3 weeks stale)
2025-12-01 | AMZN   | 1.25      | 2025-12-01      | ACTUAL
```

---

## Phase 2: Support Polygon-Only Tickers
**Priority:** Medium | **Complexity:** Medium

**Goal:** Show lit market buy/sell inference for tickers not found in FINRA data, instead of hiding them.

**Background:**
- Some tickers may not have FINRA OTC data (e.g., newer stocks, certain ETFs)
- Currently these are silently dropped from output
- We should still show Polygon's lit market directional signal

**Implementation:**
- [x] Modify `build_darkpool_estimates()` to use RIGHT JOIN from Polygon ratios instead of LEFT JOIN from FINRA
- [x] Handle NULL FINRA volume gracefully - still compute and store the row
- [x] Add `has_finra_data BOOLEAN` column to summary table
- [x] Update table renderer to show "Polygon-only" or "No FINRA volume" for these rows
- [x] Update plotter to include these tickers (with different styling or annotation)
- [x] Add config option `INCLUDE_POLYGON_ONLY_TICKERS` (default: True)

**Output Example:**
```
date       | symbol | lit_buy_ratio | finra_volume | status
2025-12-22 | NVDA   | 0.62          | 50,000,000   | OK
2025-12-22 | PLTR   | 0.58          | NULL         | Polygon-only (no FINRA)
```

---

## Phase 3: Add Daily Short Sale Volume (NEW FINRA Dataset)
**Priority:** High | **Complexity:** High

**Goal:** Integrate FINRA's Reg SHO Daily Short Sale Volume API for true daily granularity dark pool data.

**Background:**
- FINRA OTC Weekly Summary = weekly aggregates with 2-4 week delay
- FINRA Reg SHO Daily Short Sale = **DAILY** data with **no delay**
- This is the only FINRA dataset with true daily granularity
- Short sale data provides direct insight into bearish positioning

**API Details:**
- Endpoint: `https://api.finra.org/data/group/otcMarket/name/regShoDaily`
- Fields: symbol, tradeDate, shortVolume, totalVolume, shortRatio
- No authentication required (public API)
- Daily granularity, same-day availability

**Implementation:**
- [ ] Create `fetch_finra_short.py` - new module for short sale API fetching
- [ ] Add `FINRA_SHORT_SALE_URL` to `config.py`
- [ ] Add `finra_short_sale_daily` table to `db.py` schema:
  ```sql
  CREATE TABLE finra_short_sale_daily (
      date DATE,
      symbol TEXT,
      short_volume DOUBLE,
      total_volume DOUBLE,
      short_ratio DOUBLE,
      PRIMARY KEY (date, symbol)
  )
  ```
- [ ] Add `fetch_finra_short_sale()` function with date filtering
- [ ] Update `orchestrator.py` to fetch short sale data in pipeline
- [ ] Add `short_ratio` column to `darkpool_daily_summary` table
- [ ] Update `table_renderer.py` to display short sale ratio with color coding
- [ ] Create separate short sale ratio plot in `plotter.py` (optional: overlay on buy_ratio plot)
- [ ] Add toggle for short sale visualization in config

**Output Example:**
```
date       | symbol | buy_ratio | short_ratio | interpretation
2025-12-22 | AMZN   | 1.38      | 0.45        | Accumulation + 45% short
2025-12-22 | TSLA   | 0.72      | 0.62        | Distribution + 62% short
```

**New Plot:**
Short Sale Ratio chart (0-100%) showing daily short sale pressure per ticker.

---

## References
- [FINRA Reg SHO Daily Short Sale Volume API](https://developer.finra.org/docs/api-explorer/query_api-equity-reg_sho_daily_short_sale_volume)
- [FINRA Short Sale Volume Data](https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data)
- [FINRA API Catalog](https://developer.finra.org/catalog)
- [FINRA OTC Transparency Data](https://www.finra.org/filing-reporting/otc-transparency-data)

