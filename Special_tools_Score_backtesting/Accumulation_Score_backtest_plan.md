# Accumulation Signal Optimization Framework - Implementation Plan

## Objective
Build a systematic backtesting framework to find the optimal accumulation score configuration that maximizes predictive accuracy for swing trading (entry on accumulation, exit on distribution) with minimal false positives.

**Working Directory**: `C:\Users\fvign\Dropbox\Vscode\darkpool\Special_tools_Score_backtesting`

---

## Core Underlying Hypotheses

These are the fundamental beliefs driving this analysis:

1. **Institutional accumulation in darkpool/non-lit markets is a signal to go long**
2. **Institutional distribution is a signal to trim positions**
3. **Institutions do not day-trade** — they deploy large capital and stay in trades for days/weeks
4. **Correlating accumulation/distribution signals across the entire market can inform macro dynamics**

---

## Phase 1: Data Population (Manual - Run Tonight)

### 1.1 Date Range Requirements
- **Initial POC**: 2025 full year (Jan 1 - Dec 31, 2025) = ~252 trading days
- **Current data**: Oct 8, 2025 - Jan 13, 2026 (67 days)
- **Gap to fill**: Jan 1 - Oct 7, 2025 (~195 trading days)

### 1.2 Ticker Universe (Tiered Approach)

**Tier 1: ETF Screening Layer (15 ETFs)** - Run daily for macro read
```
GLOBAL_MACRO: SPY, QQQ, IWM, DIA, TLT, GLD, USO
SECTOR_CORE: XLF, XLK, XLE, XLV, XLY, XLI, XLP, XLU
```

**Tier 2: Sector Constituents** - Expand when ETF signals
```
From ticker_dictionary.py - top 10 holdings per sector ETF
~100 unique stocks across all sectors
```

**Tier 3: Always-On High Conviction (18 tickers)**
```
MAG8: MSFT, AAPL, GOOGL, AMZN, NVDA, AVGO, META, TSLA
SPECULATIVE: AMD, BE, CRWV, HIMS, HOOD, IREN, JPM, NNE, PLTR, SOFI
```

**Total for POC**: ~50-60 tickers (Tier 1 + Tier 3)
**Full expansion**: ~130 tickers (all tiers)

---

## Phase 2: Database Schema

### 2.1 Database Strategy: Same DB, Prefixed Tables

Use existing `darkpool.duckdb` with `bt_` prefix for backtest tables.

| Approach | Decision |
|----------|----------|
| Same DB, `bt_` prefixed tables | ✓ Easy joins to `daily_metrics`, single connection |
| Separate DB | ✗ Cross-DB joins painful, duplicate data |

### 2.2 New Tables (Minimal POC Schema)

**bt_signal_events** - Core backtest transaction log (start lean, expand in Phase B)
```sql
CREATE TABLE bt_signal_events (
    event_id INTEGER PRIMARY KEY,

    -- Core identity
    date DATE,
    symbol VARCHAR,
    sector_etf VARCHAR,             -- Parent ETF if applicable

    -- Signal config
    score_variant VARCHAR,          -- 'A', 'B', 'C', 'D'
    score_value FLOAT,              -- 0-100 display score
    buy_threshold INTEGER,
    sell_threshold INTEGER,

    -- Entry
    entry_price FLOAT,              -- Next-day open (realistic)
    signal_close FLOAT,             -- Close on signal day

    -- Trend context
    trend_state VARCHAR,            -- 'UP', 'DOWN', 'SIDEWAYS'
    sma_20_slope FLOAT,             -- Rate of change of 20d SMA
    price_vs_sma_20_pct FLOAT,      -- How far above/below SMA
    rsi_14 FLOAT,                   -- Overbought/oversold context

    -- Options context (liquid names only)
    opt_put_call_ratio FLOAT,       -- Day of signal
    opt_iv_percentile FLOAT,        -- IV rank 0-100
    opt_put_call_ratio_5d_chg FLOAT,-- Trend into signal

    -- Market regime
    regime VARCHAR,                 -- 'RISK_ON', 'RISK_OFF', 'TRANSITIONAL'
    spy_vs_sma_50 FLOAT,            -- SPY position relative to 50d SMA
    vix_level FLOAT,                -- VIX on signal day

    -- Outcomes
    fwd_return_5d FLOAT,
    fwd_return_10d FLOAT,
    fwd_return_20d FLOAT,
    days_to_first_dist INTEGER,
    days_to_2x_consec_dist INTEGER,
    return_at_first_dist FLOAT,
    return_at_2x_consec_dist FLOAT,
    max_drawdown_before_dist FLOAT,
    max_gain_before_dist FLOAT,

    -- Flags
    hit_at_dist BOOLEAN,            -- Profitable at distribution exit?
    hit_5d BOOLEAN,
    hit_10d BOOLEAN,
    parent_etf_aligned BOOLEAN,     -- Was parent ETF also accumulating?
    signal_quality VARCHAR,         -- 'HIGH', 'MEDIUM', 'LOW' (computed)
    failure_mode VARCHAR,           -- NULL if profitable, else reason

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**bt_backtest_runs** - Aggregated performance per configuration
```sql
CREATE TABLE bt_backtest_runs (
    run_id INTEGER PRIMARY KEY,
    run_timestamp TIMESTAMP,

    -- Configuration
    score_variant VARCHAR,
    z_window INTEGER,
    buy_threshold INTEGER,
    sell_threshold INTEGER,
    exit_strategy VARCHAR,          -- 'fixed_10d', 'first_dist', '2x_dist', 'score_decay'

    -- Universe & Regime
    ticker_universe VARCHAR,        -- 'TIER1', 'TIER1+3', 'ALL'
    regime_filter VARCHAR,          -- 'ALL', 'RISK_ON', 'RISK_OFF'
    date_start DATE,
    date_end DATE,

    -- Results
    total_signals INTEGER,
    total_trades INTEGER,
    hit_rate FLOAT,
    avg_return FLOAT,
    median_return FLOAT,
    win_loss_ratio FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    avg_hold_days FLOAT,
    signals_per_month FLOAT,

    -- Diagnostic
    false_positive_rate FLOAT,
    high_quality_hit_rate FLOAT     -- Hit rate for signal_quality='HIGH' only
);
```

**bt_frequency_baseline** - Natural rhythm analysis
```sql
CREATE TABLE bt_frequency_baseline (
    symbol VARCHAR,
    date_start DATE,
    date_end DATE,

    accum_signals_count INTEGER,
    dist_signals_count INTEGER,
    avg_gap_accum_to_dist FLOAT,
    median_gap_accum_to_dist FLOAT,
    avg_gap_dist_to_accum FLOAT,
    signals_per_month FLOAT
);
```

### 2.3 Signal Quality Logic (Computed Column)
```python
def compute_signal_quality(row):
    if row.score_value >= 80 and row.trend_state == 'DOWN' and row.opt_put_call_ratio < 0.7:
        return 'HIGH'  # Reversal setup with options confirmation
    elif row.score_value >= 75 and row.parent_etf_aligned:
        return 'HIGH'  # ETF + constituent alignment
    elif row.score_value >= 70 and (row.lit_z > 1.0 or row.parent_etf_aligned):
        return 'MEDIUM'
    else:
        return 'LOW'
```

### 2.4 Failure Mode Tracking
```python
FAILURE_MODES = [
    'IMMEDIATE_REVERSAL',  # Loss within 2 days
    'SLOW_BLEED',          # Gradual loss over hold period
    'GAP_DOWN',            # Overnight gap killed it
    'SECTOR_DRAG',         # ETF down, constituent followed
    'FALSE_SIGNAL',        # Score never confirmed (quick reversal to neutral)
]
```

---

## Phase 3: Context Enrichment

### 3.1 Trend Classification

| Trend State | Definition | Signal Interpretation |
|-------------|------------|----------------------|
| **UP** | Close > 20d SMA AND SMA slope > 0 | Accumulation = continuation |
| **DOWN** | Close < 20d SMA AND SMA slope < 0 | Accumulation = potential reversal |
| **SIDEWAYS** | Neither above | Accumulation = breakout setup |

**Implementation:**
```python
def classify_trend(close, sma_20, sma_20_prev):
    slope = (sma_20 - sma_20_prev) / sma_20_prev  # Normalized slope
    if close > sma_20 and slope > 0.001:
        return 'UP'
    elif close < sma_20 and slope < -0.001:
        return 'DOWN'
    else:
        return 'SIDEWAYS'
```

### 3.2 Options Premium Integration

**Data Source**: `options_premium_daily` and `options_premium_summary` tables (already in DB)

**Metrics to Capture** (for liquid names only: Tier 1 ETFs + MAG8):

| Metric | Why It Matters |
|--------|----------------|
| `put_call_ratio` | Low ratio = bullish sentiment |
| `iv_percentile_30d` | High IV + accumulation = potential squeeze |
| `put_call_ratio_5d_chg` | Declining = sentiment improving |
| `atm_iv` | Baseline volatility level |

**Key Hypothesis to Test:**
> Accumulation ≥ 70 + Put/Call ratio declining + IV elevated → Higher conviction
> Institution selling puts during accumulation → Reversal signal strength

**Liquidity Filter**: Only use options metrics if ticker has avg daily options volume > 10,000 contracts

### 3.2.1 Future Implementation: Abnormal Options Activity Detection (TODO)

**Status**: Not yet implemented. Infrastructure exists in `options_context.py` but not wired into backtest.

**Core Hypothesis to Test:**
> Abnormal increase in ITM/OTM put and call activity in the 0-5 days preceding a significant accumulation or VWBR signal may amplify signal quality.

**Metrics to Track:**

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| `itm_call_volume_z` | Z-score of ITM call volume vs 20d avg | Unusual ITM call buying = bullish positioning |
| `otm_call_volume_z` | Z-score of OTM call volume vs 20d avg | Speculative call buying = anticipation |
| `itm_put_volume_z` | Z-score of ITM put volume vs 20d avg | Defensive put buying or hedging |
| `otm_put_volume_z` | Z-score of OTM put volume vs 20d avg | Fear/protection buying |
| `itm_otm_call_ratio` | ITM calls / OTM calls | Higher = more conviction (in-the-money) |
| `itm_otm_put_ratio` | ITM puts / OTM puts | Higher = hedging vs speculation |
| `call_put_ratio_5d_chg` | Change in put/call ratio over 5 days | Direction of sentiment shift |
| `total_premium_5d_chg` | Change in total premium traded over 5 days | Size/conviction increasing? |

**Lookback Windows:**
- T-0: Day of signal
- T-1 to T-5: 5 trading days preceding signal
- Compare each day's activity to 20-day rolling average

**Signal Quality Boost Logic (Proposed):**
```python
def options_activity_boost(signal_row, options_lookback_df):
    """
    Check for abnormal options activity in 0-5 days before signal.
    Returns quality boost multiplier (1.0 = no boost, up to 1.5 for strong confirmation).
    """
    boost = 1.0

    # Bullish: Abnormal ITM call buying (z > 2) in lookback window
    if options_lookback_df['itm_call_volume_z'].max() > 2.0:
        boost += 0.15

    # Bullish: OTM call volume spike with declining put/call ratio
    if (options_lookback_df['otm_call_volume_z'].max() > 1.5 and
        options_lookback_df['call_put_ratio_5d_chg'].iloc[-1] > 0):
        boost += 0.10

    # Bullish: Premium size increasing (institutions scaling in)
    if options_lookback_df['total_premium_5d_chg'].iloc[-1] > 0.20:  # 20% increase
        boost += 0.10

    # Contrarian: High OTM put activity BUT accumulation signal = squeeze setup
    if (options_lookback_df['otm_put_volume_z'].max() > 2.0 and
        signal_row['score_value'] >= 75):
        boost += 0.15  # Potential short squeeze

    return min(boost, 1.5)  # Cap at 1.5x
```

**Implementation Steps (When Ready):**
1. Add ITM/OTM volume columns to `options_premium_daily` or create derived table
2. Extend `options_context.py` with `get_options_lookback()` function
3. Implement `add_options_context()` function (currently missing)
4. Wire options loading into `parameter_sweep.py` and `walk_forward.py`
5. Add options quality boost to signal classification logic

### 3.3 Market Regime Filter

| Regime | Definition | When to Use |
|--------|------------|-------------|
| **RISK_ON** | SPY > 50d SMA AND VIX < 20 | Test separately |
| **RISK_OFF** | SPY < 50d SMA AND VIX > 25 | Test separately |
| **TRANSITIONAL** | Mixed signals | Skip initially |

**Why Segment by Regime:**
- Optimal accumulation score config might be *different* per regime
- Testing across regimes blindly produces "compromise" config mediocre in both
- Test each regime first, then combined

---

## Phase 4: Signal Evaluation Framework

### 4.1 Folder Structure

```
Special_tools_Score_backtesting/
├── README.md                     # This plan
├── __init__.py
├── config.py                     # Backtest-specific configs
├── schema.py                     # bt_ table definitions
├── signal_generator.py           # Generate signals with config
├── forward_returns.py            # Compute outcomes
├── exit_detector.py              # Distribution exit logic
├── trend_classifier.py           # UP/DOWN/SIDEWAYS logic
├── options_context.py            # Options premium metrics
├── regime_classifier.py          # RISK_ON/RISK_OFF logic
├── metrics.py                    # Hit rate, Sharpe, etc.
├── parameter_sweep.py            # Grid search
├── walk_forward.py               # Validation harness
├── analysis_notebook.ipynb       # Interactive analysis
└── output/
    ├── signal_events.parquet
    ├── backtest_runs.csv
    └── charts/
```

### 4.2 Signal Generator Logic

```python
def generate_signals(
    df: pd.DataFrame,           # daily_metrics
    score_variant: str,         # 'A', 'B', 'C', 'D'
    short_z_source: str,        # Column name for short_z
    weights: dict,              # {short: 0.55, lit: 0.30, price: 0.15}
    buy_threshold: int = 70,
    sell_threshold: int = 30,
) -> pd.DataFrame:
    """
    Recompute accumulation score with given config.
    Return DataFrame with signal flags.
    """
```

### 4.3 Exit Strategy Implementations

| Strategy | Logic | Implementation |
|----------|-------|----------------|
| `fixed_Nd` | Exit after N trading days | `exit_date = entry_date + N` |
| `first_dist` | Exit on first distribution signal | `exit_date = min(date where score <= sell_threshold)` |
| `2x_consec_dist` | Exit on 2 consecutive distribution days | Requires 2-day lookback |
| `score_decay` | Exit if score < 50 for 3 days | Rolling 3-day check |
| `hybrid` | First of: distribution OR max_days | `min(first_dist, entry + max_days)` |

---

## Phase 5: Parameter Space

### 5.1 Variables to Test

**Score Variant (short_z source)**
- A: `finra_buy_volume_z`
- B: `short_buy_sell_ratio_z` (current baseline)
- C: `0.5*A + 0.5*B` (blended)
- D: `zscore(imbalance * log1p(volume))`

**Component Weights** (must sum to 1.0 or normalize)
```
w_short: [0.40, 0.50, 0.55, 0.60, 0.70]
w_lit:   [0.20, 0.25, 0.30, 0.35, 0.40]
w_price: [0.10, 0.15, 0.20]
```

**Z-Score Window**
```
z_window: [10, 15, 20, 30, 40, 60, 90]
```

**Thresholds**
```
buy_threshold:  [60, 65, 70, 75, 80]
sell_threshold: [20, 25, 30, 35, 40]
```

**Exit Strategy**
```
exit: ['fixed_10d', 'fixed_20d', 'first_dist', '2x_consec_dist', 'hybrid_30d']
```

### 5.2 Grid Search Phases

**Phase A: Coarse Sweep** (~100 combinations)
- Score variants: A, B, C, D
- Weights: 3 presets (short-heavy, balanced, lit-heavy)
- Z-window: 15, 20, 30
- Thresholds: 70/30 only
- Exit: first_dist, 2x_consec_dist

**Phase B: Fine-Tune** (top 5 from Phase A)
- Weights: 0.05 increments around best
- Z-window: all 7 options
- Thresholds: all combinations
- Exit: all strategies

---

## Phase 6: Walk-Forward Validation

### 6.1 2025 POC Structure
```
┌─────────────────────────────────────────────────────────────────┐
│ Jan-Jun 2025 (Train)  │ Jul-Aug (Test) │ Sep-Oct (Train) │ Nov-Dec (Test) │
└─────────────────────────────────────────────────────────────────┘
```

- 6-month train windows
- 2-month test windows
- Roll forward, repeat
- Aggregate out-of-sample results

### 6.2 Cross-Sectional Validation
- Same parameters across ALL tickers simultaneously
- Check: Does signal work broadly or only on specific names?

---

## Phase 7: Evaluation Metrics

### 7.1 Primary Metrics (Optimize These)
| Metric | Formula | Target |
|--------|---------|--------|
| Hit Rate | profitable_trades / total_trades | > 55% |
| Avg Return | mean(trade_returns) | > 1% per trade |
| Win/Loss Ratio | avg_win / avg_loss | > 2.0 |
| Signal Efficiency | (signals that hit) / (total signals) | Higher = better |

### 7.2 Diagnostic Metrics (Track, Don't Optimize)
| Metric | Purpose |
|--------|---------|
| Max Drawdown | Risk tolerance check |
| Avg Hold Days | Matches swing trading? |
| Signal Frequency | Too many = noise |
| False Positive Rate | Signals that reverse within 3 days |

### 7.3 Composite Ranking Score
```
Rank Score = (Hit Rate × Avg Return) × Win/Loss Ratio × (1 / sqrt(Signals/Month))
```

---

## Phase 8: Output & Analysis

### 8.1 Output Files
```
Special_tools_Score_backtesting/output/
├── signal_events.parquet       # All generated signals
├── backtest_runs.csv           # Summary per config
├── frequency_baseline.csv      # Per-ticker signal rhythm
├── parameter_sensitivity.png   # Heatmaps of parameter impact
├── regime_comparison.png       # RISK_ON vs RISK_OFF performance
├── equity_curves/              # Per-config equity curves
│   ├── variant_A_balanced.png
│   └── ...
└── walk_forward_summary.csv    # Out-of-sample aggregates
```

### 8.2 Analysis Notebook Sections
1. Signal frequency baseline (per ticker)
2. Parameter sensitivity heatmaps
3. Top 10 configurations comparison
4. Walk-forward equity curves
5. Sector breakdown analysis
6. ETF + constituent alignment test
7. Failure mode analysis (when signals fail)

---

## Implementation Order

| Step | Task | Dependency | Notes |
|------|------|------------|-------|
| 1 | **Data population** (manual tonight) | None | Backfill Jan-Oct 2025 for Tier 1+3 tickers |
| 2 | Create `bt_` tables in db.py | Step 1 | Minimal schema first |
| 3 | Build `trend_classifier.py` | Step 1 | UP/DOWN/SIDEWAYS |
| 4 | Build `regime_classifier.py` | Step 1 | RISK_ON/RISK_OFF |
| 5 | Build `options_context.py` | Step 1 | Put/call, IV metrics |
| 6 | Build `signal_generator.py` | Step 2 | Core signal logic |
| 7 | Build `forward_returns.py` + `exit_detector.py` | Step 6 | Outcome computation |
| 8 | Build `metrics.py` | Step 7 | Hit rate, Sharpe, etc. |
| 9 | Run Phase 0: signal frequency baseline | Step 7 | Understand natural rhythm |
| 10 | Run coarse grid search (~50 configs) | Step 8 | Score variants A-D |
| 11 | Segment by regime, compare results | Step 10 | RISK_ON vs RISK_OFF |
| 12 | Add options context to top configs | Step 11 | Phase B enrichment |
| 13 | Fine-tune top 5 configs | Step 12 | Weights, thresholds |
| 14 | Walk-forward validation | Step 13 | Out-of-sample testing |
| 15 | Analysis notebook | Step 14 | Visualize results |

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `darkpool_analysis/db.py` | Add `bt_signal_events`, `bt_backtest_runs`, `bt_frequency_baseline` tables |
| `darkpool_analysis/analytics.py` | Extract score computation to reusable function with config override |

## New Files to Create (in `Special_tools_Score_backtesting/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `config.py` | Backtest-specific configs |
| `schema.py` | `bt_` table definitions |
| `trend_classifier.py` | UP/DOWN/SIDEWAYS classification |
| `regime_classifier.py` | RISK_ON/RISK_OFF classification |
| `options_context.py` | Options premium metrics extraction |
| `signal_generator.py` | Generate signals with arbitrary config |
| `forward_returns.py` | Compute forward returns |
| `exit_detector.py` | Distribution exit detection |
| `metrics.py` | Performance metrics |
| `parameter_sweep.py` | Grid search |
| `walk_forward.py` | Walk-forward validation |
| `analysis_notebook.ipynb` | Interactive analysis |

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Entry timing** | Next-day Open | Realistic: signal fires after close, enter at next open |
| **Overlap handling** | Ignore new signal | One position per ticker at a time, simpler tracking |
| **ETF alignment test** | Phase B (Fine-tune) | Establish baseline first, layer on top performers |
| **Data scope** | Full 2025 | ~252 trading days for proper walk-forward validation |

---

## Machine Learning Consideration

**Recommendation**: Start without ML. The parameter space is tractable with grid search + walk-forward.

**When ML would help**:
- After establishing baseline with grid search
- If you want to learn non-linear feature interactions
- For regime detection (which config works in which market state)

**Potential ML approaches** (future):
- Gradient boosting (XGBoost) to predict forward returns from features
- LSTM for sequence patterns in accumulation signals
- Ensemble: combine multiple score variants with learned weights
