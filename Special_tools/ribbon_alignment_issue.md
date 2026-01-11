# Ribbon Alignment Issue - Circos Chart

## Problem Statement

Chord ribbons in the circos visualization are **not centered on ticker labels**. When looking at tickers like META, AVGO, NVDA, or AMZN, it's impossible to determine which ticker a ribbon belongs to because the ribbon endpoints are offset from the label position.

### Visual Symptoms
1. Ribbons appear shifted away from their ticker labels
2. Adjacent tickers (e.g., META/AVGO) have ribbons that could belong to either
3. Even with fanning disabled, ribbons don't converge to the ticker center
4. Multiple ribbons create a "line with transparent fan" effect due to alpha blending

---

## Root Cause Analysis

### Issue 1: Band Allocation for Invisible Metrics

**Location:** `band_map` construction (Cell 6, lines ~2893-2951)

The original code allocated space for ALL 5 metrics in `BAND_ORDER` regardless of visibility:
```python
BAND_ORDER = ['lit', 'accum', 'short', 'finra_buy', 'vwbr_z']
```

When only `vwbr_z` is visible (position 5/5), it still received only 20% of the chord space and was positioned at the far right of the arc - not centered.

**Fix Applied:** Filter to visible metrics only:
```python
visible_band_order = [m for m in BAND_ORDER if metric_visible(m)]
```

### Issue 2: Multiple Connections Spread Across Arc

**Location:** `allocate_intervals()` function (Cell 6, lines ~2572-2680)

Even with a single ribbon per edge (`EDGE_RIBBON_SPLITS = 1`), multiple connections to the same ticker get distributed across the available arc space:

```python
slot_width_out = usable_out / n_out  # Each connection gets 1/n of the arc
```

The cursor advances after each ribbon placement:
```python
out_info['cursor'] = a0 + out_info['slot_width'] + out_info['gap']
```

This causes ribbons to spread out even when fanning is "disabled."

### Issue 3: OUT/IN Split Offsets Center Point

**Location:** `band_slices()` function (Cell 6, lines ~2941-2949)

The chord arc for each metric is split into OUT (outbound) and IN (inbound) halves:

```python
def band_slices(start, end):
    mid = (start + end) / 2
    out_span = (start, mid - dir_gap/2)  # Left half
    in_span = (mid + dir_gap/2, end)     # Right half
```

When `allocate_intervals` calculates the ribbon position, it uses:
- `arc_center_out = (out_range[0] + out_range[1]) / 2` → Center of LEFT half
- `arc_center_in = (in_range[0] + in_range[1]) / 2` → Center of RIGHT half

**Neither equals the ticker center!** The ticker label is at `angles[t]` which equals `chord_center = (a0 + a1) / 2`, but:
- `arc_center_out` is offset LEFT of ticker center
- `arc_center_in` is offset RIGHT of ticker center

---

## Architecture Overview

```
Ticker Arc Layout:

    ←─────────── ticker arc span ───────────→
    │                                        │
    │   OUT range    │ MID │    IN range     │
    │  (outbound)    │  ↑  │   (inbound)     │
    │                │  │  │                 │
    └────────────────┴──┴──┴─────────────────┘
                        │
                   TICKER LABEL
                   (angles[t])

Problem: Ribbons converge to center of OUT or IN range, not to MID (ticker label)
```

### Key Variables Chain

1. `angles[t]` - Ticker label position (TRUE center)
2. `spans[t] = (angle - arc_span/2, angle + arc_span/2)` - Full ticker arc
3. `chord_center = (a0 + a1) / 2` - Should equal `angles[t]`
4. `metric_spans[m]` - Distributes metrics around `chord_center`
5. `band_slices()` - Splits metric span into OUT and IN
6. `allocate_intervals()` - Places ribbons within OUT/IN ranges

---

## Solution Approach

### Phase 1: Baseline Testing (Current)

Disable all fanning to isolate the centering issue:

| Setting | Baseline Value | Purpose |
|---------|---------------|---------|
| `CHORD_ARC_FRACTION` | 1.0 | Use full arc |
| `BAND_GAP_FRAC` | 0.0 | No gap between metrics |
| `DIR_GAP_FRAC` | 0.0 | No gap between OUT/IN |
| `RIBBON_GAP_RAD` | 0.0 | No gap between ribbons |
| `RIBBON_WIDTH_SCALE_BY_FLOW` | False | Uniform width |
| `EDGE_RIBBON_SPLITS` | 1 | Single ribbon per edge |
| `RIBBON_CONVERGE_TO_POINT` | True | **NEW** - Force convergence |

### Phase 2: Convergence Fix

Added `RIBBON_CONVERGE_TO_POINT` setting that:

1. **Forces `start_pos` to the TRUE ticker center:**
   ```python
   if RIBBON_CONVERGE_TO_POINT:
       start_pos = out_range[1]  # Boundary = ticker center (for outbound)
       start_pos = in_range[0]   # Boundary = ticker center (for inbound)
   ```

2. **Prevents cursor advancement:**
   ```python
   if not RIBBON_CONVERGE_TO_POINT:
       out_info['cursor'] = a0 + out_info['slot_width'] + out_info['gap']
   # When converging, cursor stays at center - all ribbons overlap
   ```

**Why `out_range[1]` and `in_range[0]`?**

When `DIR_GAP_FRAC = 0`:
- `out_span = (start, mid)` → `out_range[1] = mid`
- `in_span = (mid, end)` → `in_range[0] = mid`
- Both equal `chord_center` = `angles[t]` (ticker label position)

### Phase 3: Re-enable Fanning (Pending)

Once baseline alignment is confirmed, re-introduce settings one at a time:

1. `EDGE_RIBBON_SPLITS = 25` - Enable fanning
2. `RIBBON_WIDTH_SCALE_BY_FLOW = True` - Scale by magnitude
3. `RIBBON_GAP_RAD = 0.002` - Small ribbon gaps
4. `DIR_GAP_FRAC = 0.02` - OUT/IN direction gap
5. `BAND_GAP_FRAC = 0.02` - Metric band gaps
6. `CHORD_ARC_FRACTION = 0.5` - Use 50% of arc

---

## Ultimate Goal

**Ribbons should be perfectly centered on ticker labels with symmetric fanning:**

```
Desired Layout:

                    TICKER
                      │
         ╲    ╲   ╲   │   ╱   ╱    ╱
          ╲    ╲  ╲   │  ╱   ╱    ╱
           ╲    ╲ ╲   │ ╱   ╱    ╱
            ╲    ╲╲   │╱   ╱    ╱
             ╲    ╲   │   ╱    ╱
              ╲    ╲  │  ╱    ╱
               ╲    ╲ │ ╱    ╱
                ╲    ╲│╱    ╱
                 ╲    │    ╱
                  ────┼────  ← All ribbons converge HERE (ticker center)
                      │
```

### Success Criteria

1. **Baseline Mode:** All ribbons converge to exact ticker label position
2. **Fanning Mode:** Ribbons fan out SYMMETRICALLY from ticker center
3. **Containment:** Fanning stays within ticker's arc (no overflow into neighbors)
4. **Visual Clarity:** Easy to identify which ticker each ribbon connects to

---

## Files Modified

- `circos_v2.ipynb` (Cell 1): Added `RIBBON_CONVERGE_TO_POINT` setting
- `circos_v2.ipynb` (Cell 6): Modified `allocate_intervals()` for convergence mode
- `circos_v2.ipynb` (Cell 6): Added `visible_band_order` filter for band_map

## Helper Scripts Created

- `fix_notebook.py` - Apply baseline settings
- `add_converge_setting.py` - Add convergence mode
- `fix_convergence_center.py` - Fix convergence to true ticker center
