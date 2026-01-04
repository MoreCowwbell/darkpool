"""
Sector summary dashboard renderer.

Generates a multi-panel grid display of sector ETF metrics matching the
screenshot style with FINRA short sale data, order flow ratios, and
accumulation scores.
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .config import DEFAULT_TABLE_STYLE, SECTOR_SUMMARY_TICKERS
    from .market_calendar import is_trading_day
except ImportError:
    from config import DEFAULT_TABLE_STYLE, SECTOR_SUMMARY_TICKERS
    from market_calendar import is_trading_day

# =============================================================================
# Constants
# =============================================================================

SECTOR_NAMES = {
    "XLE": "Energy",
    "XLF": "Financials",
    "XLK": "Technology",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLV": "Health Care",
    "XLU": "Utilities",
    "XLI": "Industrials",
    "XLC": "Communications",
    "XLB": "Basic Materials",
    "XLRE": "Real Estate",
    "SPY": "Select SPDR S&P 500",
}

# Gradient colors for accumulation score (matching plotter.py)
ACCUM_GREEN = "#39FF14"   # Bright neon green - Accumulating (score >= 70)
ACCUM_NEUTRAL = "#666666"  # Neutral (30-70)
ACCUM_PURPLE = "#BF00FF"   # Bright purple - Distribution (score <= 30)

# Bright colors for Buy/Trim signals
BRIGHT_GREEN = "#39FF14"   # Neon green
BRIGHT_RED = "#FF4444"     # Bright red

# UI colors
HEADER_BG = "transparent"  # No background for headers
BORDER_COLOR = "#3a3a3e"   # Lighter gray for borders
ACCENT_ORANGE = "#FF8C00"  # Orange for headers, labels, ticker

# Thresholds
ACCUM_HIGH_THRESHOLD = 70
ACCUM_LOW_THRESHOLD = 30

# Order flow color thresholds (using log ratio for symmetric coloring)
# log(1.65) ≈ 0.5, log(0.61) ≈ -0.5
LOG_FLOW_THRESHOLD = 0.5  # Symmetric threshold for log(ratio)

# VWBR coloring mode: "mean_threshold" or "zscore"
# - mean_threshold: compares value to 30-day mean ± k*std (computed at render time)
# - zscore: uses pre-computed finra_buy_volume_z from database
VWBR_COLOR_MODE = "zscore"  # Options: "mean_threshold", "zscore"

# VWBR mean_threshold mode settings (30-day mean +/- k*std)
VWBR_LOOKBACK_DAYS = 30
VWBR_K_BUY = 1.0   # mean + k*std = green (high buying)
VWBR_K_SELL = 1.0  # mean - k*std = red (low buying)

# VWBR zscore mode thresholds (uses finra_buy_volume_z from database)
VWBR_Z_BUY = 1.8   # z >= 1.0 = green (high buying)
VWBR_Z_SELL = -1.8  # z <= -1.0 = red (low buying)


# =============================================================================
# Data Functions
# =============================================================================


def fetch_summary_metrics(
    conn: duckdb.DuckDBPyConnection,
    dates: list[date],
    tickers: list[str],
) -> pd.DataFrame:
    """Fetch metrics for sector summary display from daily_metrics table."""
    if not tickers or not dates:
        return pd.DataFrame()

    ticker_placeholders = ", ".join(["?" for _ in tickers])
    date_placeholders = ", ".join(["?" for _ in dates])

    query = f"""
        SELECT
            date,
            symbol,
            short_ratio_denominator_value AS total_volume,
            finra_buy_volume,
            finra_buy_volume_z,
            short_buy_sell_ratio AS order_flow,
            accumulation_score_display AS accum_score
        FROM daily_metrics
        WHERE date IN ({date_placeholders}) AND symbol IN ({ticker_placeholders})
        ORDER BY symbol, date DESC
    """

    params = list(dates) + list(tickers)
    return conn.execute(query, params).df()


def fetch_vwbr_stats(
    conn: duckdb.DuckDBPyConnection,
    reference_date: date,
    tickers: list[str],
    lookback_days: int = VWBR_LOOKBACK_DAYS,
) -> dict[str, tuple[float, float]]:
    """Fetch rolling mean/std of finra_buy_volume (VWBR) per ticker.

    Uses forward-valid calculation (excludes reference_date, uses prior days only).
    This avoids look-ahead bias - matching professional quant practice.

    Returns:
        Dict mapping ticker -> (mean, std). Falls back to available data if < lookback days.
    """
    if not tickers:
        return {}

    ticker_placeholders = ", ".join(["?" for _ in tickers])

    # Forward-valid: date < reference_date (excludes current day, uses prior days only)
    query = f"""
        SELECT
            symbol,
            AVG(finra_buy_volume) AS vwbr_mean,
            STDDEV_POP(finra_buy_volume) AS vwbr_std,
            COUNT(*) AS day_count
        FROM daily_metrics
        WHERE symbol IN ({ticker_placeholders})
          AND date < ?
          AND date >= ? - INTERVAL '{lookback_days}' DAY
          AND finra_buy_volume IS NOT NULL
        GROUP BY symbol
    """

    params = list(tickers) + [reference_date, reference_date]
    df = conn.execute(query, params).df()

    result = {}
    for _, row in df.iterrows():
        symbol = row["symbol"]
        mean_val = row["vwbr_mean"]
        std_val = row["vwbr_std"]
        # Fallback: if std is 0 or NULL, set to 0 (no coloring)
        if pd.isna(std_val) or std_val <= 0:
            std_val = 0
        result[symbol] = (mean_val, std_val)

    return result


# =============================================================================
# Formatting Functions
# =============================================================================


def _format_date_short(value) -> str:
    """Format date as YYMMDD (e.g., 251229)."""
    if pd.isna(value) or value is None:
        return ""
    if isinstance(value, date):
        return value.strftime("%y%m%d")
    try:
        return pd.to_datetime(value).strftime("%y%m%d")
    except Exception:
        return str(value)


def _format_volume(value: float) -> str:
    """Format volume with M/K suffix."""
    if pd.isna(value) or value is None:
        return "NA"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:,.0f}"


def _format_order_flow(value: float) -> str:
    """Format order flow ratio to 2 decimals."""
    if pd.isna(value) or value is None:
        return "NA"
    return f"{value:.2f}"


def _format_pct_avg(buy_vol: float, total_vol: float) -> str:
    """Calculate and format % Avg (buy volume as % of total)."""
    if pd.isna(buy_vol) or pd.isna(total_vol) or total_vol == 0:
        return "NA"
    pct = (buy_vol / total_vol) * 100
    return f"{pct:.0f}%"


def _get_buy_trim(
    ratio: float,
    volume: float,
    avg_volume: float,
    accum_score: float,
) -> tuple[str, str]:
    """Return (indicator_text, css_class) based on volume-weighted flow with Acc/Dist confirmation.

    Primary Signal (both required):
    - Volume > Average volume for the ticker
    - Flow: log(ratio) >= +LOG_FLOW_THRESHOLD or <= -LOG_FLOW_THRESHOLD

    Signal Strength:
    - Soft Buy/Trim: Volume + Flow only (text color only)
    - Strong Buy/Trim: Volume + Flow + Acc/Dist confirms (text + box)
    """
    # Check for valid inputs
    if pd.isna(ratio) or ratio is None or ratio <= 0:
        return ("", "indicator-neutral")
    if pd.isna(volume) or pd.isna(avg_volume) or avg_volume <= 0:
        return ("", "indicator-neutral")

    # Primary signal: Volume must be above average
    if volume <= avg_volume:
        return ("", "indicator-neutral")

    log_ratio = math.log(ratio)

    # Check flow direction
    if log_ratio >= LOG_FLOW_THRESHOLD:
        # Bullish flow + high volume
        # Check if Acc/Dist confirms (>= 60 for strong signal)
        if not pd.isna(accum_score) and accum_score >= 60:
            return ("Buy", "indicator-buy-strong")  # Strong: text + box
        return ("Buy", "indicator-buy-soft")  # Soft: text only

    if log_ratio <= -LOG_FLOW_THRESHOLD:
        # Bearish flow + high volume
        # Check if Acc/Dist confirms (<= 40 for strong signal)
        if not pd.isna(accum_score) and accum_score <= 40:
            return ("Trim", "indicator-trim-strong")  # Strong: text + box
        return ("Trim", "indicator-trim-soft")  # Soft: text only

    return ("", "indicator-neutral")


def _get_accum_color(score) -> tuple[str, str, str]:
    """Return (background_color, text_color, border_color) for accumulation score cell.

    Uses dynamic gradient coloring:
    - score >= 70: green background + green text (max saturation)
    - score <= 30: purple background + purple text (max saturation)
    - score 30-70: black background, text follows gradient coloring
    """
    if score is None or pd.isna(score):
        return ("transparent", "", "transparent")

    # Clamp score to valid range
    score = max(0, min(100, score))

    if score >= ACCUM_HIGH_THRESHOLD:
        # Max green: green background + green text
        return ("rgba(57, 255, 20, 0.2)", ACCUM_GREEN, "#000000")
    elif score <= ACCUM_LOW_THRESHOLD:
        # Max purple: purple background + purple text
        return ("rgba(191, 0, 255, 0.2)", ACCUM_PURPLE, "#000000")
    elif score >= 50:
        # Between 50-70: black background, gradient text toward green
        intensity = (score - 50) / 20  # 0 at 50, 1 at 70
        # Interpolate between gray and green for text
        # Green RGB: (57, 255, 20), Gray RGB: (102, 102, 102)
        r = int(102 + (57 - 102) * intensity)
        g = int(102 + (255 - 102) * intensity)
        b = int(102 + (20 - 102) * intensity)
        text_color = f"rgb({r}, {g}, {b})"
        return ("transparent", text_color, "transparent")
    else:
        # Between 30-50: black background, gradient text toward purple
        intensity = (50 - score) / 20  # 0 at 50, 1 at 30
        # Interpolate between gray and purple for text
        # Purple RGB: (191, 0, 255), Gray RGB: (102, 102, 102)
        r = int(102 + (191 - 102) * intensity)
        g = int(102 + (0 - 102) * intensity)
        b = int(102 + (255 - 102) * intensity)
        text_color = f"rgb({r}, {g}, {b})"
        return ("transparent", text_color, "transparent")


def _get_order_flow_color(ratio) -> tuple[str, str, str]:
    """Return (background_color, text_color, border_color) for order flow cell.

    Uses log(ratio) for symmetric coloring:
    - Green: log(ratio) >= +LOG_FLOW_THRESHOLD (buying pressure)
    - Red: log(ratio) <= -LOG_FLOW_THRESHOLD (selling pressure)
    - Neutral: between these thresholds
    """
    if ratio is None or pd.isna(ratio) or ratio <= 0:
        return ("transparent", "", "transparent")

    log_ratio = math.log(ratio)

    if log_ratio >= LOG_FLOW_THRESHOLD:
        return ("rgba(57, 255, 20, 0.2)", BRIGHT_GREEN, "#000000")  # Buying pressure = green
    if log_ratio <= -LOG_FLOW_THRESHOLD:
        return ("rgba(255, 68, 68, 0.2)", BRIGHT_RED, "#000000")  # Selling pressure = red
    return ("transparent", "", "transparent")  # Neutral


def _get_vwbr_color(value: float, mean: float, std: float) -> tuple[str, str, str]:
    """Return (background_color, text_color, border_color) for VWBR cell.

    Uses 20-day mean threshold:
    - Green: value >= mean + VWBR_K_BUY * std (high buying)
    - Red: value <= mean - VWBR_K_SELL * std (low buying)
    - Neutral: between thresholds
    """
    if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std <= 0:
        return ("transparent", "", "transparent")

    upper_threshold = mean + VWBR_K_BUY * std
    lower_threshold = mean - VWBR_K_SELL * std

    if value >= upper_threshold:
        return ("rgba(57, 255, 20, 0.2)", BRIGHT_GREEN, "#000000")  # High = green
    if value <= lower_threshold:
        return ("rgba(255, 68, 68, 0.2)", BRIGHT_RED, "#000000")    # Low = red
    return ("transparent", "", "transparent")  # Neutral


def _get_vwbr_color_zscore(z_value: float) -> tuple[str, str, str]:
    """Return (background_color, text_color, border_color) for VWBR cell using z-score.

    Uses pre-computed finra_buy_volume_z from database:
    - Green: z >= VWBR_Z_BUY (high buying relative to rolling mean)
    - Red: z <= VWBR_Z_SELL (low buying relative to rolling mean)
    - Neutral: between thresholds
    """
    if pd.isna(z_value):
        return ("transparent", "", "transparent")

    if z_value >= VWBR_Z_BUY:
        return ("rgba(57, 255, 20, 0.2)", BRIGHT_GREEN, "#000000")  # High = green
    if z_value <= VWBR_Z_SELL:
        return ("rgba(255, 68, 68, 0.2)", BRIGHT_RED, "#000000")    # Low = red
    return ("transparent", "", "transparent")  # Neutral


# =============================================================================
# HTML Rendering Functions
# =============================================================================


def _build_sector_panel_html(
    ticker: str,
    rows_df: pd.DataFrame,
    palette: dict,
    vwbr_stats: dict[str, tuple[float, float]],
) -> str:
    """Build HTML for a single sector panel."""
    sector_name = SECTOR_NAMES.get(ticker, ticker)

    # Calculate averages first (needed for signal logic)
    avg_bs_score = rows_df["order_flow"].mean() if not rows_df.empty else 0
    avg_volume = rows_df["total_volume"].mean() if not rows_df.empty else 0

    # Get VWBR stats for mean_threshold mode (only used if VWBR_COLOR_MODE == "mean_threshold")
    # Fallback to current data if not in stats (Option A fallback)
    vwbr_mean, vwbr_std = 0, 0
    if VWBR_COLOR_MODE == "mean_threshold":
        if ticker in vwbr_stats:
            vwbr_mean, vwbr_std = vwbr_stats[ticker]
        else:
            # Fallback: use available data from current rows
            vwbr_mean = rows_df["finra_buy_volume"].mean() if not rows_df.empty else 0
            vwbr_std = rows_df["finra_buy_volume"].std() if not rows_df.empty else 0

    # Build data rows
    rows_html = ""
    for _, row in rows_df.iterrows():
        date_str = _format_date_short(row.get("date"))
        volume_str = _format_volume(row.get("total_volume"))
        order_flow_str = _format_order_flow(row.get("order_flow"))
        pct_avg_str = _format_pct_avg(row.get("finra_buy_volume"), row.get("total_volume"))

        # VWBR cell coloring based on VWBR_COLOR_MODE
        vwbr_val = row.get("finra_buy_volume")
        vwbr_str = _format_volume(vwbr_val)
        if VWBR_COLOR_MODE == "zscore":
            # Use pre-computed z-score from database
            vwbr_z = row.get("finra_buy_volume_z")
            vwbr_bg, vwbr_text, vwbr_border = _get_vwbr_color_zscore(vwbr_z)
        else:
            # Default: mean_threshold mode (30-day mean ± k*std)
            vwbr_bg, vwbr_text, vwbr_border = _get_vwbr_color(vwbr_val, vwbr_mean, vwbr_std)
        vwbr_style = f"background-color: {vwbr_bg};"
        if vwbr_text:
            vwbr_style += f" color: {vwbr_text};"
        if vwbr_border and vwbr_border != "transparent":
            vwbr_style += f" border: 1px solid {vwbr_border};"

        order_flow_val = row.get("order_flow")
        order_flow_bg, order_flow_text, order_flow_border = _get_order_flow_color(order_flow_val)
        order_flow_style = f"background-color: {order_flow_bg};"
        if order_flow_text:
            order_flow_style += f" color: {order_flow_text};"
        if order_flow_border and order_flow_border != "transparent":
            order_flow_style += f" border: 1px solid {order_flow_border};"

        volume_val = row.get("total_volume")
        accum_score = row.get("accum_score")
        indicator_text, indicator_class = _get_buy_trim(order_flow_val, volume_val, avg_volume, accum_score)

        accum_score_str = f"{accum_score:.0f}" if not pd.isna(accum_score) else "NA"
        accum_bg, accum_text, accum_border = _get_accum_color(accum_score)
        accum_style = f"background-color: {accum_bg};"
        if accum_text:
            accum_style += f" color: {accum_text}; font-weight: 700;"
        if accum_border and accum_border != "transparent":
            accum_style += f" border: 1px solid {accum_border};"

        rows_html += f"""
            <tr>
                <td class="col-date">{date_str}</td>
                <td class="col-volume">{volume_str}</td>
                <td class="col-vwbr" style="{vwbr_style}">{vwbr_str}</td>
                <td class="col-order-flow" style="{order_flow_style}">{order_flow_str}</td>
                <td class="col-pct">{pct_avg_str}</td>
                <td class="col-accum" style="{accum_style}">{accum_score_str}</td>
                <td class="col-indicator {indicator_class}">{indicator_text}</td>
            </tr>
        """

    # Format average B/S score for display (already calculated above)
    avg_bs_str = f"{avg_bs_score:.2f}" if not pd.isna(avg_bs_score) else "NA"

    # Calculate median accumulation score
    med_accum_score = rows_df["accum_score"].median() if not rows_df.empty and "accum_score" in rows_df.columns else 0
    med_accum_str = f"{med_accum_score:.0f}" if not pd.isna(med_accum_score) else "NA"

    # Determine avg B/S score box style using log ratio (same as cell coloring)
    if not pd.isna(avg_bs_score) and avg_bs_score > 0:
        log_avg = math.log(avg_bs_score)
        if log_avg >= LOG_FLOW_THRESHOLD:
            avg_bs_border = BRIGHT_GREEN
            avg_bs_text = BRIGHT_GREEN
        elif log_avg <= -LOG_FLOW_THRESHOLD:
            avg_bs_border = BRIGHT_RED
            avg_bs_text = BRIGHT_RED
        else:
            avg_bs_border = palette.get("text_muted", "#8b8b8b")
            avg_bs_text = palette.get("text_muted", "#8b8b8b")
    else:
        avg_bs_border = palette.get("text_muted", "#8b8b8b")
        avg_bs_text = palette.get("text_muted", "#8b8b8b")

    # Determine median Accum score box style using dynamic gradient (same as cell coloring)
    # Border color matches text color for consistent appearance
    _, med_accum_text, _ = _get_accum_color(med_accum_score)
    if not med_accum_text:
        med_accum_text = palette.get("text_muted", "#8b8b8b")
    med_accum_border = med_accum_text  # Border matches text color

    panel_html = f"""
        <div class="sector-panel">
            <div class="panel-header">
                <span class="ticker">{ticker}</span>
                <span class="sector-name">{sector_name}</span>
            </div>
            <table class="panel-table">
                <thead>
                    <tr>
                        <th class="col-date">Date</th>
                        <th class="col-volume">Vol</th>
                        <th class="col-vwbr">VWBR</th>
                        <th class="col-order-flow">Flow</th>
                        <th class="col-pct">%</th>
                        <th class="col-accum">Acc/Dist</th>
                        <th class="col-indicator">Signal</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <div class="panel-footer">
                <div class="footer-labels">
                    <span class="avg-label">Avg B/S</span>
                    <span class="avg-label">Med Acc</span>
                </div>
                <div class="footer-scores">
                    <span class="avg-score" style="background-color: #000000; border: 2px solid {avg_bs_border}; color: {avg_bs_text};">{avg_bs_str}</span>
                    <span class="avg-score" style="background-color: #000000; border: 2px solid {med_accum_border}; color: {med_accum_text};">{med_accum_str}</span>
                </div>
            </div>
        </div>
    """

    return panel_html


def _build_summary_grid_html(panels_html: list[str]) -> str:
    """Build CSS grid layout for all panels."""
    panels_combined = "\n".join(panels_html)
    return f"""
        <div class="summary-grid">
            {panels_combined}
        </div>
    """


def build_full_page_html(
    title: str,
    grid_html: str,
    palette: dict,
) -> str:
    """Build complete HTML document with embedded CSS."""

    # Dynamic VWBR legend text based on coloring mode
    if VWBR_COLOR_MODE == "zscore":
        vwbr_legend = f"VWBR = FINRA Buy Volume. Colored: z-score &ge; {VWBR_Z_BUY} / &le; {VWBR_Z_SELL} (20-day rolling)"
    else:
        vwbr_legend = f"VWBR = FINRA Buy Volume. Colored: mean &pm; {VWBR_K_BUY}&sigma; ({VWBR_LOOKBACK_DAYS}-day)"

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: {palette.get('background', '#0f0f10')};
            color: {palette.get('text', '#e6e6e6')};
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 14px;
            line-height: 1.4;
            padding: 10px 10px;
        }}

        .page-container {{
            max-width: 2400px;
            margin: 0 auto;
            padding: 40px 60px;
            background-color: {palette.get('background', '#0f0f10')};
        }}

        .page-header {{
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid {palette.get('border', '#26262a')};
        }}

        .page-title {{
            font-size: 28px;
            font-weight: 600;
            color: {palette.get('white', '#ffffff')};
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
        }}

        .sector-panel {{
            background-color: {palette.get('row_bg', '#0f1012')};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            padding: 12px;
            min-width: 0;
        }}

        .panel-header {{
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid {BORDER_COLOR};
        }}

        .ticker {{
            font-size: 18px;
            font-weight: 700;
            color: {ACCENT_ORANGE};
        }}

        .sector-name {{
            font-size: 12px;
            color: {ACCENT_ORANGE};
        }}

        .panel-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            table-layout: fixed;
        }}

        .panel-table th,
        .panel-table td {{
            width: 14.28%;  /* 100/7 columns */
        }}

        .panel-table thead {{
            background-color: {HEADER_BG};
        }}

        .panel-table th {{
            padding: 6px 4px;
            text-align: left;
            font-weight: 600;
            font-size: 11px;
            color: {ACCENT_ORANGE};
            border-bottom: 1px solid {BORDER_COLOR};
            white-space: nowrap;
            border-radius: 0;
        }}

        .panel-table td {{
            padding: 5px 4px;
            border-bottom: 1px solid {BORDER_COLOR};
            white-space: nowrap;
        }}

        .col-date {{
            color: {palette.get('text', '#e6e6e6')};
            font-weight: 500;
        }}

        .col-volume {{
            text-align: right;
            font-family: "Consolas", "Courier New", monospace;
            color: {palette.get('text', '#e6e6e6')};
        }}

        .col-vwbr {{
            text-align: right;
            font-family: "Consolas", "Courier New", monospace;
            font-weight: 600;
            border-radius: 6px;
        }}

        .col-order-flow {{
            text-align: right;
            font-family: "Consolas", "Courier New", monospace;
            font-weight: 600;
            color: {palette.get('white', '#ffffff')};
            border-radius: 6px;
        }}

        .col-pct {{
            text-align: right;
            font-family: "Consolas", "Courier New", monospace;
            color: {palette.get('text', '#e6e6e6')};
        }}

        .col-indicator {{
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            padding: 2px 6px;
            border-radius: 6px;
        }}

        .indicator-buy-soft {{
            color: {BRIGHT_GREEN};
        }}

        .indicator-buy-strong {{
            color: {BRIGHT_GREEN};
            background-color: rgba(57, 255, 20, 0.2);
            border: 1px solid #000000;
            border-radius: 6px;
        }}

        .indicator-trim-soft {{
            color: {BRIGHT_RED};
        }}

        .indicator-trim-strong {{
            color: {BRIGHT_RED};
            background-color: rgba(255, 68, 68, 0.2);
            border: 1px solid #000000;
            border-radius: 6px;
        }}

        .indicator-neutral {{
            color: {palette.get('text_muted', '#8b8b8b')};
        }}

        .col-accum {{
            text-align: center;
            font-family: "Consolas", "Courier New", monospace;
            font-weight: 600;
            border-radius: 6px;
        }}

        .col-order-flow {{
            border-radius: 6px;
        }}

        .panel-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid {BORDER_COLOR};
        }}

        .footer-labels {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .footer-scores {{
            display: flex;
            gap: 4px;
        }}

        .avg-label {{
            font-size: 10px;
            color: {ACCENT_ORANGE};
        }}

        .avg-score {{
            font-size: 12px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 6px;
            width: 40px;
            text-align: center;
            display: inline-block;
        }}

        .footer-row {{
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: flex-start;
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid {palette.get('border', '#26262a')};
        }}

        .footer-definitions {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 2px;
            font-size: 10px;
            color: {palette.get('text_muted', '#8b8b8b')};
        }}

        .footer-right {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 8px;
        }}

        .legend {{
            display: flex;
            justify-content: flex-end;
            gap: 20px;
            font-size: 12px;
        }}

        .data-source {{
            font-size: 10px;
            color: {ACCENT_ORANGE};
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 6px;
        }}

        /* Header column alignment */
        th.col-volume,
        th.col-vwbr,
        th.col-order-flow,
        th.col-pct,
        th.col-accum {{
            text-align: right;
        }}

        th.col-indicator {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="page-container">
        <div class="page-header">
            <span class="page-title">{title}</span>
        </div>

        {grid_html}

        <div class="footer-row">
            <div class="footer-definitions">
                <span>{vwbr_legend}</span>
                <span>Order Flow = Buy/Sell Ratio. % = Buy Volume / Total Volume.</span>
                <span>Flow coloring: log(ratio) thresholds &mdash; green &ge; +0.5, red &le; -0.5 (symmetric around 1.0)</span>
                <span>Signal strength: Soft = Vol &gt; Avg + Flow | Strong = Vol &gt; Avg + Flow + Acc/Dist confirms</span>
            </div>
            <div class="footer-right">
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {BRIGHT_GREEN};"></div>
                        <span>Buy (Vol &gt; Avg + Flow &ge; 1.65)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {BRIGHT_RED};"></div>
                        <span>Trim (Vol &gt; Avg + Flow &le; 0.61)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {ACCUM_GREEN};"></div>
                        <span>Accumulation (&ge; 70)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {ACCUM_PURPLE};"></div>
                        <span>Distribution (&le; 30)</span>
                    </div>
                </div>
                <div class="data-source">
                    Data source: FINRA Reg SHO Daily Short Sale. FINRA Weekly OTC Transparency Report.
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    """

    return html


# =============================================================================
# Output Functions
# =============================================================================


def _resolve_unique_output_paths(output_dir: Path, base_name: str) -> tuple[Path, Path]:
    """Find unique file paths to avoid overwriting existing files."""
    html_path = output_dir / f"{base_name}.html"
    png_path = output_dir / f"{base_name}.png"
    if not html_path.exists() and not png_path.exists():
        return html_path, png_path

    counter = 1
    while True:
        html_path = output_dir / f"{base_name}_{counter}.html"
        png_path = output_dir / f"{base_name}_{counter}.png"
        if not html_path.exists() and not png_path.exists():
            return html_path, png_path
        counter += 1


def render_html_to_png(html_path: Path, png_path: Path, selector: str = ".page-container") -> None:
    """Render HTML to PNG using Playwright or imgkit."""
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 2400, "height": 1400}, device_scale_factor=2)
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_load_state("networkidle")
            element = page.locator(selector)
            element.screenshot(path=str(png_path))
            browser.close()
            logger.info("Rendered PNG using Playwright: %s", png_path)
            return
    except ImportError:
        logger.warning("Playwright not installed.")
    except Exception as exc:
        logger.warning("Playwright rendering failed: %s", exc)

    try:
        import imgkit

        options = {"format": "png", "width": 2400, "quality": 100, "enable-local-file-access": None}
        imgkit.from_file(str(html_path), str(png_path), options=options)
        logger.info("Rendered PNG using imgkit: %s", png_path)
        return
    except ImportError:
        logger.warning("imgkit not installed.")
    except Exception as exc:
        logger.warning("imgkit rendering failed: %s", exc)

    logger.error("Could not render PNG. Install one of: playwright, imgkit")


def render_sector_summary(
    db_path: Path,
    output_dir: Path,
    dates: list[date],
    tickers: Optional[list[str]] = None,
    title: str = "Sector Summary Dashboard",
    max_dates: int = 10,
) -> tuple[Path, Path]:
    """
    Main entry point - render sector summary dashboard as HTML and PNG.

    Args:
        db_path: Path to DuckDB database
        output_dir: Directory for output files
        dates: List of dates to include
        tickers: List of ticker symbols (defaults to SECTOR_SUMMARY_TICKERS)
        title: Page title
        max_dates: Maximum number of dates to show per ticker (default 10)

    Returns:
        Tuple of (html_path, png_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = SECTOR_SUMMARY_TICKERS

    # Get palette from config
    palette = DEFAULT_TABLE_STYLE.get("palette", {})

    # Build file name
    dates = [d for d in dates if is_trading_day(d)]
    if not dates:
        raise ValueError("No trading dates provided for sector summary rendering.")
    sorted_dates = sorted(dates, reverse=True)
    if len(sorted_dates) == 1:
        date_label = sorted_dates[0].strftime("%Y-%m-%d")
    else:
        date_label = f"{sorted_dates[-1].strftime('%Y-%m-%d')}_to_{sorted_dates[0].strftime('%Y-%m-%d')}"

    # Fetch data
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        df = fetch_summary_metrics(conn, dates, tickers)
        if df.empty:
            logger.warning("No data found for dates %s", [d.strftime("%Y-%m-%d") for d in dates])

        # Convert date column and filter out non-trading days (weekends and holidays)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df[df["date"].apply(is_trading_day)]

        # Fetch VWBR stats only if using mean_threshold mode
        vwbr_stats = {}
        if VWBR_COLOR_MODE == "mean_threshold":
            reference_date = sorted_dates[0]  # Most recent date
            vwbr_stats = fetch_vwbr_stats(conn, reference_date, tickers)

        # Build panels for each ticker
        panels_html = []
        for ticker in tickers:
            ticker_df = df[df["symbol"] == ticker].copy()
            # Limit to max_dates most recent dates per ticker
            ticker_df = ticker_df.head(max_dates)
            panel_html = _build_sector_panel_html(ticker, ticker_df, palette, vwbr_stats)
            panels_html.append(panel_html)

        # Build grid and full page
        grid_html = _build_summary_grid_html(panels_html)
        full_html = build_full_page_html(title, grid_html, palette)

        # Write output files
        base_name = f"sector_summary_{date_label}"
        html_path, png_path = _resolve_unique_output_paths(output_dir, base_name)

        html_path.write_text(full_html, encoding="utf-8")
        logger.info("Saved HTML: %s", html_path)

        render_html_to_png(html_path, png_path)

        return html_path, png_path

    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Render sector summary dashboard as HTML/PNG")
    parser.add_argument(
        "--dates",
        type=str,
        required=True,
        help="Comma-separated dates in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers (default: SECTOR_SUMMARY_TICKERS)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sector Summary Dashboard",
        help="Page title",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=10,
        help="Maximum number of dates to show per ticker (default: 10)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from .config import load_config
    except ImportError:
        from config import load_config

    config = load_config()
    dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",")]
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    output_dir = Path(args.output_dir) if args.output_dir else config.table_dir

    render_sector_summary(
        db_path=config.db_path,
        output_dir=output_dir,
        dates=dates,
        tickers=tickers,
        title=args.title,
        max_dates=args.max_dates,
    )


if __name__ == "__main__":
    main()
