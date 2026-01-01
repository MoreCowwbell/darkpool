"""
Sector summary dashboard renderer.

Generates a multi-panel grid display of sector ETF metrics matching the
screenshot style with FINRA short sale data, order flow ratios, and
accumulation scores.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .config import DEFAULT_TABLE_STYLE, SECTOR_SUMMARY_TICKERS
except ImportError:
    from config import DEFAULT_TABLE_STYLE, SECTOR_SUMMARY_TICKERS

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
BUY_THRESHOLD = 1.5       # ratio >= 1.5 = Buy
TRIM_THRESHOLD = 0.70     # ratio <= 0.70 = Trim
ACCUM_HIGH_THRESHOLD = 70
ACCUM_LOW_THRESHOLD = 30

# Order flow color thresholds
ORDER_FLOW_HIGH = 1.5   # Strong buy signal
ORDER_FLOW_LOW = 0.75   # Trim signal

# US stock market holidays (NYSE/NASDAQ closed)
US_MARKET_HOLIDAYS = {
    # 2024
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 15),   # MLK Day
    date(2024, 2, 19),   # Presidents' Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 27),   # Memorial Day
    date(2024, 6, 19),   # Juneteenth
    date(2024, 7, 4),    # Independence Day
    date(2024, 9, 2),    # Labor Day
    date(2024, 11, 28),  # Thanksgiving
    date(2024, 12, 25),  # Christmas
    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents' Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}

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
            short_buy_volume,
            short_buy_sell_ratio AS order_flow,
            accumulation_score_display AS accum_score
        FROM daily_metrics
        WHERE date IN ({date_placeholders}) AND symbol IN ({ticker_placeholders})
        ORDER BY symbol, date DESC
    """

    params = list(dates) + list(tickers)
    return conn.execute(query, params).df()


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


def _get_buy_trim(ratio: float) -> tuple[str, str]:
    """Return (indicator_text, css_class) based on ratio threshold."""
    if pd.isna(ratio) or ratio is None:
        return ("", "indicator-neutral")
    if ratio >= BUY_THRESHOLD:
        return ("Buy", "indicator-buy")
    if ratio <= TRIM_THRESHOLD:
        return ("Trim", "indicator-trim")
    return ("", "indicator-neutral")


def _get_accum_color(score) -> tuple[str, str, str]:
    """Return (background_color, text_color, border_color) for accumulation score cell."""
    if score is None or pd.isna(score):
        return ("transparent", "", "transparent")
    if score >= ACCUM_HIGH_THRESHOLD:
        return (f"rgba(57, 255, 20, 0.2)", ACCUM_GREEN, "#000000")  # Green bg + green text + black border
    if score <= ACCUM_LOW_THRESHOLD:
        return (f"rgba(191, 0, 255, 0.2)", ACCUM_PURPLE, "#000000")  # Purple bg + purple text + black border
    return ("transparent", ACCUM_NEUTRAL, "transparent")  # Neutral gray text


def _get_order_flow_color(ratio) -> tuple[str, str, str]:
    """Return (background_color, text_color, border_color) for order flow cell."""
    if ratio is None or pd.isna(ratio):
        return ("transparent", "", "transparent")
    if ratio >= ORDER_FLOW_HIGH:
        return (f"rgba(57, 255, 20, 0.2)", BRIGHT_GREEN, "#000000")  # Strong buy + black border
    if ratio >= BUY_THRESHOLD:
        return (f"rgba(57, 255, 20, 0.15)", BRIGHT_GREEN, "#000000")  # Buy + black border
    if ratio <= ORDER_FLOW_LOW:
        return (f"rgba(255, 68, 68, 0.2)", BRIGHT_RED, "#000000")  # Trim + black border
    return ("transparent", "", "transparent")  # Neutral


# =============================================================================
# HTML Rendering Functions
# =============================================================================


def _build_sector_panel_html(
    ticker: str,
    rows_df: pd.DataFrame,
    palette: dict,
) -> str:
    """Build HTML for a single sector panel."""
    sector_name = SECTOR_NAMES.get(ticker, ticker)

    # Build data rows
    rows_html = ""
    for _, row in rows_df.iterrows():
        date_str = _format_date_short(row.get("date"))
        volume_str = _format_volume(row.get("total_volume"))
        order_flow_str = _format_order_flow(row.get("order_flow"))
        pct_avg_str = _format_pct_avg(row.get("short_buy_volume"), row.get("total_volume"))

        order_flow_val = row.get("order_flow")
        order_flow_bg, order_flow_text, order_flow_border = _get_order_flow_color(order_flow_val)
        order_flow_style = f"background-color: {order_flow_bg};"
        if order_flow_text:
            order_flow_style += f" color: {order_flow_text};"
        if order_flow_border and order_flow_border != "transparent":
            order_flow_style += f" border: 1px solid {order_flow_border};"

        indicator_text, indicator_class = _get_buy_trim(order_flow_val)

        accum_score = row.get("accum_score")
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
                <td class="col-order-flow" style="{order_flow_style}">{order_flow_str}</td>
                <td class="col-pct">{pct_avg_str}</td>
                <td class="col-indicator {indicator_class}">{indicator_text}</td>
                <td class="col-accum" style="{accum_style}">{accum_score_str}</td>
            </tr>
        """

    # Calculate average weighted score (average of order_flow ratios)
    avg_bs_score = rows_df["order_flow"].mean() if not rows_df.empty else 0
    avg_bs_str = f"{avg_bs_score:.2f}" if not pd.isna(avg_bs_score) else "NA"

    # Calculate average accumulation score
    avg_accum_score = rows_df["accum_score"].mean() if not rows_df.empty and "accum_score" in rows_df.columns else 0
    avg_accum_str = f"{avg_accum_score:.0f}" if not pd.isna(avg_accum_score) else "NA"

    # Determine avg B/S score box style: black bg, colored border and text
    if not pd.isna(avg_bs_score):
        if avg_bs_score >= BUY_THRESHOLD:
            avg_bs_border = BRIGHT_GREEN
            avg_bs_text = BRIGHT_GREEN
        elif avg_bs_score <= TRIM_THRESHOLD:
            avg_bs_border = BRIGHT_RED
            avg_bs_text = BRIGHT_RED
        else:
            avg_bs_border = palette.get("text_muted", "#8b8b8b")
            avg_bs_text = palette.get("text_muted", "#8b8b8b")
    else:
        avg_bs_border = palette.get("text_muted", "#8b8b8b")
        avg_bs_text = palette.get("text_muted", "#8b8b8b")

    # Determine avg Accum score box style: black bg, colored border and text based on accum thresholds
    if not pd.isna(avg_accum_score):
        if avg_accum_score >= ACCUM_HIGH_THRESHOLD:
            avg_accum_border = ACCUM_GREEN
            avg_accum_text = ACCUM_GREEN
        elif avg_accum_score <= ACCUM_LOW_THRESHOLD:
            avg_accum_border = ACCUM_PURPLE
            avg_accum_text = ACCUM_PURPLE
        else:
            avg_accum_border = ACCUM_NEUTRAL
            avg_accum_text = ACCUM_NEUTRAL
    else:
        avg_accum_border = palette.get("text_muted", "#8b8b8b")
        avg_accum_text = palette.get("text_muted", "#8b8b8b")

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
                        <th class="col-order-flow">Flow</th>
                        <th class="col-pct">%</th>
                        <th class="col-indicator"></th>
                        <th class="col-accum">Acc</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <div class="panel-footer">
                <div class="footer-labels">
                    <span class="avg-label">Avg B/S</span>
                    <span class="avg-label">Avg Acc</span>
                </div>
                <div class="footer-scores">
                    <span class="avg-score" style="background-color: #000000; border: 2px solid {avg_bs_border}; color: {avg_bs_text};">{avg_bs_str}</span>
                    <span class="avg-score" style="background-color: #000000; border: 2px solid {avg_accum_border}; color: {avg_accum_text};">{avg_accum_str}</span>
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
            padding: 20px;
        }}

        .page-container {{
            max-width: 2400px;
            margin: 0 auto;
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
            width: 16.66%;
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

        .indicator-buy {{
            color: {BRIGHT_GREEN};
            background-color: rgba(57, 255, 20, 0.15);
            border: 1px solid #000000;
            border-radius: 6px;
        }}

        .indicator-trim {{
            color: {BRIGHT_RED};
            background-color: rgba(255, 68, 68, 0.15);
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
        }}

        .page-footer {{
            margin-top: 20px;
            padding-top: 12px;
            border-top: 1px solid {palette.get('border', '#26262a')};
            font-size: 11px;
            color: {ACCENT_ORANGE};
            text-align: center;
        }}

        .legend {{
            display: flex;
            justify-content: flex-end;
            gap: 20px;
            margin-top: 16px;
            font-size: 12px;
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

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: {ACCUM_GREEN};"></div>
                <span>Accumulating (&ge; 70)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {ACCUM_NEUTRAL};"></div>
                <span>Neutral (30-70)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {ACCUM_PURPLE};"></div>
                <span>Distribution (&le; 30)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {BRIGHT_GREEN};"></div>
                <span>Buy (Flow &ge; 1.25)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {BRIGHT_RED};"></div>
                <span>Trim (Flow &lt; 0.75)</span>
            </div>
        </div>

        <div class="page-footer">
            Data source: FINRA Reg SHO Daily Short Sale. Order Flow = Buy/Sell Ratio. % = Buy Volume / Total Volume.
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
            # Filter out weekends (Saturday=5, Sunday=6) and US market holidays
            df = df[df["date"].apply(lambda d: d.weekday() < 5 and d not in US_MARKET_HOLIDAYS)]

        # Build panels for each ticker
        panels_html = []
        for ticker in tickers:
            ticker_df = df[df["symbol"] == ticker].copy()
            # Limit to max_dates most recent dates per ticker
            ticker_df = ticker_df.head(max_dates)
            panel_html = _build_sector_panel_html(ticker, ticker_df, palette)
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
