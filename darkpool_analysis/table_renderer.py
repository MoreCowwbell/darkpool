"""
Dark-theme table renderer for darkpool analysis.

Generates PNG and HTML table outputs from DuckDB darkpool_daily_summary data.
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

# =============================================================================
# Dark Theme Color Palette
# =============================================================================
COLORS = {
    "background": "#0f0f10",
    "header_bg": "#1b1b1d",
    "row_bg": "#0f0f10",
    "row_alt_bg": "#141416",
    "text": "#e6e6e6",
    "text_muted": "#8b8b8b",
    "border": "#2a2a2d",
    "green": "#00ff88",
    "green_bg": "#0a3d2a",
    "red": "#ff6b6b",
    "red_bg": "#3d1a1a",
    "yellow": "#ffd700",
    "cyan": "#00d4ff",
    "white": "#ffffff",
}

# =============================================================================
# Data Fetching
# =============================================================================


def fetch_summary_df(
    conn: duckdb.DuckDBPyConnection,
    dates: list[date],
    tickers: list[str],
) -> pd.DataFrame:
    """
    Fetch darkpool_daily_summary data for multiple dates and tickers.

    Args:
        conn: DuckDB connection (read-only preferred)
        dates: List of dates to fetch data for
        tickers: List of ticker symbols to include

    Returns:
        DataFrame with columns: symbol, estimated_bought, estimated_sold,
        buy_ratio, total_off_exchange_volume, finra_period_type
    """
    if not tickers or not dates:
        return pd.DataFrame()

    # Build parameterized query with placeholders
    ticker_placeholders = ", ".join(["?" for _ in tickers])
    date_placeholders = ", ".join(["?" for _ in dates])
    query = f"""
        SELECT
            date,
            symbol,
            estimated_bought,
            estimated_sold,
            buy_ratio,
            sell_ratio,
            total_off_exchange_volume,
            finra_period_type,
            finra_week_used
        FROM darkpool_daily_summary
        WHERE date IN ({date_placeholders}) AND symbol IN ({ticker_placeholders})
        ORDER BY date DESC, symbol
    """

    params = list(dates) + list(tickers)
    df = conn.execute(query, params).df()

    return df


# =============================================================================
# Formatting Utilities
# =============================================================================


def _format_volume(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if pd.isna(value) or value is None:
        return "—"

    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{value / 1_000:.0f}K"
    else:
        return f"{value:,.0f}"


def _format_ratio(value: float) -> str:
    """Format buy ratio with 2 decimal places."""
    if pd.isna(value) or value is None:
        return "—"
    return f"{value:.2f}"


def _format_pct(value: float) -> str:
    """Format percentage values."""
    if pd.isna(value) or value is None:
        return "—"
    return f"{value:.0f}%"


def _get_signal(buy_ratio: float) -> str:
    """Determine Accumulation/Distribution signal based on buy_ratio."""
    if pd.isna(buy_ratio) or buy_ratio is None:
        return ""
    if buy_ratio > 1.25:
        return "Accumulation"
    elif buy_ratio < 0.80:
        return "Distribution"
    return ""


def _get_ratio_color(buy_ratio: float) -> str:
    """Get color for buy_ratio based on thresholds."""
    if pd.isna(buy_ratio) or buy_ratio is None:
        return COLORS["text_muted"]
    if buy_ratio > 1.25:
        return COLORS["green"]
    elif buy_ratio < 0.80:
        return COLORS["red"]
    return COLORS["text"]


def _get_sell_ratio_color(sell_ratio: float) -> str:
    """Get color for sell_ratio (inverse logic of buy_ratio)."""
    if pd.isna(sell_ratio) or sell_ratio is None:
        return COLORS["text_muted"]
    if sell_ratio > 1.25:  # High selling pressure
        return COLORS["red"]
    elif sell_ratio < 0.80:  # Low selling pressure
        return COLORS["green"]
    return COLORS["text"]


def _get_data_quality(snapshot_date: date, finra_week: date) -> tuple[str, str]:
    """
    Determine data quality based on date vs finra_week_used.

    Returns:
        Tuple of (quality_label, color)
    """
    if finra_week is None or pd.isna(finra_week):
        return ("—", COLORS["text_muted"])

    # Convert to date objects if needed
    if hasattr(snapshot_date, 'date'):
        snapshot_date = snapshot_date.date() if callable(getattr(snapshot_date, 'date')) else snapshot_date
    if hasattr(finra_week, 'date'):
        finra_week = finra_week.date() if callable(getattr(finra_week, 'date')) else finra_week

    # Calculate weeks difference
    days_diff = (snapshot_date - finra_week).days
    weeks_diff = days_diff // 7

    if weeks_diff <= 0:
        return ("ACTUAL", COLORS["green"])
    elif weeks_diff == 1:
        return ("1 week stale", COLORS["yellow"])
    else:
        return (f"{weeks_diff} weeks stale", COLORS["red"])


def format_display_df(
    df: pd.DataFrame,
    tickers: list[str],
    dates: list[date],
) -> pd.DataFrame:
    """
    Format DataFrame for display, organized by date then ticker.

    Args:
        df: Raw DataFrame from fetch_summary_df
        tickers: Ordered list of tickers (display order)
        dates: List of dates (sorted newest first)

    Returns:
        Formatted DataFrame with display strings and additional columns
    """
    display_data = []

    # Convert df dates to Python date objects for comparison
    if not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # Sort dates descending (newest first)
    sorted_dates = sorted(dates, reverse=True)

    for target_date in sorted_dates:
        for ticker in tickers:
            ticker_data = df[(df["symbol"] == ticker) & (df["date"] == target_date)]

            if ticker_data.empty:
                # Missing data row
                display_data.append({
                    "date": target_date.strftime("%Y-%m-%d"),
                    "symbol": ticker,
                    "estimated_bought": "—",
                    "estimated_sold": "—",
                    "buy_pct": "—",
                    "buy_ratio": "—",
                    "sell_ratio": "—",
                    "total_volume": "—",
                    "finra_week": "—",
                    "data_quality": "—",
                    "signal": "",
                    "_buy_ratio_raw": None,
                    "_sell_ratio_raw": None,
                    "_data_quality_color": COLORS["text_muted"],
                    "_status": "missing",
                })
            else:
                row = ticker_data.iloc[0]
                bought = row.get("estimated_bought", 0) or 0
                sold = row.get("estimated_sold", 0) or 0
                total = bought + sold
                buy_pct = (bought / total * 100) if total > 0 else None
                buy_ratio_raw = row.get("buy_ratio")
                sell_ratio_raw = row.get("sell_ratio")
                finra_week_raw = row.get("finra_week_used")

                # Format finra_week
                if finra_week_raw is not None and not pd.isna(finra_week_raw):
                    if hasattr(finra_week_raw, 'strftime'):
                        finra_week_str = finra_week_raw.strftime("%Y-%m-%d")
                    else:
                        finra_week_str = str(finra_week_raw)
                else:
                    finra_week_str = "—"

                # Calculate data quality
                quality_label, quality_color = _get_data_quality(target_date, finra_week_raw)

                display_data.append({
                    "date": target_date.strftime("%Y-%m-%d"),
                    "symbol": ticker,
                    "estimated_bought": _format_volume(bought),
                    "estimated_sold": _format_volume(sold),
                    "buy_pct": _format_pct(buy_pct),
                    "buy_ratio": _format_ratio(buy_ratio_raw),
                    "sell_ratio": _format_ratio(sell_ratio_raw),
                    "total_volume": _format_volume(row.get("total_off_exchange_volume", 0)),
                    "finra_week": finra_week_str,
                    "data_quality": quality_label,
                    "signal": _get_signal(buy_ratio_raw),
                    "_buy_ratio_raw": buy_ratio_raw,
                    "_sell_ratio_raw": sell_ratio_raw,
                    "_data_quality_color": quality_color,
                    "_status": "ok",
                })

    return pd.DataFrame(display_data)


# =============================================================================
# HTML Generation
# =============================================================================


def build_styled_html(
    df: pd.DataFrame,
    title: str = "Dark Pool Volume",
    subtitle: str = "",
) -> str:
    """
    Build styled HTML table with dark theme.

    Args:
        df: Formatted DataFrame from format_display_df
        title: Table title
        subtitle: Optional subtitle (e.g., date range)

    Returns:
        Complete HTML string
    """
    # Calculate summary stats
    total_volume = 0
    total_bought = 0
    total_sold = 0
    ratio_values = []

    for _, row in df.iterrows():
        if row["_status"] == "ok":
            # Parse back the raw values for summary
            ratio = row["_buy_ratio_raw"]
            if ratio is not None and not pd.isna(ratio):
                ratio_values.append(ratio)

    # Build table rows
    rows_html = ""
    for idx, row in df.iterrows():
        buy_ratio_raw = row["_buy_ratio_raw"]
        sell_ratio_raw = row["_sell_ratio_raw"]
        buy_ratio_color = _get_ratio_color(buy_ratio_raw)
        sell_ratio_color = _get_sell_ratio_color(sell_ratio_raw)
        data_quality_color = row.get("_data_quality_color", COLORS["text_muted"])
        signal = row["signal"]

        # Signal styling
        if signal == "Accumulation":
            signal_html = f'<span class="signal signal-accum">Accumulation</span>'
        elif signal == "Distribution":
            signal_html = f'<span class="signal signal-dist">Distribution</span>'
        else:
            signal_html = ""

        # Row background alternation
        row_bg = COLORS["row_alt_bg"] if idx % 2 == 1 else COLORS["row_bg"]

        rows_html += f"""
        <tr style="background-color: {row_bg};">
            <td class="col-date">{row['date']}</td>
            <td class="col-symbol">{row['symbol']}</td>
            <td class="col-numeric col-bought">{row['estimated_bought']}</td>
            <td class="col-numeric col-sold">{row['estimated_sold']}</td>
            <td class="col-numeric col-pct">{row['buy_pct']}</td>
            <td class="col-numeric col-ratio" style="color: {buy_ratio_color};">{row['buy_ratio']}</td>
            <td class="col-numeric col-ratio" style="color: {sell_ratio_color};">{row['sell_ratio']}</td>
            <td class="col-numeric col-volume">{row['total_volume']}</td>
            <td class="col-date">{row['finra_week']}</td>
            <td class="col-quality" style="color: {data_quality_color};">{row['data_quality']}</td>
            <td class="col-signal">{signal_html}</td>
        </tr>
        """

    # Calculate aggregate stats if we have data
    agg_ratio = sum(ratio_values) / len(ratio_values) if ratio_values else None
    agg_ratio_str = f"{agg_ratio:.2f}" if agg_ratio else "—"
    agg_ratio_color = _get_ratio_color(agg_ratio)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            padding: 24px;
        }}

        #table-container {{
            background-color: {COLORS['background']};
            border-radius: 12px;
            padding: 24px;
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid {COLORS['border']};
        }}

        .title {{
            font-size: 24px;
            font-weight: 600;
            color: {COLORS['white']};
        }}

        .subtitle {{
            font-size: 14px;
            color: {COLORS['text_muted']};
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
        }}

        thead {{
            background-color: {COLORS['header_bg']};
        }}

        th {{
            padding: 12px 16px;
            text-align: left;
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {COLORS['text_muted']};
            border-bottom: 1px solid {COLORS['border']};
        }}

        th.col-numeric {{
            text-align: right;
        }}

        td {{
            padding: 12px 16px;
            border-bottom: 1px solid {COLORS['border']};
            vertical-align: middle;
        }}

        .col-date {{
            color: {COLORS['text_muted']};
            font-size: 12px;
        }}

        .col-symbol {{
            font-weight: 600;
            color: {COLORS['cyan']};
        }}

        .col-numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .col-bought {{
            color: {COLORS['green']};
        }}

        .col-sold {{
            color: {COLORS['text']};
        }}

        .col-pct {{
            color: {COLORS['text_muted']};
        }}

        .col-ratio {{
            font-weight: 600;
        }}

        .col-volume {{
            color: {COLORS['yellow']};
        }}

        .col-quality {{
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
        }}

        .col-signal {{
            text-align: center;
            width: 70px;
        }}

        .signal {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .signal-accum {{
            background-color: {COLORS['green_bg']};
            color: {COLORS['green']};
        }}

        .signal-dist {{
            background-color: {COLORS['red_bg']};
            color: {COLORS['red']};
        }}

        .summary {{
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px solid {COLORS['border']};
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}

        .summary-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .summary-label {{
            font-size: 12px;
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .summary-value {{
            font-size: 20px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}

        .footer {{
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid {COLORS['border']};
            font-size: 11px;
            color: {COLORS['text_muted']};
            text-align: center;
        }}
    </style>
</head>
<body>
    <div id="table-container">
        <div class="header">
            <span class="title">{title}</span>
            <span class="subtitle">{subtitle}</span>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th class="col-numeric">Bought</th>
                    <th class="col-numeric">Sold</th>
                    <th class="col-numeric">% Avg</th>
                    <th class="col-numeric">Buy Ratio</th>
                    <th class="col-numeric">Sell Ratio</th>
                    <th class="col-numeric">Total Volume</th>
                    <th>FINRA Week</th>
                    <th>Data Quality</th>
                    <th>Signal</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div class="summary">
            <div class="summary-item">
                <span class="summary-label">Rows</span>
                <span class="summary-value">{len(df)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Dates</span>
                <span class="summary-value">{df['date'].nunique() if not df.empty else 0}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Avg Buy Ratio</span>
                <span class="summary-value" style="color: {agg_ratio_color};">{agg_ratio_str}</span>
            </div>
        </div>

        <div class="footer">
            FINRA does not publish trade direction. Buy/Sell values are inferred estimates.
        </div>
    </div>
</body>
</html>
    """

    return html


# =============================================================================
# PNG Rendering
# =============================================================================


def render_html_to_png(
    html_path: Path,
    png_path: Path,
    selector: str = "#table-container",
) -> None:
    """
    Render HTML to PNG using Playwright (preferred) or fallback methods.

    Args:
        html_path: Path to input HTML file
        png_path: Path to output PNG file
        selector: CSS selector for element to screenshot
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Playwright first (best quality)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1400, "height": 900},
                device_scale_factor=2,  # Retina quality
            )

            # Load HTML file
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_load_state("networkidle")

            # Screenshot the table container
            element = page.locator(selector)
            element.screenshot(path=str(png_path))

            browser.close()
            logger.info("Rendered PNG using Playwright: %s", png_path)
            return

    except ImportError:
        logger.warning(
            "Playwright not installed. Install with: pip install playwright && playwright install chromium"
        )
    except Exception as e:
        logger.warning("Playwright rendering failed: %s", e)

    # Fallback: Try imgkit/wkhtmltoimage
    try:
        import imgkit

        options = {
            "format": "png",
            "width": 1400,
            "quality": 100,
            "enable-local-file-access": None,
        }
        imgkit.from_file(str(html_path), str(png_path), options=options)
        logger.info("Rendered PNG using imgkit: %s", png_path)
        return

    except ImportError:
        logger.warning("imgkit not installed. Install with: pip install imgkit")
    except Exception as e:
        logger.warning("imgkit rendering failed: %s", e)

    # Final fallback: Use selenium
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1400,900")

        driver = webdriver.Chrome(options=options)
        driver.get(f"file://{html_path.resolve()}")

        element = driver.find_element(By.CSS_SELECTOR, selector)
        element.screenshot(str(png_path))

        driver.quit()
        logger.info("Rendered PNG using Selenium: %s", png_path)
        return

    except ImportError:
        logger.warning("selenium not installed. Install with: pip install selenium")
    except Exception as e:
        logger.warning("Selenium rendering failed: %s", e)

    logger.error(
        "Could not render PNG. Please install one of: playwright, imgkit, or selenium. "
        "Playwright is recommended: pip install playwright && playwright install chromium"
    )


# =============================================================================
# Main Rendering Function
# =============================================================================


def render_darkpool_table(
    db_path: Path,
    output_dir: Path,
    dates: list[date],
    tickers: list[str],
    title: str = "Dark Pool Volume",
) -> tuple[Path, Path]:
    """
    Main entry point: render combined darkpool table as HTML and PNG.

    Args:
        db_path: Path to DuckDB database
        output_dir: Output directory for generated files
        dates: List of dates to include (will be sorted newest first)
        tickers: List of ticker symbols (in display order)
        title: Table title

    Returns:
        Tuple of (html_path, png_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort dates for display
    sorted_dates = sorted(dates, reverse=True)
    if len(sorted_dates) == 1:
        subtitle = f"Date: {sorted_dates[0].strftime('%Y-%m-%d')}"
        file_suffix = sorted_dates[0].strftime("%Y-%m-%d")
    else:
        subtitle = f"{sorted_dates[-1].strftime('%Y-%m-%d')} to {sorted_dates[0].strftime('%Y-%m-%d')}"
        file_suffix = "combined"

    # Connect to database
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        # Fetch data for all dates
        df = fetch_summary_df(conn, dates, tickers)

        if df.empty:
            logger.warning("No data found for dates %s", [d.strftime("%Y-%m-%d") for d in dates])

        # Format for display
        display_df = format_display_df(df, tickers, dates)

        # Build HTML
        html_content = build_styled_html(
            display_df,
            title=title,
            subtitle=subtitle,
        )

        # Save HTML
        html_path = output_dir / f"darkpool_table_{file_suffix}.html"
        html_path.write_text(html_content, encoding="utf-8")
        logger.info("Saved HTML: %s", html_path)

        # Render PNG
        png_path = output_dir / f"darkpool_table_{file_suffix}.png"
        render_html_to_png(html_path, png_path)

        return html_path, png_path

    finally:
        conn.close()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for table renderer."""
    parser = argparse.ArgumentParser(
        description="Render dark pool analysis table as HTML/PNG"
    )
    parser.add_argument(
        "--dates",
        type=str,
        required=True,
        help="Comma-separated dates in YYYY-MM-DD format (e.g., 2025-12-19,2025-12-18)",
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
        help="Comma-separated list of tickers (default: from config)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load config
    try:
        from .config import load_config
    except ImportError:
        from config import load_config

    config = load_config()

    # Parse dates
    dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",")]

    # Get tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = config.tickers

    # Get output dir
    output_dir = Path(args.output_dir) if args.output_dir else config.table_dir

    # Render
    html_path, png_path = render_darkpool_table(
        db_path=config.db_path,
        output_dir=output_dir,
        dates=dates,
        tickers=tickers,
    )

    print(f"Generated: {html_path}")
    print(f"Generated: {png_path}")


if __name__ == "__main__":
    main()
