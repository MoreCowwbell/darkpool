"""
Dark-theme table renderer for daily metrics.

Generates PNG and HTML outputs from DuckDB daily_metrics data.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

COLORS = {
    "background": "#0f0f10",
    "header_bg": "#1b1b1d",
    "panel_bg": "#141416",
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


def fetch_metrics_df(
    conn: duckdb.DuckDBPyConnection,
    dates: list[date],
    tickers: list[str],
) -> pd.DataFrame:
    if not tickers or not dates:
        return pd.DataFrame()

    ticker_placeholders = ", ".join(["?" for _ in tickers])
    date_placeholders = ", ".join(["?" for _ in dates])
    query = f"""
        SELECT
            dm.date,
            dm.symbol,
            dm.log_buy_sell,
            dm.short_ratio,
            dm.short_ratio_z,
            dm.short_ratio_denominator_type,
            dm.short_ratio_denominator_value,
            dm.close,
            dm.return_1d,
            dm.return_z,
            dm.volume,
            dm.otc_off_exchange_volume,
            dm.data_quality,
            dm.pressure_context_label,
            ld.lit_buy_volume,
            ld.lit_sell_volume,
            ld.lit_buy_ratio
        FROM daily_metrics dm
        LEFT JOIN lit_direction_daily ld
            ON dm.symbol = ld.symbol AND dm.date = ld.date
        WHERE dm.date IN ({date_placeholders}) AND dm.symbol IN ({ticker_placeholders})
        ORDER BY dm.symbol, dm.date DESC
    """

    params = list(dates) + list(tickers)
    return conn.execute(query, params).df()


def _format_volume(value: float) -> str:
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


def _format_ratio(value: float) -> str:
    if pd.isna(value) or value is None:
        return "NA"
    return f"{value:.2f}"


def _format_log(value: float) -> str:
    if pd.isna(value) or value is None:
        return "NA"
    return f"{value:.3f}"


def _format_pct(value: float) -> str:
    if pd.isna(value) or value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def _format_z(value: float) -> str:
    if pd.isna(value) or value is None:
        return "NA"
    return f"{value:.2f}"


def _get_sign_color(value: float) -> str:
    if pd.isna(value) or value is None:
        return COLORS["text_muted"]
    if value > 0:
        return COLORS["green"]
    if value < 0:
        return COLORS["red"]
    return COLORS["text"]


def _get_quality_color(value: str) -> str:
    if value == "OTC_ANCHORED":
        return COLORS["green"]
    if value == "PRE_OTC":
        return COLORS["yellow"]
    if value == "MIXED":
        return COLORS["text_muted"]
    return COLORS["text_muted"]


def _get_pressure_color(label: str) -> str:
    if label == "SHORT_INTO_STRENGTH":
        return COLORS["yellow"]
    if label == "SHORT_ON_WEAKNESS":
        return COLORS["red"]
    if label == "LOW_SHORT_STRONG_UP":
        return COLORS["green"]
    if label == "LOW_SHORT_SELL_OFF":
        return COLORS["red_bg"]
    if label == "MIXED":
        return COLORS["text_muted"]
    return COLORS["text_muted"]


def format_display_df(
    df: pd.DataFrame,
    tickers: list[str],
    dates: list[date],
) -> pd.DataFrame:
    display_rows = []

    df = df.copy()
    required_cols = [
        "date",
        "symbol",
        "close",
        "return_1d",
        "return_z",
        "log_buy_sell",
        "short_ratio",
        "short_ratio_z",
        "short_ratio_denominator_type",
        "short_ratio_denominator_value",
        "volume",
        "otc_off_exchange_volume",
        "data_quality",
        "pressure_context_label",
        "lit_buy_ratio",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    df["otc_buy_volume"] = df["otc_off_exchange_volume"] * df["lit_buy_ratio"]
    df["otc_sell_volume"] = df["otc_off_exchange_volume"] - df["otc_buy_volume"]

    sorted_dates = sorted(dates, reverse=True)
    numeric_cols = [
        "close",
        "return_1d",
        "return_z",
        "log_buy_sell",
        "lit_buy_ratio",
        "otc_buy_volume",
        "otc_sell_volume",
        "volume",
        "short_ratio",
        "short_ratio_z",
        "short_ratio_denominator_value",
        "otc_off_exchange_volume",
    ]

    def _resolve_label(series: pd.Series, fallback: str = "NA") -> str:
        values = [str(v) for v in series.dropna().unique().tolist() if str(v).strip()]
        if not values:
            return fallback
        if len(values) == 1:
            return values[0]
        return "MIXED"

    for ticker in tickers:
        ticker_df = df[df["symbol"] == ticker]
        avg_values = ticker_df[numeric_cols].mean(numeric_only=True)
        avg_denom = _resolve_label(ticker_df["short_ratio_denominator_type"])
        avg_quality = _resolve_label(ticker_df["data_quality"])
        avg_pressure = _resolve_label(ticker_df["pressure_context_label"])

        display_rows.append(
            {
                "date": "AVG",
                "symbol": ticker,
                "close": _format_ratio(avg_values.get("close")),
                "return_1d": _format_pct(avg_values.get("return_1d")),
                "return_z": _format_z(avg_values.get("return_z")),
                "total_volume": _format_volume(avg_values.get("volume")),
                "log_buy_sell": _format_log(avg_values.get("log_buy_sell")),
                "buy_ratio": _format_ratio(avg_values.get("lit_buy_ratio")),
                "bought": _format_volume(avg_values.get("otc_buy_volume")),
                "sold": _format_volume(avg_values.get("otc_sell_volume")),
                "short_ratio": _format_ratio(avg_values.get("short_ratio")),
                "short_ratio_z": _format_z(avg_values.get("short_ratio_z")),
                "short_denom_type": avg_denom,
                "short_denom_value": _format_volume(avg_values.get("short_ratio_denominator_value")),
                "otc_volume": _format_volume(avg_values.get("otc_off_exchange_volume")),
                "data_quality": avg_quality,
                "pressure_label": avg_pressure,
                "_log_raw": avg_values.get("log_buy_sell"),
                "_short_z_raw": avg_values.get("short_ratio_z"),
                "_return_z_raw": avg_values.get("return_z"),
                "_data_quality_color": _get_quality_color(avg_quality),
                "_pressure_color": _get_pressure_color(avg_pressure),
                "_row_type": "avg",
                "_total_volume_raw": avg_values.get("volume"),
                "_buy_raw": avg_values.get("otc_buy_volume"),
                "_sell_raw": avg_values.get("otc_sell_volume"),
                "_status": "avg",
            }
        )

        for target_date in sorted_dates:
            row_df = ticker_df[ticker_df["date"] == target_date]
            if row_df.empty:
                display_rows.append(
                    {
                        "date": target_date.strftime("%Y-%m-%d"),
                        "symbol": ticker,
                        "close": "NA",
                        "return_1d": "NA",
                        "return_z": "NA",
                        "total_volume": "NA",
                        "log_buy_sell": "NA",
                        "buy_ratio": "NA",
                        "bought": "NA",
                        "sold": "NA",
                        "short_ratio": "NA",
                        "short_ratio_z": "NA",
                        "short_denom_type": "NA",
                        "short_denom_value": "NA",
                        "otc_volume": "NA",
                        "data_quality": "NA",
                        "pressure_label": "NA",
                        "_log_raw": None,
                        "_short_z_raw": None,
                        "_return_z_raw": None,
                        "_data_quality_color": COLORS["text_muted"],
                        "_pressure_color": COLORS["text_muted"],
                        "_row_type": "missing",
                        "_total_volume_raw": None,
                        "_buy_raw": None,
                        "_sell_raw": None,
                        "_status": "missing",
                    }
                )
                continue

            row = row_df.iloc[0]
            buy_val = row.get("otc_buy_volume")
            sell_val = row.get("otc_sell_volume")
            display_rows.append(
                {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "symbol": ticker,
                    "close": _format_ratio(row.get("close")),
                    "return_1d": _format_pct(row.get("return_1d")),
                    "return_z": _format_z(row.get("return_z")),
                    "total_volume": _format_volume(row.get("volume")),
                    "log_buy_sell": _format_log(row.get("log_buy_sell")),
                    "buy_ratio": _format_ratio(row.get("lit_buy_ratio")),
                    "bought": _format_volume(buy_val),
                    "sold": _format_volume(sell_val),
                    "short_ratio": _format_ratio(row.get("short_ratio")),
                    "short_ratio_z": _format_z(row.get("short_ratio_z")),
                    "short_denom_type": row.get("short_ratio_denominator_type") or "NA",
                    "short_denom_value": _format_volume(row.get("short_ratio_denominator_value")),
                    "otc_volume": _format_volume(row.get("otc_off_exchange_volume")),
                    "data_quality": row.get("data_quality") or "NA",
                    "pressure_label": row.get("pressure_context_label") or "NA",
                    "_log_raw": row.get("log_buy_sell"),
                    "_short_z_raw": row.get("short_ratio_z"),
                    "_return_z_raw": row.get("return_z"),
                    "_data_quality_color": _get_quality_color(row.get("data_quality")),
                    "_pressure_color": _get_pressure_color(row.get("pressure_context_label")),
                    "_row_type": "data",
                    "_total_volume_raw": row.get("volume"),
                    "_buy_raw": buy_val,
                    "_sell_raw": sell_val,
                    "_status": "ok",
                }
            )

    return pd.DataFrame(display_rows)


def build_styled_html(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
) -> str:
    rows_html = ""
    for idx, row in df.iterrows():
        row_type = row.get("_row_type")
        row_bg = COLORS["row_alt_bg"] if idx % 2 == 1 else COLORS["row_bg"]
        row_class = ""
        if row_type == "avg":
            row_bg = COLORS["header_bg"]
            row_class = "row-avg"

        log_color = _get_sign_color(row["_log_raw"])
        short_z_color = _get_sign_color(row["_short_z_raw"])
        return_z_color = _get_sign_color(row["_return_z_raw"])
        quality_color = row["_data_quality_color"]
        pressure_color = row["_pressure_color"]

        rows_html += f"""
        <tr class="{row_class}" style="background-color: {row_bg};">
            <td class="col-date">{row['date']}</td>
            <td class="col-symbol">{row['symbol']}</td>
            <td class="col-numeric">{row['close']}</td>
            <td class="col-numeric" style="color: {return_z_color};">{row['return_1d']}</td>
            <td class="col-numeric" style="color: {return_z_color};">{row['return_z']}</td>
            <td class="col-numeric">{row['total_volume']}</td>
            <td class="col-numeric" style="color: {log_color};">{row['log_buy_sell']}</td>
            <td class="col-numeric">{row['buy_ratio']}</td>
            <td class="col-numeric">{row['bought']}</td>
            <td class="col-numeric">{row['sold']}</td>
            <td class="col-numeric">{row['short_ratio']}</td>
            <td class="col-numeric" style="color: {short_z_color};">{row['short_ratio_z']}</td>
            <td class="col-numeric">{row['short_denom_type']}</td>
            <td class="col-numeric">{row['short_denom_value']}</td>
            <td class="col-numeric">{row['otc_volume']}</td>
            <td class="col-quality" style="color: {quality_color};">{row['data_quality']}</td>
            <td class="col-quality" style="color: {pressure_color};">{row['pressure_label']}</td>
        </tr>
        """

    if not df.empty and "_row_type" in df.columns:
        data_rows = df[df["_row_type"] == "data"]
    else:
        data_rows = pd.DataFrame()
    total_volume = data_rows["_total_volume_raw"].sum(min_count=1) if not data_rows.empty else pd.NA
    total_bought = data_rows["_buy_raw"].sum(min_count=1) if not data_rows.empty else pd.NA
    total_sold = data_rows["_sell_raw"].sum(min_count=1) if not data_rows.empty else pd.NA
    avg_volume = data_rows["_total_volume_raw"].mean() if not data_rows.empty else pd.NA
    avg_bought = data_rows["_buy_raw"].mean() if not data_rows.empty else pd.NA
    avg_sold = data_rows["_sell_raw"].mean() if not data_rows.empty else pd.NA

    summary_html = f"""
        <div class="summary-panel">
            <div class="summary-title">Table Summary (All Tickers / Dates)</div>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-label">Total Vol (Polygon)</span>
                    <span class="summary-value">{_format_volume(total_volume)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Total Bought (OTC)</span>
                    <span class="summary-value">{_format_volume(total_bought)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Total Sold (OTC)</span>
                    <span class="summary-value">{_format_volume(total_sold)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Average Vol (Polygon)</span>
                    <span class="summary-value">{_format_volume(avg_volume)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Average Buy Vol (OTC)</span>
                    <span class="summary-value">{_format_volume(avg_bought)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Average Sell Vol (OTC)</span>
                    <span class="summary-value">{_format_volume(avg_sold)}</span>
                </div>
            </div>
        </div>
    """

    definitions = [
        ("Close", "Polygon daily close price."),
        ("Return", "Close vs prior close (Pct)."),
        ("Return Z", "Rolling z-score of Return."),
        ("Total Vol", "Polygon daily total volume."),
        ("Log(B/S)", "log(Lit Buy / Lit Sell) from Polygon trades."),
        ("Buy Ratio", "Lit Buy / (Lit Buy + Lit Sell) from Polygon."),
        ("Bought", "Inferred OTC buy volume = OTC Vol * Buy Ratio."),
        ("Sold", "Inferred OTC sell volume = OTC Vol * (1 - Buy Ratio)."),
        ("Short Ratio", "Short volume / Denom Vol (FINRA or Polygon)."),
        ("Short Z", "Rolling z-score of Short Ratio."),
        ("Denom", "Denominator source for Short Ratio."),
        ("Denom Vol", "Denominator volume used for Short Ratio."),
        ("OTC Vol", "FINRA weekly off-exchange volume."),
        ("Quality", "OTC_ANCHORED if week covers date; else PRE_OTC."),
        ("Pressure", "Short pressure vs price context label."),
    ]
    definition_items = "\n".join(
        [f"<li><span class=\"def-term\">{term}</span> - {desc}</li>" for term, desc in definitions]
    )

    definitions_html = f"""
        <div class="definitions-panel">
            <div class="definitions-title">Column Definitions</div>
            <ul class="definitions-list">
                {definition_items}
            </ul>
            <div class="definitions-note">
                Sources: OTC weekly = FINRA weekly summary. Short Ratio = FINRA Reg SHO daily.
                Lit direction = Polygon trades. Price/Volume = Polygon daily aggregates.
            </div>
        </div>
    """

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
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 13px;
            line-height: 1.4;
            padding: 24px;
        }}

        #table-container {{
            background-color: {COLORS['background']};
            border-radius: 12px;
            padding: 24px;
            max-width: 1800px;
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
            font-size: 22px;
            font-weight: 600;
            color: {COLORS['white']};
        }}

        .subtitle {{
            font-size: 12px;
            color: {COLORS['text_muted']};
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Consolas, "Courier New", monospace;
            font-size: 12px;
        }}

        thead {{
            background-color: {COLORS['header_bg']};
        }}

        th {{
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {COLORS['text_muted']};
            border-bottom: 1px solid {COLORS['border']};
        }}

        th.col-numeric {{
            text-align: right;
        }}

        td {{
            padding: 10px 12px;
            border-bottom: 1px solid {COLORS['border']};
            vertical-align: middle;
        }}

        .col-date {{
            color: {COLORS['text_muted']};
        }}

        .col-symbol {{
            font-weight: 600;
            color: {COLORS['cyan']};
        }}

        .col-numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .col-quality {{
            text-align: left;
            font-weight: 600;
        }}

        .row-avg td {{
            font-weight: 600;
        }}

        .summary-panel {{
            margin-top: 18px;
            padding: 16px;
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            background-color: {COLORS['panel_bg']};
        }}

        .summary-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {COLORS['text_muted']};
            margin-bottom: 12px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px 18px;
        }}

        .summary-item {{
            display: flex;
            justify-content: space-between;
            gap: 12px;
            font-size: 12px;
        }}

        .summary-label {{
            color: {COLORS['text_muted']};
        }}

        .summary-value {{
            color: {COLORS['white']};
            font-variant-numeric: tabular-nums;
        }}

        .definitions-panel {{
            margin-top: 16px;
            padding: 16px;
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            background-color: {COLORS['panel_bg']};
        }}

        .definitions-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {COLORS['text_muted']};
            margin-bottom: 10px;
        }}

        .definitions-list {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px 16px;
            font-size: 11px;
        }}

        .definitions-list li {{
            color: {COLORS['text']};
        }}

        .def-term {{
            color: {COLORS['cyan']};
            font-weight: 600;
        }}

        .definitions-note {{
            margin-top: 10px;
            font-size: 11px;
            color: {COLORS['text_muted']};
        }}

        .footer {{
            margin-top: 18px;
            padding-top: 12px;
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
                    <th class="col-numeric">Close</th>
                    <th class="col-numeric">Return</th>
                    <th class="col-numeric">Return Z</th>
                    <th class="col-numeric">Total Vol</th>
                    <th class="col-numeric">Log(B/S)</th>
                    <th class="col-numeric">Buy Ratio</th>
                    <th class="col-numeric">Bought</th>
                    <th class="col-numeric">Sold</th>
                    <th class="col-numeric">Short Ratio</th>
                    <th class="col-numeric">Short Z</th>
                    <th class="col-numeric">Denom</th>
                    <th class="col-numeric">Denom Vol</th>
                    <th class="col-numeric">OTC Vol</th>
                    <th>Quality</th>
                    <th>Pressure</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        {summary_html}
        {definitions_html}

        <div class="footer">
            FINRA does not publish trade direction. OTC volume is weekly; short sale data is daily pressure only.
        </div>
    </div>
</body>
</html>
    """

    return html


def render_html_to_png(html_path: Path, png_path: Path, selector: str = "#table-container") -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1900, "height": 1200}, device_scale_factor=2)
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

        options = {"format": "png", "width": 1900, "quality": 100, "enable-local-file-access": None}
        imgkit.from_file(str(html_path), str(png_path), options=options)
        logger.info("Rendered PNG using imgkit: %s", png_path)
        return
    except ImportError:
        logger.warning("imgkit not installed.")
    except Exception as exc:
        logger.warning("imgkit rendering failed: %s", exc)

    logger.error(
        "Could not render PNG. Install one of: playwright, imgkit, selenium."
    )


def render_daily_metrics_table(
    db_path: Path,
    output_dir: Path,
    dates: list[date],
    tickers: list[str],
    title: str = "Daily Metrics",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_dates = sorted(dates, reverse=True)
    if len(sorted_dates) == 1:
        subtitle = f"Date: {sorted_dates[0].strftime('%Y-%m-%d')}"
        file_suffix = sorted_dates[0].strftime("%Y-%m-%d")
    else:
        subtitle = f"{sorted_dates[-1].strftime('%Y-%m-%d')} to {sorted_dates[0].strftime('%Y-%m-%d')}"
        file_suffix = "combined"

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        df = fetch_metrics_df(conn, dates, tickers)
        if df.empty:
            logger.warning("No data found for dates %s", [d.strftime("%Y-%m-%d") for d in dates])

        display_df = format_display_df(df, tickers, dates)
        html_content = build_styled_html(display_df, title=title, subtitle=subtitle)

        html_path = output_dir / f"daily_metrics_{file_suffix}.html"
        html_path.write_text(html_content, encoding="utf-8")
        logger.info("Saved HTML: %s", html_path)

        png_path = output_dir / f"daily_metrics_{file_suffix}.png"
        render_html_to_png(html_path, png_path)
        return html_path, png_path
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render daily metrics table as HTML/PNG")
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
        help="Comma-separated list of tickers (default: from config)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from .config import load_config
    except ImportError:
        from config import load_config

    config = load_config()
    dates = [date.fromisoformat(d.strip()) for d in args.dates.split(",")]
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else config.tickers
    output_dir = Path(args.output_dir) if args.output_dir else config.table_dir

    render_daily_metrics_table(
        db_path=config.db_path,
        output_dir=output_dir,
        dates=dates,
        tickers=tickers,
    )


if __name__ == "__main__":
    main()
