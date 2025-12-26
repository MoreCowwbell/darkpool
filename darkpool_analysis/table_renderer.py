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
            date,
            symbol,
            return_z,
            short_ratio,
            short_ratio_z,
            short_ratio_denominator_type,
            short_ratio_denominator_value,
            short_buy_volume,
            short_sell_volume,
            lit_total_volume,
            lit_buy_volume,
            lit_sell_volume,
            lit_buy_ratio,
            log_buy_sell,
            lit_buy_ratio_z,
            otc_off_exchange_volume,
            otc_buy_volume,
            otc_sell_volume,
            otc_weekly_buy_ratio,
            otc_buy_ratio_z,
            otc_status,
            otc_week_used,
            pressure_context_label
        FROM daily_metrics
        WHERE date IN ({date_placeholders}) AND symbol IN ({ticker_placeholders})
        ORDER BY symbol, date DESC
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


def _get_otc_status_color(value: str) -> str:
    if value == "Anchored":
        return COLORS["green"]
    if value == "Staled":
        return COLORS["yellow"]
    return COLORS["text_muted"]


def _get_pressure_color(label: str) -> str:
    if label == "Accumulating":
        return COLORS["green"]
    if label == "Distribution":
        return COLORS["red"]
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
        "return_z",
        "short_ratio",
        "short_ratio_z",
        "short_ratio_denominator_type",
        "short_ratio_denominator_value",
        "short_buy_volume",
        "short_sell_volume",
        "lit_total_volume",
        "lit_buy_volume",
        "lit_sell_volume",
        "lit_buy_ratio",
        "log_buy_sell",
        "lit_buy_ratio_z",
        "otc_off_exchange_volume",
        "otc_buy_volume",
        "otc_sell_volume",
        "otc_weekly_buy_ratio",
        "otc_buy_ratio_z",
        "otc_status",
        "otc_week_used",
        "pressure_context_label",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    def _format_date(value: object) -> str:
        if pd.isna(value) or value is None:
            return "None"
        if isinstance(value, date):
            return value.strftime("%Y-%m-%d")
        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        except Exception:
            return str(value)

    sorted_dates = sorted(dates, reverse=True)
    numeric_cols = [
        "return_z",
        "short_ratio",
        "short_ratio_z",
        "short_ratio_denominator_value",
        "short_buy_volume",
        "short_sell_volume",
        "lit_total_volume",
        "lit_buy_volume",
        "lit_sell_volume",
        "lit_buy_ratio",
        "log_buy_sell",
        "lit_buy_ratio_z",
        "otc_off_exchange_volume",
        "otc_buy_volume",
        "otc_sell_volume",
        "otc_weekly_buy_ratio",
        "otc_buy_ratio_z",
    ]

    for ticker in tickers:
        ticker_df = df[df["symbol"] == ticker]
        avg_values = ticker_df[numeric_cols].mean(numeric_only=True)

        display_rows.append(
            {
                "date": "AVG",
                "symbol": ticker,
                "return_z": _format_z(avg_values.get("return_z")),
                "short_total_vol": _format_volume(avg_values.get("short_ratio_denominator_value")),
                "short_buy_vol": _format_volume(avg_values.get("short_buy_volume")),
                "short_sell_vol": _format_volume(avg_values.get("short_sell_volume")),
                "short_buy_ratio": _format_ratio(avg_values.get("short_ratio")),
                "short_z": _format_z(avg_values.get("short_ratio_z")),
                "lit_total_vol": _format_volume(avg_values.get("lit_total_volume")),
                "lit_buy_vol": _format_volume(avg_values.get("lit_buy_volume")),
                "lit_sell_vol": _format_volume(avg_values.get("lit_sell_volume")),
                "lit_buy_ratio": _format_ratio(avg_values.get("lit_buy_ratio")),
                "log_buy_ratio": _format_log(avg_values.get("log_buy_sell")),
                "lit_buy_z": _format_z(avg_values.get("lit_buy_ratio_z")),
                "otc_total_vol": _format_volume(avg_values.get("otc_off_exchange_volume")),
                "otc_buy_vol": _format_volume(avg_values.get("otc_buy_volume")),
                "otc_sell_vol": _format_volume(avg_values.get("otc_sell_volume")),
                "otc_weekly_buy_ratio": _format_ratio(avg_values.get("otc_weekly_buy_ratio")),
                "otc_buy_z": _format_z(avg_values.get("otc_buy_ratio_z")),
                "short_vol_source": "NA",
                "otc_status": "NA",
                "otc_week_used": "NA",
                "pressure_label": "NA",
                "_return_z_raw": avg_values.get("return_z"),
                "_short_z_raw": avg_values.get("short_ratio_z"),
                "_lit_z_raw": avg_values.get("lit_buy_ratio_z"),
                "_otc_z_raw": avg_values.get("otc_buy_ratio_z"),
                "_otc_status_color": COLORS["text_muted"],
                "_pressure_color": COLORS["text_muted"],
                "_row_type": "avg",
                "_short_total_raw": avg_values.get("short_ratio_denominator_value"),
                "_short_buy_raw": avg_values.get("short_buy_volume"),
                "_short_sell_raw": avg_values.get("short_sell_volume"),
                "_short_ratio_raw": avg_values.get("short_ratio"),
                "_lit_total_raw": avg_values.get("lit_total_volume"),
                "_lit_buy_raw": avg_values.get("lit_buy_volume"),
                "_lit_sell_raw": avg_values.get("lit_sell_volume"),
                "_lit_ratio_raw": avg_values.get("lit_buy_ratio"),
                "_log_ratio_raw": avg_values.get("log_buy_sell"),
                "_otc_total_raw": avg_values.get("otc_off_exchange_volume"),
                "_otc_buy_raw": avg_values.get("otc_buy_volume"),
                "_otc_sell_raw": avg_values.get("otc_sell_volume"),
                "_otc_ratio_raw": avg_values.get("otc_weekly_buy_ratio"),
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
                        "return_z": "NA",
                        "short_total_vol": "NA",
                        "short_buy_vol": "NA",
                        "short_sell_vol": "NA",
                        "short_buy_ratio": "NA",
                        "short_z": "NA",
                        "lit_total_vol": "NA",
                        "lit_buy_vol": "NA",
                        "lit_sell_vol": "NA",
                        "lit_buy_ratio": "NA",
                        "log_buy_ratio": "NA",
                        "lit_buy_z": "NA",
                        "otc_total_vol": "NA",
                        "otc_buy_vol": "NA",
                        "otc_sell_vol": "NA",
                        "otc_weekly_buy_ratio": "NA",
                        "otc_buy_z": "NA",
                        "short_vol_source": "NA",
                        "otc_status": "NA",
                        "otc_week_used": "None",
                        "pressure_label": "NA",
                        "_return_z_raw": None,
                        "_short_z_raw": None,
                        "_lit_z_raw": None,
                        "_otc_z_raw": None,
                        "_otc_status_color": COLORS["text_muted"],
                        "_pressure_color": COLORS["text_muted"],
                        "_row_type": "missing",
                        "_short_total_raw": None,
                        "_short_buy_raw": None,
                        "_short_sell_raw": None,
                        "_short_ratio_raw": None,
                        "_lit_total_raw": None,
                        "_lit_buy_raw": None,
                        "_lit_sell_raw": None,
                        "_lit_ratio_raw": None,
                        "_log_ratio_raw": None,
                        "_otc_total_raw": None,
                        "_otc_buy_raw": None,
                        "_otc_sell_raw": None,
                        "_otc_ratio_raw": None,
                        "_status": "missing",
                    }
                )
                continue

            row = row_df.iloc[0]
            otc_status = row.get("otc_status") or "None"
            pressure_label = row.get("pressure_context_label") or "Neutral"
            display_rows.append(
                {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "symbol": ticker,
                    "return_z": _format_z(row.get("return_z")),
                    "short_total_vol": _format_volume(row.get("short_ratio_denominator_value")),
                    "short_buy_vol": _format_volume(row.get("short_buy_volume")),
                    "short_sell_vol": _format_volume(row.get("short_sell_volume")),
                    "short_buy_ratio": _format_ratio(row.get("short_ratio")),
                    "short_z": _format_z(row.get("short_ratio_z")),
                    "lit_total_vol": _format_volume(row.get("lit_total_volume")),
                    "lit_buy_vol": _format_volume(row.get("lit_buy_volume")),
                    "lit_sell_vol": _format_volume(row.get("lit_sell_volume")),
                    "lit_buy_ratio": _format_ratio(row.get("lit_buy_ratio")),
                    "log_buy_ratio": _format_log(row.get("log_buy_sell")),
                    "lit_buy_z": _format_z(row.get("lit_buy_ratio_z")),
                    "otc_total_vol": _format_volume(row.get("otc_off_exchange_volume")),
                    "otc_buy_vol": _format_volume(row.get("otc_buy_volume")),
                    "otc_sell_vol": _format_volume(row.get("otc_sell_volume")),
                    "otc_weekly_buy_ratio": _format_ratio(row.get("otc_weekly_buy_ratio")),
                    "otc_buy_z": _format_z(row.get("otc_buy_ratio_z")),
                    "short_vol_source": row.get("short_ratio_denominator_type") or "NA",
                    "otc_status": otc_status,
                    "otc_week_used": _format_date(row.get("otc_week_used")),
                    "pressure_label": pressure_label,
                    "_return_z_raw": row.get("return_z"),
                    "_short_z_raw": row.get("short_ratio_z"),
                    "_lit_z_raw": row.get("lit_buy_ratio_z"),
                    "_otc_z_raw": row.get("otc_buy_ratio_z"),
                    "_otc_status_color": _get_otc_status_color(otc_status),
                    "_pressure_color": _get_pressure_color(pressure_label),
                    "_row_type": "data",
                    "_short_total_raw": row.get("short_ratio_denominator_value"),
                    "_short_buy_raw": row.get("short_buy_volume"),
                    "_short_sell_raw": row.get("short_sell_volume"),
                    "_short_ratio_raw": row.get("short_ratio"),
                    "_lit_total_raw": row.get("lit_total_volume"),
                    "_lit_buy_raw": row.get("lit_buy_volume"),
                    "_lit_sell_raw": row.get("lit_sell_volume"),
                    "_lit_ratio_raw": row.get("lit_buy_ratio"),
                    "_log_ratio_raw": row.get("log_buy_sell"),
                    "_otc_total_raw": row.get("otc_off_exchange_volume"),
                    "_otc_buy_raw": row.get("otc_buy_volume"),
                    "_otc_sell_raw": row.get("otc_sell_volume"),
                    "_otc_ratio_raw": row.get("otc_weekly_buy_ratio"),
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

        return_z_color = _get_sign_color(row["_return_z_raw"])
        short_z_color = _get_sign_color(row["_short_z_raw"])
        lit_z_color = _get_sign_color(row["_lit_z_raw"])
        otc_z_color = _get_sign_color(row["_otc_z_raw"])
        otc_status_color = row["_otc_status_color"]
        pressure_color = row["_pressure_color"]

        rows_html += f"""
        <tr class="{row_class}" style="background-color: {row_bg};">
            <td class="col-date">{row['date']}</td>
            <td class="col-symbol">{row['symbol']}</td>
            <td class="col-numeric" style="color: {return_z_color};">{row['return_z']}</td>
            <td class="col-numeric">{row['short_total_vol']}</td>
            <td class="col-numeric">{row['short_buy_vol']}</td>
            <td class="col-numeric">{row['short_sell_vol']}</td>
            <td class="col-numeric">{row['short_buy_ratio']}</td>
            <td class="col-numeric" style="color: {short_z_color};">{row['short_z']}</td>
            <td class="col-numeric">{row['lit_total_vol']}</td>
            <td class="col-numeric">{row['lit_buy_vol']}</td>
            <td class="col-numeric">{row['lit_sell_vol']}</td>
            <td class="col-numeric">{row['lit_buy_ratio']}</td>
            <td class="col-numeric">{row['log_buy_ratio']}</td>
            <td class="col-numeric" style="color: {lit_z_color};">{row['lit_buy_z']}</td>
            <td class="col-numeric">{row['otc_total_vol']}</td>
            <td class="col-numeric">{row['otc_buy_vol']}</td>
            <td class="col-numeric">{row['otc_sell_vol']}</td>
            <td class="col-numeric">{row['otc_weekly_buy_ratio']}</td>
            <td class="col-numeric" style="color: {otc_z_color};">{row['otc_buy_z']}</td>
            <td class="col-quality">{row['short_vol_source']}</td>
            <td class="col-quality" style="color: {otc_status_color};">{row['otc_status']}</td>
            <td class="col-date">{row['otc_week_used']}</td>
            <td class="col-quality" style="color: {pressure_color};">{row['pressure_label']}</td>
        </tr>
        """

    if not df.empty and "_row_type" in df.columns:
        data_rows = df[df["_row_type"] == "data"]
    else:
        data_rows = pd.DataFrame()
    def _safe_sum(series: pd.Series) -> float:
        if series.empty:
            return pd.NA
        return series.sum(min_count=1)

    def _safe_mean(series: pd.Series) -> float:
        if series.empty:
            return pd.NA
        return series.mean()

    total_short_total = _safe_sum(data_rows["_short_total_raw"])
    total_short_buy = _safe_sum(data_rows["_short_buy_raw"])
    total_short_sell = _safe_sum(data_rows["_short_sell_raw"])
    total_lit_total = _safe_sum(data_rows["_lit_total_raw"])
    total_lit_buy = _safe_sum(data_rows["_lit_buy_raw"])
    total_lit_sell = _safe_sum(data_rows["_lit_sell_raw"])
    total_otc_total = _safe_sum(data_rows["_otc_total_raw"])
    total_otc_buy = _safe_sum(data_rows["_otc_buy_raw"])
    total_otc_sell = _safe_sum(data_rows["_otc_sell_raw"])

    avg_short_total = _safe_mean(data_rows["_short_total_raw"])
    avg_short_buy = _safe_mean(data_rows["_short_buy_raw"])
    avg_short_sell = _safe_mean(data_rows["_short_sell_raw"])
    avg_lit_total = _safe_mean(data_rows["_lit_total_raw"])
    avg_lit_buy = _safe_mean(data_rows["_lit_buy_raw"])
    avg_lit_sell = _safe_mean(data_rows["_lit_sell_raw"])
    avg_otc_total = _safe_mean(data_rows["_otc_total_raw"])
    avg_otc_buy = _safe_mean(data_rows["_otc_buy_raw"])
    avg_otc_sell = _safe_mean(data_rows["_otc_sell_raw"])

    avg_return_z = _safe_mean(data_rows["_return_z_raw"])
    avg_short_ratio = _safe_mean(data_rows["_short_ratio_raw"])
    avg_short_z = _safe_mean(data_rows["_short_z_raw"])
    avg_lit_ratio = _safe_mean(data_rows["_lit_ratio_raw"])
    avg_log_ratio = _safe_mean(data_rows["_log_ratio_raw"])
    avg_lit_z = _safe_mean(data_rows["_lit_z_raw"])
    avg_otc_ratio = _safe_mean(data_rows["_otc_ratio_raw"])
    avg_otc_z = _safe_mean(data_rows["_otc_z_raw"])

    summary_html = f"""
        <div class="summary-panel">
            <div class="summary-title">Summary (All Tickers / Dates)</div>
            <div class="summary-section">
                <div class="summary-subtitle">Totals (Volumes)</div>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="summary-label">Short Total Vol</span>
                        <span class="summary-value">{_format_volume(total_short_total)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Short Buy Vol</span>
                        <span class="summary-value">{_format_volume(total_short_buy)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Short Sell Vol</span>
                        <span class="summary-value">{_format_volume(total_short_sell)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Total Vol</span>
                        <span class="summary-value">{_format_volume(total_lit_total)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Buy Vol</span>
                        <span class="summary-value">{_format_volume(total_lit_buy)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Sell Vol</span>
                        <span class="summary-value">{_format_volume(total_lit_sell)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Total Vol</span>
                        <span class="summary-value">{_format_volume(total_otc_total)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Buy Vol</span>
                        <span class="summary-value">{_format_volume(total_otc_buy)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Sell Vol</span>
                        <span class="summary-value">{_format_volume(total_otc_sell)}</span>
                    </div>
                </div>
            </div>
            <div class="summary-section">
                <div class="summary-subtitle">Averages (Volumes)</div>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="summary-label">Short Total Vol</span>
                        <span class="summary-value">{_format_volume(avg_short_total)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Short Buy Vol</span>
                        <span class="summary-value">{_format_volume(avg_short_buy)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Short Sell Vol</span>
                        <span class="summary-value">{_format_volume(avg_short_sell)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Total Vol</span>
                        <span class="summary-value">{_format_volume(avg_lit_total)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Buy Vol</span>
                        <span class="summary-value">{_format_volume(avg_lit_buy)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Sell Vol</span>
                        <span class="summary-value">{_format_volume(avg_lit_sell)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Total Vol</span>
                        <span class="summary-value">{_format_volume(avg_otc_total)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Buy Vol</span>
                        <span class="summary-value">{_format_volume(avg_otc_buy)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Sell Vol</span>
                        <span class="summary-value">{_format_volume(avg_otc_sell)}</span>
                    </div>
                </div>
            </div>
            <div class="summary-section">
                <div class="summary-subtitle">Averages (Ratios / Z)</div>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="summary-label">Return Z</span>
                        <span class="summary-value">{_format_z(avg_return_z)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Short Sale Buy Ratio</span>
                        <span class="summary-value">{_format_ratio(avg_short_ratio)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Short Z</span>
                        <span class="summary-value">{_format_z(avg_short_z)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Buy Ratio</span>
                        <span class="summary-value">{_format_ratio(avg_lit_ratio)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Log Buy Ratio</span>
                        <span class="summary-value">{_format_log(avg_log_ratio)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Lit Buy Z</span>
                        <span class="summary-value">{_format_z(avg_lit_z)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Weekly Buy Ratio</span>
                        <span class="summary-value">{_format_ratio(avg_otc_ratio)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">OTC Buy Z</span>
                        <span class="summary-value">{_format_z(avg_otc_z)}</span>
                    </div>
                </div>
            </div>
        </div>
    """

    definitions = [
        ("Return Z", "Rolling z-score of daily return (unusual strength/weakness)."),
        ("Short Total Vol", "Consolidated daily volume used for short ratio denominator."),
        ("Short Buy Vol", "FINRA short sale volume (buy-side proxy; short-exempt excluded)."),
        ("Short Sell Vol", "Short Total Vol - Short Buy Vol."),
        ("Short Sale Buy Ratio", "Short Buy Vol / Short Total Vol (short-exempt excluded)."),
        ("Short Z", "Rolling z-score of Short Sale Buy Ratio."),
        ("Lit Total Vol", "Lit buy + lit sell volume (Polygon trades)."),
        ("Lit Buy Vol", "Lit buy volume from Option B inference."),
        ("Lit Sell Vol", "Lit sell volume from Option B inference."),
        ("Lit Buy Ratio", "Lit Buy Vol / (Lit Buy Vol + Lit Sell Vol)."),
        ("Log Buy Ratio", "ln(Lit Buy Vol / Lit Sell Vol)."),
        ("Lit Buy Z", "Rolling z-score of Lit Buy Ratio."),
        ("OTC Total Vol", "FINRA weekly off-exchange volume."),
        ("OTC Buy Vol", "OTC Total Vol * OTC Weekly Buy Ratio (proxy)."),
        ("OTC Sell Vol", "OTC Total Vol - OTC Buy Vol."),
        ("OTC Weekly Buy Ratio", "Weekly lit buy ratio mapped to the OTC week used."),
        ("OTC Buy Z", "Rolling z-score of OTC Weekly Buy Ratio."),
        ("Short Vol Source", "FINRA total when present, else Polygon total volume."),
        ("OTC Status", "Anchored if week covers date; Staled if week is older; None if missing."),
        ("OTC Week Used", "Week-start date used for OTC calculations."),
        ("Pressure", "Neutral/Accumulating/Distribution using Short Z + Return Z."),
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
                Sources: OTC weekly = FINRA weekly summary. Short Sale Buy Ratio = FINRA Reg SHO daily (consolidated).
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
            max-width: 2400px;
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

        .summary-section {{
            margin-bottom: 14px;
        }}

        .summary-subtitle {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: {COLORS['text_muted']};
            margin-bottom: 8px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
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
                    <th class="col-numeric">Return Z</th>
                    <th class="col-numeric">Short Total Vol</th>
                    <th class="col-numeric">Short Buy Vol</th>
                    <th class="col-numeric">Short Sell Vol</th>
                    <th class="col-numeric">Short Sale Buy Ratio</th>
                    <th class="col-numeric">Short Z</th>
                    <th class="col-numeric">Lit Total Vol</th>
                    <th class="col-numeric">Lit Buy Vol</th>
                    <th class="col-numeric">Lit Sell Vol</th>
                    <th class="col-numeric">Lit Buy Ratio</th>
                    <th class="col-numeric">Log Buy Ratio</th>
                    <th class="col-numeric">Lit Buy Z</th>
                    <th class="col-numeric">OTC Total Vol</th>
                    <th class="col-numeric">OTC Buy Vol</th>
                    <th class="col-numeric">OTC Sell Vol</th>
                    <th class="col-numeric">OTC Weekly Buy Ratio</th>
                    <th class="col-numeric">OTC Buy Z</th>
                    <th>Short Vol Source</th>
                    <th>OTC Status</th>
                    <th>OTC Week Used</th>
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
            FINRA does not publish trade direction. Short sale volume is a buy-side proxy; OTC volume is weekly.
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
