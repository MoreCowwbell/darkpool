"""
Dark-theme table renderer for daily metrics.

Generates PNG and HTML outputs from DuckDB daily_metrics data.
"""
from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from .config import DEFAULT_TABLE_STYLE
except ImportError:
    from config import DEFAULT_TABLE_STYLE

GLYPH_MAP = {"dot": "●", "up": "▲", "down": "▼"}


def _resolve_table_style(raw_style: Optional[dict]) -> dict:
    style = deepcopy(DEFAULT_TABLE_STYLE)
    if raw_style:
        for key, value in raw_style.items():
            if key == "palette":
                style["palette"].update(value)
            elif key == "modes":
                style["modes"].update(value)
            elif key == "zones":
                style["zones"].update(value)
            else:
                style[key] = value

    mode = style.get("mode", "scan")
    mode_overrides = style.get("modes", {}).get(mode, {})
    for key, value in mode_overrides.items():
        style[key] = value

    return style


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _rgba(value: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(value)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _blend_hex(base: str, overlay: str, alpha: float) -> str:
    base_rgb = _hex_to_rgb(base)
    overlay_rgb = _hex_to_rgb(overlay)
    blended = tuple(
        int(round(base_c * (1 - alpha) + over_c * alpha))
        for base_c, over_c in zip(base_rgb, overlay_rgb)
    )
    return "#{:02x}{:02x}{:02x}".format(*blended)


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
            short_buy_sell_ratio,
            short_buy_sell_ratio_z,
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


def _get_sign_color(value: float, palette: dict, opacity: float) -> str:
    if pd.isna(value) or value is None:
        return _rgba(palette["text_muted"], opacity)
    if value > 0:
        return _rgba(palette["green"], opacity)
    if value < 0:
        return _rgba(palette["red"], opacity)
    return _rgba(palette["text"], opacity)


def _get_otc_status_color(value: str, palette: dict) -> str:
    if value == "Anchored":
        return palette["green"]
    if value == "Staled":
        return palette["yellow"]
    return palette["text_muted"]


def _get_pressure_color(label: str, palette: dict) -> str:
    if label == "Accumulating":
        return palette["green"]
    if label == "Distribution":
        return palette["red"]
    return palette["text_muted"]


def _get_status_glyph(label: str, glyph_map: dict, fallback: str = "dot") -> str:
    key = glyph_map.get(label, fallback)
    return GLYPH_MAP.get(key, GLYPH_MAP["dot"])


def _format_status_html(
    label: str,
    color: str,
    glyph_map: dict,
    text_color: Optional[str] = None,
) -> str:
    if label in (None, "", "NA"):
        return "NA"
    glyph = _get_status_glyph(label, glyph_map)
    text_style = f' style="color: {text_color};"' if text_color else ""
    return (
        f'<span class="status-symbol" style="color: {color};">{glyph}</span>'
        f'<span class="status-text"{text_style}>{label}</span>'
    )


def format_display_df(
    df: pd.DataFrame,
    tickers: list[str],
    dates: list[date],
    palette: dict,
) -> pd.DataFrame:
    display_rows = []

    df = df.copy()
    required_cols = [
        "date",
        "symbol",
        "return_z",
        "short_ratio",
        "short_ratio_z",
        "short_buy_sell_ratio",
        "short_buy_sell_ratio_z",
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
        "short_buy_sell_ratio",
        "short_buy_sell_ratio_z",
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
                "short_buy_ratio": _format_ratio(avg_values.get("short_buy_sell_ratio")),
                "short_z": _format_z(avg_values.get("short_buy_sell_ratio_z")),
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
                "_short_z_raw": avg_values.get("short_buy_sell_ratio_z"),
                "_lit_z_raw": avg_values.get("lit_buy_ratio_z"),
                "_otc_z_raw": avg_values.get("otc_buy_ratio_z"),
                "_otc_status_color": palette["text_muted"],
                "_pressure_color": palette["text_muted"],
                "_row_type": "avg",
                "_short_total_raw": avg_values.get("short_ratio_denominator_value"),
                "_short_buy_raw": avg_values.get("short_buy_volume"),
                "_short_sell_raw": avg_values.get("short_sell_volume"),
                "_short_ratio_raw": avg_values.get("short_buy_sell_ratio"),
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
                        "_otc_status_color": palette["text_muted"],
                        "_pressure_color": palette["text_muted"],
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
                    "short_buy_ratio": _format_ratio(row.get("short_buy_sell_ratio")),
                    "short_z": _format_z(row.get("short_buy_sell_ratio_z")),
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
                    "_short_z_raw": row.get("short_buy_sell_ratio_z"),
                    "_lit_z_raw": row.get("lit_buy_ratio_z"),
                    "_otc_z_raw": row.get("otc_buy_ratio_z"),
                    "_otc_status_color": _get_otc_status_color(otc_status, palette),
                    "_pressure_color": _get_pressure_color(pressure_label, palette),
                    "_row_type": "data",
                    "_short_total_raw": row.get("short_ratio_denominator_value"),
                    "_short_buy_raw": row.get("short_buy_volume"),
                    "_short_sell_raw": row.get("short_sell_volume"),
                    "_short_ratio_raw": row.get("short_buy_sell_ratio"),
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
    table_style: Optional[dict] = None,
) -> str:
    style = _resolve_table_style(table_style)
    palette = style["palette"]
    zones = style["zones"]
    strong_signal_cols = set(style.get("strong_signal_columns", []))
    muted_signal_cols = set(style.get("muted_signal_columns", []))
    strong_signal_opacity = style.get("signal_opacity_strong", 0.95)
    muted_signal_opacity = style.get("signal_opacity_muted", 0.6)
    neutral_text = _rgba(palette["text"], style["neutral_text_opacity"])
    group_bg_even = palette["row_bg"]
    group_bg_odd = _blend_hex(palette["row_bg"], palette["row_alt_bg"], style["group_alt_strength"])

    def _signal_color(value: float, col_name: str) -> str:
        if col_name in strong_signal_cols:
            return _get_sign_color(value, palette, strong_signal_opacity)
        if col_name in muted_signal_cols:
            return _get_sign_color(value, palette, muted_signal_opacity)
        return neutral_text

    rows_html = ""
    group_index = -1
    last_symbol = None

    for idx, row in df.iterrows():
        row_type = row.get("_row_type")
        symbol = row.get("symbol")
        is_new_group = symbol != last_symbol
        if is_new_group:
            group_index += 1
            last_symbol = symbol

        group_class = "group-even" if group_index % 2 == 0 else "group-odd"
        row_class_parts = [group_class]
        if row_type == "avg":
            row_class_parts.append("row-avg")
        if row_type == "missing":
            row_class_parts.append("row-missing")
        if is_new_group and group_index > 0:
            row_class_parts.append("group-start")

        row_class = " ".join(row_class_parts)

        return_z_color = _signal_color(row["_return_z_raw"], "return_z")
        short_z_color = _signal_color(row["_short_z_raw"], "short_z")
        lit_z_color = _signal_color(row["_lit_z_raw"], "lit_buy_z")
        otc_z_color = _signal_color(row["_otc_z_raw"], "otc_buy_z")
        otc_status_color = row["_otc_status_color"]
        pressure_color = row["_pressure_color"]
        otc_status_html = _format_status_html(
            row.get("otc_status"), otc_status_color, style.get("status_glyphs", {}).get("otc_status", {})
        )
        pressure_html = _format_status_html(
            row.get("pressure_label"),
            pressure_color,
            style.get("status_glyphs", {}).get("pressure", {}),
            text_color=pressure_color,
        )

        rows_html += f"""
        <tr class="{row_class}">
            <td class="col-date col-anchor zone-id">{row['date']}</td>
            <td class="col-symbol col-anchor zone-id">{row['symbol']}</td>
            <td class="col-numeric zone-ratio col-signal" style="color: {return_z_color};">{row['return_z']}</td>
            <td class="col-numeric zone-volume">{row['short_total_vol']}</td>
            <td class="col-numeric zone-volume">{row['short_buy_vol']}</td>
            <td class="col-numeric zone-volume">{row['short_sell_vol']}</td>
            <td class="col-numeric zone-ratio">{row['short_buy_ratio']}</td>
            <td class="col-numeric zone-ratio col-signal" style="color: {short_z_color};">{row['short_z']}</td>
            <td class="col-numeric zone-volume">{row['lit_total_vol']}</td>
            <td class="col-numeric zone-volume">{row['lit_buy_vol']}</td>
            <td class="col-numeric zone-volume">{row['lit_sell_vol']}</td>
            <td class="col-numeric zone-ratio">{row['lit_buy_ratio']}</td>
            <td class="col-numeric zone-ratio">{row['log_buy_ratio']}</td>
            <td class="col-numeric zone-ratio col-signal" style="color: {lit_z_color};">{row['lit_buy_z']}</td>
            <td class="col-numeric zone-volume">{row['otc_total_vol']}</td>
            <td class="col-numeric zone-volume">{row['otc_buy_vol']}</td>
            <td class="col-numeric zone-volume">{row['otc_sell_vol']}</td>
            <td class="col-numeric zone-ratio">{row['otc_weekly_buy_ratio']}</td>
            <td class="col-numeric zone-ratio col-signal" style="color: {otc_z_color};">{row['otc_buy_z']}</td>
            <td class="col-quality zone-status">{row['short_vol_source']}</td>
            <td class="col-quality col-status zone-status">{otc_status_html}</td>
            <td class="col-date zone-status">{row['otc_week_used']}</td>
            <td class="col-quality col-status zone-status">{pressure_html}</td>
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
                <div class="summary-subtitle">Volumes (Totals / Averages)</div>
                <div class="summary-table">
                    <div class="summary-row summary-header">
                        <div class="summary-cell summary-label"></div>
                        <div class="summary-cell summary-group">Daily Short Sale</div>
                        <div class="summary-cell summary-group">Lit Market</div>
                        <div class="summary-cell summary-group">Darkpool Weekly</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Total Vol (Sum)</div>
                        <div class="summary-cell summary-value">{_format_volume(total_short_total)}</div>
                        <div class="summary-cell summary-value">{_format_volume(total_lit_total)}</div>
                        <div class="summary-cell summary-value">{_format_volume(total_otc_total)}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Buy Vol (Sum)</div>
                        <div class="summary-cell summary-value">{_format_volume(total_short_buy)}</div>
                        <div class="summary-cell summary-value">{_format_volume(total_lit_buy)}</div>
                        <div class="summary-cell summary-value">{_format_volume(total_otc_buy)}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Sell Vol (Sum)</div>
                        <div class="summary-cell summary-value">{_format_volume(total_short_sell)}</div>
                        <div class="summary-cell summary-value">{_format_volume(total_lit_sell)}</div>
                        <div class="summary-cell summary-value">{_format_volume(total_otc_sell)}</div>
                    </div>
                    <div class="summary-row summary-divider">
                        <div class="summary-cell"></div>
                        <div class="summary-cell"></div>
                        <div class="summary-cell"></div>
                        <div class="summary-cell"></div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Total Vol (Avg)</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_short_total)}</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_lit_total)}</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_otc_total)}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Buy Vol (Avg)</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_short_buy)}</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_lit_buy)}</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_otc_buy)}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Sell Vol (Avg)</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_short_sell)}</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_lit_sell)}</div>
                        <div class="summary-cell summary-value">{_format_volume(avg_otc_sell)}</div>
                    </div>
                </div>
            </div>
            <div class="summary-section">
                <div class="summary-subtitle">Ratios / Z (Averages)</div>
                <div class="summary-table">
                    <div class="summary-row summary-header">
                        <div class="summary-cell summary-label"></div>
                        <div class="summary-cell summary-group">Daily Short Sale</div>
                        <div class="summary-cell summary-group">Lit Market</div>
                        <div class="summary-cell summary-group">Darkpool Weekly</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Buy/Sell Ratio (Avg)</div>
                        <div class="summary-cell summary-value">{_format_ratio(avg_short_ratio)}</div>
                        <div class="summary-cell summary-value">{_format_ratio(avg_lit_ratio)}</div>
                        <div class="summary-cell summary-value">{_format_ratio(avg_otc_ratio)}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Z-Score (Avg)</div>
                        <div class="summary-cell summary-value">{_format_z(avg_short_z)}</div>
                        <div class="summary-cell summary-value">{_format_z(avg_lit_z)}</div>
                        <div class="summary-cell summary-value">{_format_z(avg_otc_z)}</div>
                    </div>
                    <div class="summary-row">
                        <div class="summary-cell summary-label">Log Buy Ratio (Avg)</div>
                        <div class="summary-cell summary-value">NA</div>
                        <div class="summary-cell summary-value">{_format_log(avg_log_ratio)}</div>
                        <div class="summary-cell summary-value">NA</div>
                    </div>
                </div>
            </div>
            <div class="summary-context">
                <span class="summary-context-label">Return Z (Avg)</span>
                <span class="summary-context-value">{_format_z(avg_return_z)}</span>
            </div>
        </div>
    """

    definitions = [
        ("Return Z", "Rolling z-score of daily return (unusual strength/weakness)."),
        ("Short Total Vol", "Consolidated daily volume used for short ratio denominator."),
        ("Short Buy Vol", "FINRA short sale volume (buy-side proxy; short-exempt excluded)."),
        ("Short Sell Vol", "Short Total Vol - Short Buy Vol."),
        ("Short Sale Buy/Sell Ratio", "Short Buy Vol / Short Sell Vol (short-exempt excluded)."),
        ("Short Z", "Rolling z-score of Short Sale Buy/Sell Ratio."),
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
                Sources: OTC weekly = FINRA weekly summary. Short Sale Buy/Sell Ratio = FINRA Reg SHO daily (consolidated).
                Lit direction = Polygon trades. Price/Volume = Polygon daily aggregates.
            </div>
        </div>
    """

    legend_html = f"""
        <div class="legend-panel">
            <span class="legend-item"><span class="legend-symbol" style="color: {palette['green']};">{GLYPH_MAP['up']}</span>Accumulating</span>
            <span class="legend-item"><span class="legend-symbol" style="color: {palette['red']};">{GLYPH_MAP['down']}</span>Distribution</span>
            <span class="legend-item"><span class="legend-symbol" style="color: {palette['text_muted']};">{GLYPH_MAP['dot']}</span>Neutral</span>
            <span class="legend-item"><span class="legend-symbol" style="color: {palette['green']};">{GLYPH_MAP['dot']}</span>OTC Anchored</span>
            <span class="legend-item"><span class="legend-symbol" style="color: {palette['yellow']};">{GLYPH_MAP['dot']}</span>OTC Staled</span>
        </div>
    """

    header_text = _rgba(palette["text"], style["header_opacity"])
    gridline_color = _rgba(palette["border"], 0.35)
    border_color = _rgba(palette["border"], 0.55)
    group_sep_color = _rgba(palette["border"], style["group_separator_opacity"])
    zone_id_bg = _rgba(zones["id"], style["zone_tint_alpha"])
    zone_volume_bg = _rgba(zones["volume"], style["zone_tint_alpha"])
    zone_ratio_bg = _rgba(zones["ratio"], style["zone_tint_alpha"])
    zone_status_bg = _rgba(zones["status"], style["zone_tint_alpha"])
    row_pad_y = style["row_padding_y"]
    row_pad_x = style["row_padding_x"]
    group_start_pad = row_pad_y + style["group_separator_padding"]
    base_font_size = style["base_font_size"]
    numeric_font_size = style["base_font_size"] * style["numeric_font_scale"]
    header_font_size = style["header_font_size"]
    gridline_every = style["gridline_every"]
    avg_row_bg = _blend_hex(palette["header_bg"], palette["white"], 0.06)
    group_label_color = _rgba(palette["text"], 0.65)

    group_header_html = """
                <tr class="group-row">
                    <th class="group-spacer zone-id" colspan="3"></th>
                    <th class="group-header zone-volume" colspan="5">Daily Short Sale Volume</th>
                    <th class="group-header zone-volume" colspan="6">Lit Market</th>
                    <th class="group-header zone-volume" colspan="5">Darkpool Weekly Volume</th>
                    <th class="group-spacer zone-status" colspan="4"></th>
                </tr>
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
            background-color: {palette['background']};
            color: {palette['text']};
            font-family: {style['font_family']};
            font-size: {base_font_size}px;
            line-height: 1.45;
            padding: 24px;
        }}

        #table-container {{
            background-color: {palette['background']};
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
            border-bottom: 1px solid {border_color};
        }}

        .title {{
            font-size: 22px;
            font-weight: 600;
            color: {palette['white']};
        }}

        .subtitle {{
            font-size: 12px;
            color: {palette['text_muted']};
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: {style['font_family']};
            font-size: {base_font_size}px;
        }}

        thead {{
            background-color: {palette['header_bg']};
        }}

        th {{
            padding: {row_pad_y}px {row_pad_x}px;
            text-align: left;
            font-weight: 600;
            font-size: {header_font_size}px;
            text-transform: none;
            letter-spacing: 0.2px;
            color: {header_text};
            border-bottom: 1px solid {border_color};
            white-space: normal;
            line-height: 1.2;
        }}

        th.col-numeric {{
            text-align: right;
        }}

        .group-row th {{
            background-color: transparent;
            text-align: center;
            font-size: 11px;
            letter-spacing: 0.4px;
            color: {group_label_color};
            padding: 6px 8px 4px;
        }}

        .group-header {{
            border-bottom: 2px solid {border_color};
        }}

        .group-spacer {{
            border-bottom: none;
        }}

        td {{
            padding: {row_pad_y}px {row_pad_x}px;
            border-bottom: 1px solid transparent;
            vertical-align: middle;
            white-space: nowrap;
        }}

        .col-date {{
            color: {palette['text']};
            font-weight: 600;
        }}

        .col-symbol {{
            font-weight: 600;
            color: {palette['cyan']};
        }}

        .col-numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
            font-feature-settings: "tnum";
            font-family: {style['font_family_numeric']};
            font-size: {numeric_font_size}px;
            color: {neutral_text};
        }}

        .col-quality {{
            text-align: left;
            font-weight: 600;
        }}

        .col-anchor {{
            letter-spacing: 0.2px;
        }}

        .col-status {{
            font-weight: 600;
        }}

        .status-symbol {{
            display: inline-block;
            margin-right: 6px;
            font-size: 10px;
            transform: translateY(-1px);
        }}

        .status-text {{
            color: {palette['text']};
        }}

        .group-even {{
            background-color: {group_bg_even};
        }}

        .group-odd {{
            background-color: {group_bg_odd};
        }}

        .group-start td {{
            border-top: {style['group_separator_px']}px solid {group_sep_color};
            padding-top: {group_start_pad}px;
        }}

        .row-avg td {{
            font-weight: 600;
            background-color: {avg_row_bg};
            font-size: {base_font_size + 1}px;
        }}

        .row-missing {{
            opacity: 0.7;
        }}

        tbody tr:nth-child({gridline_every}n) td {{
            border-bottom: 1px solid {gridline_color};
        }}

        .zone-id {{
            background-color: {zone_id_bg};
        }}

        .zone-volume {{
            background-color: {zone_volume_bg};
        }}

        .zone-ratio {{
            background-color: {zone_ratio_bg};
        }}

        .zone-status {{
            background-color: {zone_status_bg};
        }}

        .summary-panel {{
            margin-top: 18px;
            padding: 16px;
            border: 1px solid {border_color};
            border-radius: 10px;
            background-color: {palette['panel_bg']};
        }}

        .summary-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {palette['text_muted']};
            margin-bottom: 12px;
        }}

        .summary-section {{
            margin-bottom: 14px;
        }}

        .summary-subtitle {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: {palette['text_muted']};
            margin-bottom: 8px;
        }}

        .summary-table {{
            display: grid;
            grid-template-columns: 1.3fr 1fr 1fr 1fr;
            border: 1px solid {border_color};
            border-radius: 8px;
            overflow: hidden;
        }}

        .summary-row {{
            display: contents;
        }}

        .summary-cell {{
            padding: 8px 10px;
            border-bottom: 1px solid {gridline_color};
            font-size: 13px;
        }}

        .summary-header .summary-cell {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: {palette['text_muted']};
            background-color: {palette['header_bg']};
            border-bottom: 1px solid {border_color};
        }}

        .summary-group {{
            text-align: center;
            font-weight: 600;
        }}

        .summary-label {{
            color: {palette['text_muted']};
        }}

        .summary-value {{
            color: {palette['white']};
            font-variant-numeric: tabular-nums;
            text-align: right;
        }}

        .summary-divider {{
            height: 1px;
        }}

        .summary-divider .summary-cell {{
            padding: 0;
            border-bottom: 1px solid {border_color};
            background-color: {palette['header_bg']};
        }}

        .summary-context {{
            margin-top: 10px;
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            font-size: 13px;
        }}

        .summary-context-label {{
            color: {palette['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.3px;
            font-size: 11px;
        }}

        .summary-context-value {{
            color: {palette['white']};
            font-variant-numeric: tabular-nums;
        }}

        .definitions-panel {{
            margin-top: 16px;
            padding: 16px;
            border: 1px solid {border_color};
            border-radius: 10px;
            background-color: {palette['panel_bg']};
        }}

        .definitions-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {palette['text_muted']};
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
            color: {palette['text']};
        }}

        .def-term {{
            color: {palette['cyan']};
            font-weight: 600;
        }}

        .definitions-note {{
            margin-top: 10px;
            font-size: 11px;
            color: {palette['text_muted']};
        }}

        .legend-panel {{
            margin-top: 14px;
            display: flex;
            justify-content: flex-end;
            gap: 18px;
            font-size: 11px;
            color: {palette['text_muted']};
        }}

        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .legend-symbol {{
            font-size: 10px;
        }}

        .footer {{
            margin-top: 18px;
            padding-top: 12px;
            border-top: 1px solid {border_color};
            font-size: 11px;
            color: {palette['text_muted']};
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
                {group_header_html}
                <tr>
                    <th class="zone-id col-anchor">Date</th>
                    <th class="zone-id col-anchor">Symbol</th>
                    <th class="col-numeric zone-ratio">Return Z</th>
                    <th class="col-numeric zone-volume">Short Total Vol</th>
                    <th class="col-numeric zone-volume">Short Buy Vol</th>
                    <th class="col-numeric zone-volume">Short Sell Vol</th>
                    <th class="col-numeric zone-ratio">Short Sale Buy/Sell Ratio</th>
                    <th class="col-numeric zone-ratio">Short Z</th>
                    <th class="col-numeric zone-volume">Lit Total Vol</th>
                    <th class="col-numeric zone-volume">Lit Buy Vol</th>
                    <th class="col-numeric zone-volume">Lit Sell Vol</th>
                    <th class="col-numeric zone-ratio">Lit Buy Ratio</th>
                    <th class="col-numeric zone-ratio">Log Buy Ratio</th>
                    <th class="col-numeric zone-ratio">Lit Buy Z</th>
                    <th class="col-numeric zone-volume">OTC Total Vol</th>
                    <th class="col-numeric zone-volume">OTC Buy Vol</th>
                    <th class="col-numeric zone-volume">OTC Sell Vol</th>
                    <th class="col-numeric zone-ratio">OTC Weekly Buy Ratio</th>
                    <th class="col-numeric zone-ratio">OTC Buy Z</th>
                    <th class="zone-status">Short Vol Source</th>
                    <th class="zone-status">OTC Status</th>
                    <th class="zone-status">OTC Week Used</th>
                    <th class="zone-status">Pressure</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        {legend_html}
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
    table_style: Optional[dict] = None,
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

        style = _resolve_table_style(table_style)
        display_df = format_display_df(df, tickers, dates, palette=style["palette"])
        html_content = build_styled_html(
            display_df, title=title, subtitle=subtitle, table_style=style
        )

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
        table_style=config.table_style,
    )


if __name__ == "__main__":
    main()
