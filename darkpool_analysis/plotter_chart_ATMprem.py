"""
Standalone OHLCV price chart plotter for institutional pressure analysis.

Generates a separate PNG per ticker with OHLC bars, volume bars, and
accumulation-score buy/sell markers.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from typing import Optional

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from .config import load_config, DAILY_EXPIRATION_TICKERS
except ImportError:
    from config import load_config, DAILY_EXPIRATION_TICKERS

logger = logging.getLogger(__name__)

COLORS = {
    "background": "#0f0f10",
    "panel_bg": "#141416",
    "text": "#e6e6e6",
    "grid": "#2a2a2d",
    "white": "#ffffff",
    "green": "#00ff88",
    "red": "#ff6b6b",
    "yellow": "#ffd700",
    "blue": "#4aa3ff",
    "orange": "#ff9f43",
    "neutral": "#6b6b6b",
    # Options premium colors (TOS style)
    "call_premium": "#6478c8",  # Steel blue (like TOS alternate)
    "put_premium": "#cc0000",   # Deep red
    # ITM/OTM breakdown colors - Bullish/Bearish interpretation
    # Cyan/Teal tones = Bullish pressure, Red/Orange tones = Bearish pressure
    "otm_call": "#00BFFF",      # Cyan - OTM Calls (Bullish speculation)
    "itm_call": "#FF6B35",      # Orange - ITM Calls (Bearish hedge signal)
    "otm_put": "#FF4444",       # Red - OTM Puts (Bearish speculation)
    "itm_put": "#00CED1",       # Teal - ITM Puts (Bullish - sold protective puts)
    "hedge_warning": "#ffd700", # Yellow for ITM call hedge warning
}

GRID_ALPHA = 0.18
OHLC_LINE_WIDTH = 1.2
SIGNAL_MARKER_SIZE = 60
SIGNAL_EDGE_WIDTH = 0.6
VOLUME_ALPHA = 0.55

TIMEFRAME_RULES = {
    "daily": "D",
    "weekly": "W-FRI",
    "monthly": "M",
}


def _compute_fig_width(bar_count: int) -> float:
    if bar_count <= 0:
        return 12.0
    buckets = int(np.ceil(bar_count / 50))
    return 12.0 * max(1, buckets)


def _apply_axis_style(ax) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.grid(True, alpha=GRID_ALPHA, color=COLORS["grid"], linestyle="--")


def _resolve_timeframe(value: str) -> str:
    if not value:
        return "daily"
    normalized = value.strip().lower()
    if normalized in TIMEFRAME_RULES:
        return normalized
    logger.warning("Unknown price bar timeframe '%s', falling back to daily.", value)
    return "daily"


def fetch_price_data(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    dates: list[date],
) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()

    symbol = symbol.upper()
    min_date = min(dates)
    max_date = max(dates)
    price_query = """
        SELECT
            trade_date AS date,
            open,
            high,
            low,
            close,
            volume
        FROM polygon_daily_agg_raw
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    score_query = """
        SELECT
            date,
            accumulation_score_display,
            confidence
        FROM daily_metrics
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """
    price_df = conn.execute(price_query, [symbol, min_date, max_date]).df()
    if price_df.empty:
        return price_df

    score_df = conn.execute(score_query, [symbol, min_date, max_date]).df()
    if score_df.empty:
        price_df["accumulation_score_display"] = pd.NA
        price_df["confidence"] = pd.NA
    else:
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
        score_df["date"] = pd.to_datetime(score_df["date"]).dt.date
        price_df = price_df.merge(score_df, on="date", how="left")

    price_df["date"] = pd.to_datetime(price_df["date"])
    for col in ["open", "high", "low", "close", "volume", "accumulation_score_display", "confidence"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    return price_df


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "accumulation_score_display" not in df.columns:
        df["accumulation_score_display"] = pd.NA
    if "confidence" not in df.columns:
        df["confidence"] = pd.NA

    timeframe = _resolve_timeframe(timeframe)
    if timeframe == "daily":
        return df.sort_values("date").reset_index(drop=True)

    rule = TIMEFRAME_RULES[timeframe]
    df = df.sort_values("date").set_index("date")
    aggregated = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "accumulation_score_display": "last",
            "confidence": "last",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "close"])
    return aggregated.reset_index()


def _plot_ohlcv_bars(ax, df: pd.DataFrame) -> tuple[np.ndarray, float]:
    dates = mdates.date2num(df["date"])
    if len(dates) > 1:
        bar_width = np.median(np.diff(dates)) * 0.6
    else:
        bar_width = 0.6
    half_width = bar_width / 2

    for x, open_, high, low, close in zip(
        dates, df["open"], df["high"], df["low"], df["close"]
    ):
        if pd.isna(open_) or pd.isna(close) or pd.isna(high) or pd.isna(low):
            continue
        color = COLORS["blue"] if close >= open_ else COLORS["orange"]
        ax.vlines(x, low, high, color=color, linewidth=OHLC_LINE_WIDTH, zorder=3)
        ax.hlines(open_, x - half_width, x, color=color, linewidth=OHLC_LINE_WIDTH, zorder=3)
        ax.hlines(close, x, x + half_width, color=color, linewidth=OHLC_LINE_WIDTH, zorder=3)

    return dates, bar_width


def _plot_volume_bars(ax, df: pd.DataFrame, dates: np.ndarray, bar_width: float) -> None:
    colors = np.where(df["close"] >= df["open"], COLORS["blue"], COLORS["orange"])
    ax.bar(dates, df["volume"], width=bar_width, color=colors, alpha=VOLUME_ALPHA, zorder=2)


def fetch_options_premium_data(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    dates: list[date],
    expiration_type: str,
) -> pd.DataFrame:
    """Fetch options premium summary data for plotting, including ITM/OTM breakdown."""
    if not dates:
        return pd.DataFrame()

    symbol = symbol.upper()
    min_date = min(dates)
    max_date = max(dates)

    query = """
        SELECT
            trade_date AS date,
            total_call_premium,
            total_put_premium,
            net_premium,
            log_ratio,
            strikes_count,
            otm_call_premium,
            itm_call_premium,
            otm_put_premium,
            itm_put_premium,
            directional_score
        FROM options_premium_summary
        WHERE symbol = ? AND trade_date BETWEEN ? AND ? AND expiration_type = ?
        ORDER BY trade_date
    """
    df = conn.execute(query, [symbol, min_date, max_date, expiration_type]).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = [
            "total_call_premium", "total_put_premium", "net_premium", "log_ratio",
            "otm_call_premium", "itm_call_premium", "otm_put_premium", "itm_put_premium",
            "directional_score"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _draw_directional_gauge(ax, directional_score: float, max_range: float = 100.0) -> None:
    """
    Draw a horizontal bullish/bearish gauge in the top-left corner.

    Args:
        ax: matplotlib axis
        directional_score: The computed score (positive=bullish, negative=bearish)
        max_range: Maximum absolute value for the gauge scale
    """
    from matplotlib.patches import Rectangle

    # Clamp score to range
    clamped_score = max(-max_range, min(max_range, directional_score))
    normalized = clamped_score / max_range  # -1 to +1

    # Gauge dimensions (in axes coordinates)
    gauge_left = 0.02
    gauge_bottom = 0.82
    gauge_width = 0.22
    gauge_height = 0.10

    # Draw gauge background (gradient from red to green)
    # Left half (bearish - red)
    ax.add_patch(Rectangle(
        (gauge_left, gauge_bottom), gauge_width / 2, gauge_height,
        transform=ax.transAxes, facecolor="#FF4444", alpha=0.3,
        edgecolor="none", zorder=10
    ))
    # Right half (bullish - green)
    ax.add_patch(Rectangle(
        (gauge_left + gauge_width / 2, gauge_bottom), gauge_width / 2, gauge_height,
        transform=ax.transAxes, facecolor="#00FF88", alpha=0.3,
        edgecolor="none", zorder=10
    ))

    # Draw gauge border
    ax.add_patch(Rectangle(
        (gauge_left, gauge_bottom), gauge_width, gauge_height,
        transform=ax.transAxes, facecolor="none",
        edgecolor=COLORS.get("grid", "#2a2a2d"), linewidth=1, zorder=11
    ))

    # Draw center line
    center_x = gauge_left + gauge_width / 2
    ax.plot([center_x, center_x], [gauge_bottom, gauge_bottom + gauge_height],
            transform=ax.transAxes, color=COLORS.get("text", "#e6e6e6"),
            linewidth=1, zorder=12)

    # Draw marker/needle position
    marker_x = gauge_left + gauge_width / 2 + (normalized * gauge_width / 2)
    marker_color = "#00FF88" if directional_score >= 0 else "#FF4444"
    ax.plot(marker_x, gauge_bottom + gauge_height / 2, "o",
            transform=ax.transAxes, color=marker_color,
            markersize=8, markeredgecolor="white", markeredgewidth=1, zorder=13)

    # Draw score label
    label = f"{directional_score:+.1f}M"
    ax.text(marker_x, gauge_bottom - 0.02, label,
            transform=ax.transAxes, fontsize=8,
            ha="center", va="top", color=marker_color, fontweight="bold", zorder=13)

    # Draw gauge labels
    ax.text(gauge_left, gauge_bottom + gauge_height + 0.01, "Bearish",
            transform=ax.transAxes, fontsize=6, ha="left", va="bottom",
            color="#FF4444", zorder=13)
    ax.text(gauge_left + gauge_width, gauge_bottom + gauge_height + 0.01, "Bullish",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            color="#00FF88", zorder=13)


def _plot_options_premium_panel(
    ax,
    prem_df: pd.DataFrame,
    price_df: pd.DataFrame,
    bar_width: float,
    expiration_type: str,
    min_premium_highlight: float = 2.0,
    display_mode: str = "WTD_STYLE",
    itm_call_hedge_threshold: float = 0.30,
) -> None:
    """
    Plot options premium as asymmetrical histogram with configurable display modes.

    Display modes:
    - TOTAL: Original behavior - total call/put premium (backwards compatible)
    - WTD_STYLE: OTM focus with ITM call hedge warning (per WTD's methodology)
    - FULL_BREAKDOWN: Show all 4 categories (OTM/ITM x Call/Put) as stacked bars
    """
    if prem_df.empty:
        ax.text(
            0.5, 0.5, f"No {expiration_type} options data",
            ha="center", va="center", transform=ax.transAxes,
            color=COLORS["neutral"], fontsize=10,
        )
        ax.set_ylabel("Premium ($M)", color=COLORS["white"], fontsize=10)
        return

    prem_df = prem_df.copy()
    dates = mdates.date2num(prem_df["date"])
    bar_width_premium = bar_width * 0.6

    # Extract data based on display mode
    total_call = prem_df["total_call_premium"].fillna(0).values
    total_put = prem_df["total_put_premium"].fillna(0).values

    # Check if ITM/OTM columns exist (for backwards compatibility with old data)
    has_itm_otm = "otm_call_premium" in prem_df.columns and prem_df["otm_call_premium"].notna().any()

    if has_itm_otm:
        otm_call = prem_df["otm_call_premium"].fillna(0).values
        itm_call = prem_df["itm_call_premium"].fillna(0).values
        otm_put = prem_df["otm_put_premium"].fillna(0).values
        itm_put = prem_df["itm_put_premium"].fillna(0).values
    else:
        # Fallback to total if ITM/OTM not available
        otm_call = total_call
        itm_call = np.zeros_like(total_call)
        otm_put = total_put
        itm_put = np.zeros_like(total_put)

    legend_handles = []

    if display_mode == "TOTAL" or not has_itm_otm:
        # Original behavior: total call/put premium
        ax.bar(dates, total_call, width=bar_width_premium, color=COLORS["call_premium"],
               alpha=0.85, edgecolor="none", zorder=2)
        ax.bar(dates, -total_put, width=bar_width_premium, color=COLORS["put_premium"],
               alpha=0.85, edgecolor="none", zorder=2)

        legend_handles = [
            Patch(facecolor=COLORS["call_premium"], edgecolor="none", alpha=0.85, label="Calls"),
            Patch(facecolor=COLORS["put_premium"], edgecolor="none", alpha=0.85, label="Puts"),
        ]
        max_val = max(total_call.max() if len(total_call) > 0 else 0,
                      total_put.max() if len(total_put) > 0 else 0)

    elif display_mode == "WTD_STYLE":
        # WTD style: OTM only with ITM call hedge warning
        # OTM Calls = Cyan (bullish), OTM Puts = Red (bearish)
        ax.bar(dates, otm_call, width=bar_width_premium, color=COLORS["otm_call"],
               alpha=0.85, edgecolor="none", zorder=2)
        ax.bar(dates, -otm_put, width=bar_width_premium, color=COLORS["otm_put"],
               alpha=0.85, edgecolor="none", zorder=2)

        # Add ITM call hedge warning markers with size proportional to magnitude
        for i, (d, itm_c, total_c) in enumerate(zip(dates, itm_call, total_call)):
            if total_c > 0 and (itm_c / total_c) > itm_call_hedge_threshold:
                # Scale marker size based on ITM magnitude (min 60, max 200)
                itm_ratio = itm_c / total_c
                marker_size = min(200, max(60, 60 + (itm_ratio - itm_call_hedge_threshold) * 400))
                ax.scatter(d, otm_call[i] + 0.5, marker='^', s=marker_size,
                          color=COLORS["hedge_warning"], edgecolor="black",
                          linewidth=0.5, zorder=5)

        legend_handles = [
            Patch(facecolor=COLORS["otm_call"], edgecolor="none", alpha=0.85, label="OTM Calls (Bullish)"),
            Patch(facecolor=COLORS["otm_put"], edgecolor="none", alpha=0.85, label="OTM Puts (Bearish)"),
            Line2D([0], [0], marker='^', color='none', markerfacecolor=COLORS["hedge_warning"],
                   markeredgecolor="black", markersize=8, label="ITM Hedge Warning"),
        ]
        max_val = max(otm_call.max() if len(otm_call) > 0 else 0,
                      otm_put.max() if len(otm_put) > 0 else 0)

        # Draw directional gauge if score available
        if "directional_score" in prem_df.columns:
            recent_scores = prem_df["directional_score"].dropna()
            if len(recent_scores) > 0:
                latest_score = recent_scores.iloc[-1]
                _draw_directional_gauge(ax, latest_score)

    elif display_mode == "FULL_BREAKDOWN":
        # Full breakdown: stacked bars for all 4 categories
        # Calls (upward): OTM on bottom, ITM stacked on top
        ax.bar(dates, otm_call, width=bar_width_premium, color=COLORS["otm_call"],
               alpha=0.85, edgecolor="none", zorder=2)
        ax.bar(dates, itm_call, width=bar_width_premium, color=COLORS["itm_call"],
               alpha=0.85, edgecolor="none", zorder=2, bottom=otm_call)

        # Puts (downward): OTM on bottom, ITM stacked below
        ax.bar(dates, -otm_put, width=bar_width_premium, color=COLORS["otm_put"],
               alpha=0.85, edgecolor="none", zorder=2)
        ax.bar(dates, -itm_put, width=bar_width_premium, color=COLORS["itm_put"],
               alpha=0.85, edgecolor="none", zorder=2, bottom=-otm_put)

        legend_handles = [
            Patch(facecolor=COLORS["otm_call"], edgecolor="none", alpha=0.85, label="OTM Calls (Bullish)"),
            Patch(facecolor=COLORS["itm_call"], edgecolor="none", alpha=0.85, label="ITM Calls (Hedge)"),
            Patch(facecolor=COLORS["otm_put"], edgecolor="none", alpha=0.85, label="OTM Puts (Bearish)"),
            Patch(facecolor=COLORS["itm_put"], edgecolor="none", alpha=0.85, label="ITM Puts (Bullish)"),
        ]
        max_val = max((otm_call + itm_call).max() if len(otm_call) > 0 else 0,
                      (otm_put + itm_put).max() if len(otm_put) > 0 else 0)

        # Draw directional gauge if score available
        if "directional_score" in prem_df.columns:
            recent_scores = prem_df["directional_score"].dropna()
            if len(recent_scores) > 0:
                latest_score = recent_scores.iloc[-1]
                _draw_directional_gauge(ax, latest_score)

    else:
        # Default to TOTAL if unknown mode
        ax.bar(dates, total_call, width=bar_width_premium, color=COLORS["call_premium"],
               alpha=0.85, edgecolor="none", zorder=2)
        ax.bar(dates, -total_put, width=bar_width_premium, color=COLORS["put_premium"],
               alpha=0.85, edgecolor="none", zorder=2)
        legend_handles = [
            Patch(facecolor=COLORS["call_premium"], edgecolor="none", alpha=0.85, label="Calls"),
            Patch(facecolor=COLORS["put_premium"], edgecolor="none", alpha=0.85, label="Puts"),
        ]
        max_val = max(total_call.max() if len(total_call) > 0 else 0,
                      total_put.max() if len(total_put) > 0 else 0)

    # Zero line
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

    # Auto-scale y-axis symmetrically
    if max_val > 0:
        margin = max_val * 0.20  # Slightly larger margin for warning markers
        ax.set_ylim(-max_val - margin, max_val + margin)

    ax.set_ylabel("Premium ($M)", color=COLORS["white"], fontsize=10)

    # Title with expiration type and display mode
    title_text = "0DTE Options Premium" if expiration_type == "0DTE" else "Weekly Options Premium"
    if display_mode == "WTD_STYLE":
        title_text += " (OTM)"
    elif display_mode == "FULL_BREAKDOWN":
        title_text += " (ITM/OTM)"
    ax.set_title(
        title_text,
        color=COLORS["white"],
        fontsize=9,
        fontweight="bold",
        loc="left",
    )

    # Legend
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text"])


def _plot_signal_markers(
    ax,
    df: pd.DataFrame,
    dates: np.ndarray,
    offset: float,
) -> None:
    scores = df["accumulation_score_display"]
    if scores.isna().all():
        return

    buy_mask = scores >= 70
    sell_mask = scores <= 30
    if buy_mask.any():
        ax.scatter(
            dates[buy_mask],
            df.loc[buy_mask, "low"] - offset,
            s=SIGNAL_MARKER_SIZE,
            c=COLORS["green"],
            edgecolors=COLORS["white"],
            linewidths=SIGNAL_EDGE_WIDTH,
            zorder=6,
            label="Accumulation (>70)",
        )
    if sell_mask.any():
        ax.scatter(
            dates[sell_mask],
            df.loc[sell_mask, "high"] + offset,
            s=SIGNAL_MARKER_SIZE,
            c=COLORS["red"],
            edgecolors=COLORS["white"],
            linewidths=SIGNAL_EDGE_WIDTH,
            zorder=6,
            label="Distribution (<30)",
        )


def plot_price_chart(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path,
    timeframe: str,
    options_0dte_df: Optional[pd.DataFrame] = None,
    options_weekly_df: Optional[pd.DataFrame] = None,
    min_premium_highlight: float = 2.0,
    options_premium_display_mode: str = "WTD_STYLE",
    itm_call_hedge_threshold: float = 0.30,
) -> Path:
    if df.empty:
        logger.warning("No OHLCV data to plot for %s", symbol)
        return output_path

    timeframe = _resolve_timeframe(timeframe)
    df = _resample_ohlcv(df, timeframe)
    if df.empty:
        logger.warning("No OHLCV data after resample for %s", symbol)
        return output_path
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        logger.warning("No complete OHLC data to plot for %s", symbol)
        return output_path

    fig_width = _compute_fig_width(len(df))

    # Determine which options panels to show
    has_0dte = options_0dte_df is not None and not options_0dte_df.empty
    has_weekly = options_weekly_df is not None and not options_weekly_df.empty
    is_index = symbol.upper() in DAILY_EXPIRATION_TICKERS

    # Build panel configuration dynamically
    # Base panels: Price, Volume, Score
    # Optional: 0DTE (indices only), Weekly
    panel_count = 3  # price, volume, score
    height_ratios = [4.5, 1, 1.6]

    if has_weekly:
        panel_count += 1
        height_ratios.append(1.2)  # Weekly options panel

    if has_0dte and is_index:
        panel_count += 1
        height_ratios.insert(3, 1.2)  # 0DTE before weekly

    # Adjust total figure height based on panel count
    fig_height = 10.5 + (panel_count - 3) * 2.0

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )

    # Unpack axes based on what panels we have
    ax_idx = 0
    ax_price = axes[ax_idx]; ax_idx += 1
    ax_vol = axes[ax_idx]; ax_idx += 1
    ax_score = axes[ax_idx]; ax_idx += 1

    ax_0dte = None
    ax_weekly = None

    if has_0dte and is_index:
        ax_0dte = axes[ax_idx]; ax_idx += 1
    if has_weekly:
        ax_weekly = axes[ax_idx]; ax_idx += 1

    fig.patch.set_facecolor(COLORS["background"])

    _apply_axis_style(ax_price)
    _apply_axis_style(ax_vol)
    _apply_axis_style(ax_score)
    if ax_0dte is not None:
        _apply_axis_style(ax_0dte)
    if ax_weekly is not None:
        _apply_axis_style(ax_weekly)

    dates, bar_width = _plot_ohlcv_bars(ax_price, df)
    _plot_volume_bars(ax_vol, df, dates, bar_width)
    if len(dates) > 0:
        pad = bar_width
        ax_score.set_xlim(dates.min() - pad, dates.max() + pad)

    y_min = df["low"].min()
    y_max = df["high"].max()
    if pd.isna(y_min) or pd.isna(y_max):
        y_min = df["close"].min()
        y_max = df["close"].max()
    y_span = y_max - y_min if pd.notna(y_max) and pd.notna(y_min) else 0
    if not y_span or y_span <= 0:
        y_span = float(df["close"].mean() or 1.0)
    offset = y_span * 0.04

    _plot_signal_markers(ax_price, df, dates, offset)

    ax_price.set_ylabel("Price", color=COLORS["white"], fontsize=10)
    ax_vol.set_ylabel("Volume", color=COLORS["white"], fontsize=10)
    ax_price.set_ylim(y_min - offset * 3, y_max + offset * 3)

    timeframe_label = timeframe.capitalize()
    ax_price.set_title(
        f"{symbol} Price Action ({timeframe_label})",
        color=COLORS["white"],
        fontsize=11,
        fontweight="bold",
        loc="left",
    )

    if timeframe == "monthly":
        locator = mdates.MonthLocator(interval=1)
        formatter = mdates.DateFormatter("%y-%m")
    elif timeframe == "weekly":
        locator = mdates.WeekdayLocator(byweekday=mdates.FR, interval=1)
        formatter = mdates.DateFormatter("%y-%m-%d")
    else:
        locator = mdates.DayLocator(interval=1)
        formatter = mdates.DateFormatter("%y-%m-%d")

    ax_price.tick_params(axis="x", labelbottom=False)
    ax_vol.tick_params(axis="x", labelbottom=False)

    # Determine the bottom-most panel for x-axis labels
    if ax_weekly is not None:
        ax_bottom = ax_weekly
        ax_score.tick_params(axis="x", labelbottom=False)
        if ax_0dte is not None:
            ax_0dte.tick_params(axis="x", labelbottom=False)
    elif ax_0dte is not None:
        ax_bottom = ax_0dte
        ax_score.tick_params(axis="x", labelbottom=False)
    else:
        ax_bottom = ax_score

    ax_bottom.xaxis.set_major_locator(locator)
    ax_bottom.xaxis.set_major_formatter(formatter)
    plt.setp(ax_bottom.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    score_display = df["accumulation_score_display"].fillna(50)
    confidence = df["confidence"].fillna(0.5)
    if score_display.isna().all():
        logger.warning("Accumulation score missing for %s in price chart range.", symbol)

    from matplotlib.colors import LinearSegmentedColormap

    score_cmap = LinearSegmentedColormap.from_list(
        "score_cmap",
        [
            (0.0, "#b026ff"),
            (0.5, "#555555"),
            (1.0, "#00ff88"),
        ],
    )

    for d, score, conf in zip(df["date"], score_display, confidence):
        if pd.isna(score):
            score = 50
        if pd.isna(conf):
            conf = 0.5

        norm_score = np.clip(score / 100.0, 0, 1)
        dev = norm_score - 0.5
        norm_score = 0.5 + np.sign(dev) * (abs(dev) ** 0.85)
        norm_score = np.clip(norm_score, 0, 1)
        bar_color = score_cmap(norm_score)
        alpha = 0.8 if conf >= 0.6 else 0.4
        bar_height = score / 100.0
        ax_score.bar(d, bar_height, bottom=0, color=bar_color, alpha=alpha, width=0.8, zorder=2)

        conf_height = 0.08 * conf
        if conf >= 0.7:
            conf_color = COLORS["green"]
        elif conf >= 0.4:
            conf_color = COLORS["yellow"]
        else:
            conf_color = COLORS["red"]
        ax_score.bar(d, conf_height, bottom=-0.12, color=conf_color, alpha=0.6, width=0.6, zorder=3)

        label_y = max(bar_height / 2, 0.15)
        ax_score.text(
            d,
            label_y,
            f"{score:.0f}",
            ha="center",
            va="center",
            fontsize=7,
            color=COLORS["white"],
            fontweight="bold",
            zorder=4,
        )

    ax_score.axhline(y=0.30, color=COLORS["red"], linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    ax_score.axhline(y=0.50, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    ax_score.axhline(y=0.70, color=COLORS["green"], linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)

    ax_score.set_ylim(-0.15, 1.05)
    ax_score.set_yticks([0, 0.3, 0.5, 0.7, 1.0])
    ax_score.set_yticklabels(["0", "30", "50", "70", "100"])
    ax_score.set_ylabel("Score", color=COLORS["white"], fontsize=10)
    ax_score.set_title(
        "Accumulation Score (0-100)",
        color=COLORS["white"],
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    ax_score.grid(False)
    for spine in ax_score.spines.values():
        spine.set_visible(False)

    legend_handles_score = [
        Patch(facecolor="#00ff88", edgecolor="none", alpha=0.8, label=">70 Accum"),
        Patch(facecolor="#555555", edgecolor="none", alpha=0.8, label="30-70 Neutral"),
        Patch(facecolor="#b026ff", edgecolor="none", alpha=0.8, label="<30 Distrib"),
        Line2D([0], [0], color=COLORS["green"], linewidth=4, alpha=0.6, label="High Conf"),
        Line2D([0], [0], color=COLORS["yellow"], linewidth=4, alpha=0.6, label="Med Conf"),
        Line2D([0], [0], color=COLORS["red"], linewidth=4, alpha=0.6, label="Low Conf"),
    ]
    legend = ax_score.legend(
        legend_handles_score,
        [h.get_label() for h in legend_handles_score],
        loc="upper right",
        ncol=len(legend_handles_score),
        fontsize=7,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text"])

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"],
               markeredgecolor=COLORS["white"], markersize=6, label="Accumulation (>70)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["red"],
               markeredgecolor=COLORS["white"], markersize=6, label="Distribution (<30)"),
    ]
    legend = ax_price.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text"])

    # Plot options premium panels if data available
    if ax_0dte is not None and options_0dte_df is not None:
        _plot_options_premium_panel(
            ax_0dte, options_0dte_df, df, bar_width, "0DTE", min_premium_highlight,
            options_premium_display_mode, itm_call_hedge_threshold
        )

    if ax_weekly is not None and options_weekly_df is not None:
        _plot_options_premium_panel(
            ax_weekly, options_weekly_df, df, bar_width, "WEEKLY", min_premium_highlight,
            options_premium_display_mode, itm_call_hedge_threshold
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=COLORS["background"], edgecolor="none")
    plt.close(fig)

    logger.info("Saved price chart: %s", output_path)
    return output_path


def render_price_charts(
    db_path: Path,
    output_dir: Path,
    dates: list[date],
    tickers: list[str],
    timeframe: str = "daily",
    min_premium_highlight: float = 2.0,
    options_premium_display_mode: str = "WTD_STYLE",
    itm_call_hedge_threshold: float = 0.30,
) -> list[Path]:
    if not dates or not tickers:
        return []

    output_paths: list[Path] = []
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        for symbol in tickers:
            df = fetch_price_data(conn, symbol, dates)
            if df.empty:
                logger.warning("No OHLCV data for %s, skipping price chart", symbol)
                continue

            # Fetch options premium data if available
            options_0dte_df = fetch_options_premium_data(conn, symbol, dates, "0DTE")
            options_weekly_df = fetch_options_premium_data(conn, symbol, dates, "WEEKLY")

            if len(dates) == 1:
                file_suffix = dates[0].strftime("%Y-%m-%d")
            else:
                sorted_dates = sorted(dates)
                file_suffix = f"{sorted_dates[0].strftime('%Y%m%d')}_{sorted_dates[-1].strftime('%Y%m%d')}"

            safe_timeframe = _resolve_timeframe(timeframe)
            output_path = output_dir / f"{symbol.lower()}_price_ohlcv_{safe_timeframe}_{file_suffix}.png"
            plot_price_chart(
                df, symbol, output_path, safe_timeframe,
                options_0dte_df=options_0dte_df,
                options_weekly_df=options_weekly_df,
                min_premium_highlight=min_premium_highlight,
                options_premium_display_mode=options_premium_display_mode,
                itm_call_hedge_threshold=itm_call_hedge_threshold,
            )
            output_paths.append(output_path)
    finally:
        conn.close()

    return output_paths


def main() -> None:
    """CLI entry point for standalone OHLCV chart generation."""
    parser = argparse.ArgumentParser(description="Render OHLCV price charts as PNG")
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
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        choices=list(TIMEFRAME_RULES.keys()),
        help="Bar timeframe: daily, weekly, or monthly (default: config)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config()
    dates = [date.fromisoformat(item.strip()) for item in args.dates.split(",") if item.strip()]
    tickers = (
        [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
        if args.tickers
        else config.tickers
    )
    output_dir = Path(args.output_dir) if args.output_dir else config.price_chart_dir
    timeframe = args.timeframe or config.price_bar_timeframe

    render_price_charts(
        db_path=config.db_path,
        output_dir=output_dir,
        dates=dates,
        tickers=tickers,
        timeframe=timeframe,
    )


if __name__ == "__main__":
    main()
