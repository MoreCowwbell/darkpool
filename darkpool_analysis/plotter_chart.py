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

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from .config import load_config
except ImportError:
    from config import load_config

logger = logging.getLogger(__name__)

COLORS = {
    "background": "#0f0f10",
    "panel_bg": "#141416",
    "text": "#e6e6e6",
    "grid": "#2a2a2d",
    "white": "#ffffff",
    "green": "#00ff88",
    "red": "#ff6b6b",
    "blue": "#4aa3ff",
    "orange": "#ff9f43",
    "neutral": "#6b6b6b",
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
    query = """
        SELECT
            p.trade_date AS date,
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume,
            d.accumulation_score_display
        FROM polygon_daily_agg_raw p
        LEFT JOIN daily_metrics d
            ON d.symbol = p.symbol AND d.date = p.trade_date
        WHERE p.symbol = ? AND p.trade_date BETWEEN ? AND ?
        ORDER BY p.trade_date
    """
    df = conn.execute(query, [symbol, min_date, max_date]).df()
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "accumulation_score_display"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "accumulation_score_display" not in df.columns:
        df["accumulation_score_display"] = pd.NA

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
            df.loc[buy_mask, "high"] + offset,
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
            df.loc[sell_mask, "low"] - offset,
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

    plt.style.use("dark_background")
    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor(COLORS["background"])

    _apply_axis_style(ax_price)
    _apply_axis_style(ax_vol)

    dates, bar_width = _plot_ohlcv_bars(ax_price, df)
    _plot_volume_bars(ax_vol, df, dates, bar_width)

    y_min = df["low"].min()
    y_max = df["high"].max()
    if pd.isna(y_min) or pd.isna(y_max):
        y_min = df["close"].min()
        y_max = df["close"].max()
    y_span = y_max - y_min if pd.notna(y_max) and pd.notna(y_min) else 0
    if not y_span or y_span <= 0:
        y_span = float(df["close"].mean() or 1.0)
    offset = y_span * 0.02

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

    ax_vol.xaxis.set_major_locator(locator)
    ax_vol.xaxis.set_major_formatter(formatter)
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    legend_handles = [
        Patch(facecolor=COLORS["blue"], edgecolor=COLORS["blue"], label="Up bar"),
        Patch(facecolor=COLORS["orange"], edgecolor=COLORS["orange"], label="Down bar"),
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

            if len(dates) == 1:
                file_suffix = dates[0].strftime("%Y-%m-%d")
            else:
                sorted_dates = sorted(dates)
                file_suffix = f"{sorted_dates[0].strftime('%Y%m%d')}_{sorted_dates[-1].strftime('%Y%m%d')}"

            safe_timeframe = _resolve_timeframe(timeframe)
            output_path = output_dir / f"{symbol.lower()}_price_ohlcv_{safe_timeframe}_{file_suffix}.png"
            plot_price_chart(df, symbol, output_path, safe_timeframe)
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
