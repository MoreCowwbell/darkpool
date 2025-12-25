"""
Multi-panel dark-theme plotter for daily metrics.

Generates PNG visualizations from DuckDB daily_metrics data:
- Panel 1: log(Buy/Sell) directional flow
- Panel 2: Short ratio z-score (pressure signal)
- Panel 3: OTC volume (participation anchor)
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

# Dark theme colors (matching table_renderer.py)
COLORS = {
    "background": "#0f0f10",
    "panel_bg": "#141416",
    "text": "#e6e6e6",
    "text_muted": "#8b8b8b",
    "grid": "#2a2a2d",
    "green": "#00ff88",
    "red": "#ff6b6b",
    "yellow": "#ffd700",
    "cyan": "#00d4ff",
    "white": "#ffffff",
    "neutral": "#6b6b6b",
}

# Thresholds for log(Buy/Sell)
LOG_THRESHOLD_HIGH = 0.223  # ln(1.25) - accumulation signal
LOG_THRESHOLD_LOW = -0.223  # ln(0.80) - distribution signal


def fetch_metrics_for_plot(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    dates: list[date],
) -> pd.DataFrame:
    """Fetch daily_metrics for a single symbol across multiple dates."""
    if not dates:
        return pd.DataFrame()

    date_placeholders = ", ".join(["?" for _ in dates])
    query = f"""
        SELECT
            date,
            symbol,
            log_buy_sell,
            short_ratio,
            short_ratio_z,
            return_1d,
            return_z,
            otc_off_exchange_volume,
            data_quality,
            pressure_context_label
        FROM daily_metrics
        WHERE symbol = ? AND date IN ({date_placeholders})
        ORDER BY date
    """
    params = [symbol] + list(dates)
    df = conn.execute(query, params).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _get_point_color(value: float, high: float, low: float) -> str:
    """Return color based on threshold crossing."""
    if pd.isna(value):
        return COLORS["neutral"]
    if value > high:
        return COLORS["green"]
    if value < low:
        return COLORS["red"]
    return COLORS["cyan"]


def _format_volume(value: float) -> str:
    """Format volume for display."""
    if pd.isna(value):
        return "NA"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:,.0f}"


def _plot_smooth_line(ax, dates, values, color, valid_mask):
    """Plot a smooth PCHIP-interpolated line through valid data points."""
    if valid_mask.sum() >= 3:
        valid_dates = dates[valid_mask]
        valid_values = values[valid_mask]
        date_nums = mdates.date2num(valid_dates)

        x_smooth = np.linspace(date_nums.min(), date_nums.max(), 150)
        interp = PchipInterpolator(date_nums, valid_values.to_numpy())
        y_smooth = interp(x_smooth)

        ax.plot(
            mdates.num2date(x_smooth),
            y_smooth,
            color=color,
            linewidth=2,
            alpha=0.6,
            zorder=2,
        )
    else:
        ax.plot(dates[valid_mask], values[valid_mask], color=color, linewidth=1.5, alpha=0.7)


def plot_symbol_metrics(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    """
    Generate a 3-panel plot for a single symbol.

    Panel 1: log(Buy/Sell) with accumulation/distribution thresholds
    Panel 2: Short ratio z-score with high/low thresholds
    Panel 3: OTC off-exchange volume bars
    """
    if df.empty:
        logger.warning("No data to plot for %s", symbol)
        return output_path

    # Set up dark theme
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor(COLORS["background"])

    for ax in axes:
        ax.set_facecolor(COLORS["panel_bg"])
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["grid"])
        ax.spines["bottom"].set_color(COLORS["grid"])
        ax.grid(True, alpha=0.3, color=COLORS["grid"], linestyle="--")

    dates = df["date"]

    # Panel 1: log(Buy/Sell)
    ax1 = axes[0]
    log_bs = df["log_buy_sell"]
    colors1 = [_get_point_color(v, LOG_THRESHOLD_HIGH, LOG_THRESHOLD_LOW) for v in log_bs]

    ax1.axhline(y=0, color=COLORS["neutral"], linestyle="-", linewidth=1, alpha=0.5)
    ax1.axhline(y=LOG_THRESHOLD_HIGH, color=COLORS["green"], linestyle="--", linewidth=1, alpha=0.7)
    ax1.axhline(y=LOG_THRESHOLD_LOW, color=COLORS["red"], linestyle="--", linewidth=1, alpha=0.7)

    # Plot smooth line and scatter
    valid_mask = ~log_bs.isna()
    if valid_mask.any():
        _plot_smooth_line(ax1, dates, log_bs, COLORS["cyan"], valid_mask)
        ax1.scatter(dates, log_bs, c=colors1, s=60, zorder=5, edgecolors=COLORS["white"], linewidths=0.5)

    ax1.set_ylabel("log(Buy/Sell)", color=COLORS["text"], fontsize=10)
    ax1.set_title(f"{symbol} - Directional Flow", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")

    # Add threshold labels
    ax1.text(0.99, 0.95, "Accumulation", transform=ax1.transAxes, fontsize=8,
             color=COLORS["green"], ha="right", va="top")
    ax1.text(0.99, 0.05, "Distribution", transform=ax1.transAxes, fontsize=8,
             color=COLORS["red"], ha="right", va="bottom")

    # Panel 2: Short ratio z-score
    ax2 = axes[1]
    short_z = df["short_ratio_z"]
    colors2 = [_get_point_color(v, 1.0, -1.0) for v in short_z]

    ax2.axhline(y=0, color=COLORS["neutral"], linestyle="-", linewidth=1, alpha=0.5)
    ax2.axhline(y=1.0, color=COLORS["red"], linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(y=-1.0, color=COLORS["green"], linestyle="--", linewidth=1, alpha=0.7)

    valid_mask2 = ~short_z.isna()
    if valid_mask2.any():
        _plot_smooth_line(ax2, dates, short_z, COLORS["yellow"], valid_mask2)
        ax2.scatter(dates, short_z, c=colors2, s=60, zorder=5, edgecolors=COLORS["white"], linewidths=0.5)

    ax2.set_ylabel("Short Ratio Z", color=COLORS["text"], fontsize=10)
    ax2.set_title("Short Pressure (Z-Score)", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")

    ax2.text(0.99, 0.95, "High Pressure", transform=ax2.transAxes, fontsize=8,
             color=COLORS["red"], ha="right", va="top")
    ax2.text(0.99, 0.05, "Low Pressure", transform=ax2.transAxes, fontsize=8,
             color=COLORS["green"], ha="right", va="bottom")

    # Panel 3: OTC Volume bars
    ax3 = axes[2]
    otc_vol = df["otc_off_exchange_volume"]
    data_quality = df["data_quality"]

    bar_colors = [COLORS["green"] if q == "OTC_ANCHORED" else COLORS["yellow"] for q in data_quality]

    valid_mask3 = ~otc_vol.isna()
    if valid_mask3.any():
        ax3.bar(dates[valid_mask3], otc_vol[valid_mask3], color=[bar_colors[i] for i, v in enumerate(valid_mask3) if v],
                alpha=0.8, width=0.8)

    ax3.set_ylabel("OTC Volume", color=COLORS["text"], fontsize=10)
    ax3.set_title("Off-Exchange Participation", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_volume(x)))

    # Legend for data quality
    ax3.text(0.99, 0.95, "OTC_ANCHORED", transform=ax3.transAxes, fontsize=8,
             color=COLORS["green"], ha="right", va="top")
    ax3.text(0.99, 0.85, "PRE_OTC", transform=ax3.transAxes, fontsize=8,
             color=COLORS["yellow"], ha="right", va="top")

    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Title and footer
    fig.suptitle(
        f"Institutional Pressure Analysis{title_suffix}",
        color=COLORS["white"],
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5, 0.01,
        "FINRA does not publish trade direction. All buy/sell values are inferred estimates.",
        ha="center",
        fontsize=8,
        color=COLORS["text_muted"],
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=COLORS["background"], edgecolor="none")
    plt.close(fig)

    logger.info("Saved plot: %s", output_path)
    return output_path


def render_metrics_plots(
    db_path: Path,
    output_dir: Path,
    dates: list[date],
    tickers: list[str],
) -> list[Path]:
    """Render multi-panel plots for each ticker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        for symbol in tickers:
            df = fetch_metrics_for_plot(conn, symbol, dates)
            if df.empty:
                logger.warning("No data for %s, skipping plot", symbol)
                continue

            if len(dates) == 1:
                file_suffix = dates[0].strftime("%Y-%m-%d")
            else:
                sorted_dates = sorted(dates)
                file_suffix = f"{sorted_dates[0].strftime('%Y%m%d')}_{sorted_dates[-1].strftime('%Y%m%d')}"

            output_path = output_dir / f"{symbol.lower()}_metrics_{file_suffix}.png"
            plot_symbol_metrics(df, symbol, output_path)
            output_paths.append(output_path)
    finally:
        conn.close()

    return output_paths


def main() -> None:
    """CLI entry point for standalone plot generation."""
    parser = argparse.ArgumentParser(description="Render daily metrics plots as PNG")
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
    dates_list = [date.fromisoformat(d.strip()) for d in args.dates.split(",")]
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else config.tickers
    output_dir = Path(args.output_dir) if args.output_dir else config.plot_dir

    paths = render_metrics_plots(
        db_path=config.db_path,
        output_dir=output_dir,
        dates=dates_list,
        tickers=tickers,
    )
    print(f"Generated {len(paths)} plot(s)")


if __name__ == "__main__":
    main()
