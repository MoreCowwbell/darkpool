"""
Multi-panel dark-theme plotter for daily metrics.

Generates PNG visualizations from DuckDB daily_metrics data:
- Layered: short sale buy ratio, lit buy ratios, OTC buy/sell with decision strip
- Short-only: short ratio, short sale volume, close price
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

MAIN_LINE_WIDTH = 2.3
SECONDARY_LINE_WIDTH = 1.8
THRESHOLD_LINE_WIDTH = 1.0
CENTERLINE_WIDTH = 1.4
GRID_ALPHA = 0.18
MARKER_SIZE = 28
MARKER_SIZE_SMALL = 22
BAR_ALPHA_PRIMARY = 0.35
BAR_ALPHA_SECONDARY = 0.25
BOT_THRESHOLD = 1.25
SELL_THRESHOLD = 0.75
NEUTRAL_RATIO = 1.0


def _clean_upper_bound(value: float, min_upper: float = 2.0) -> float:
    if value <= min_upper:
        return min_upper
    if value <= 5.0:
        return np.ceil(value * 2) / 2
    return float(np.ceil(value))


def compute_abs_ratio_ylim(
    values: pd.Series,
    min_upper: float = 2.0,
    lower: float = 0.0,
) -> tuple[float, float, float]:
    series = pd.to_numeric(values, errors="coerce")
    max_val = series.max(skipna=True)
    if pd.isna(max_val):
        max_val = min_upper
    max_val = max(float(max_val), 1.0)
    upper = _clean_upper_bound(max_val, min_upper=min_upper)
    step = 0.5 if upper <= 5.0 else 1.0
    return lower, upper, step


def compute_log_ratio_ylim(values: pd.Series, headroom: float = 0.08) -> tuple[float, float, float]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        bound = 0.5
    else:
        max_abs = series.abs().max()
        bound = max_abs * (1 + headroom)
    if bound <= 0:
        bound = 0.5
    if bound <= 0.5:
        step = 0.1
    elif bound <= 1.0:
        step = 0.2
    elif bound <= 2.0:
        step = 0.5
    else:
        step = 1.0
    bound = float(np.ceil(bound / step) * step)
    return -bound, bound, step


def _apply_primary_axis_style(ax) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.grid(True, alpha=GRID_ALPHA, color=COLORS["grid"], linestyle="--")


def _apply_secondary_axis_style(ax) -> None:
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_color(COLORS["grid"])
    ax.spines["left"].set_visible(False)
    ax.grid(False)


def _add_panel_legend(ax, handles, labels, loc: str = "upper left") -> None:
    if not handles:
        return
    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        fontsize=8,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text"])


def _set_abs_ratio_axis(
    ax,
    values: pd.Series,
    neutral_value: float = NEUTRAL_RATIO,
    draw_neutral: bool = True,
    linestyle: str = "--",
    linewidth: float = CENTERLINE_WIDTH,
    alpha: float = 0.7,
) -> Line2D | None:
    ymin, ymax, step = compute_abs_ratio_ylim(values)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + step * 0.5, step))
    if not draw_neutral:
        return None
    return ax.axhline(
        y=neutral_value,
        color=COLORS["neutral"],
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )


def _set_log_ratio_axis(
    ax,
    values: pd.Series,
    neutral_value: float = 0.0,
    draw_neutral: bool = True,
    linestyle: str = "--",
    linewidth: float = CENTERLINE_WIDTH,
    alpha: float = 0.7,
) -> Line2D | None:
    ymin, ymax, step = compute_log_ratio_ylim(values)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + step * 0.5, step))
    if not draw_neutral:
        return None
    return ax.axhline(
        y=neutral_value,
        color=COLORS["neutral"],
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )


def _add_ratio_thresholds(ax, bot: float = BOT_THRESHOLD, sell: float = SELL_THRESHOLD) -> None:
    ax.axhline(
        y=bot,
        color=COLORS["green"],
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        alpha=0.8,
        zorder=1,
    )
    ax.axhline(
        y=sell,
        color=COLORS["red"],
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        alpha=0.8,
        zorder=1,
    )

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
            short_ratio,
            short_ratio_z,
            short_buy_sell_ratio,
            short_buy_sell_ratio_z,
            short_ratio_denominator_type,
            short_ratio_denominator_value,
            short_buy_volume,
            log_buy_sell,
            lit_buy_ratio,
            lit_buy_ratio_z,
            lit_buy_volume,
            lit_sell_volume,
            lit_total_volume,
            otc_off_exchange_volume,
            otc_buy_volume,
            otc_sell_volume,
            otc_weekly_buy_ratio,
            otc_buy_ratio_z,
            close,
            return_1d,
            return_z,
            otc_status,
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


def _get_denom_color(value: str) -> str:
    if value is None or pd.isna(value):
        return COLORS["neutral"]
    if value == "FINRA_TOTAL":
        return COLORS["cyan"]
    if value == "POLYGON_TOTAL":
        return COLORS["yellow"]
    return COLORS["neutral"]


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


def _plot_smooth_line(
    ax,
    dates,
    values,
    color,
    valid_mask,
    linewidth: float = MAIN_LINE_WIDTH,
    alpha: float = 0.85,
    zorder: int = 3,
):
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
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        ax.plot(
            dates[valid_mask],
            values[valid_mask],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )


def plot_symbol_metrics(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    """
    Generate a 3-panel plot plus decision strip for a single symbol.

    Panel 1: Short Sale Buy Ratio
    Panel 2: Lit Buy Ratio + Log Buy Ratio
    Panel 3: OTC buy/sell volumes + OTC Weekly Buy Ratio
    Decision strip: Accumulating/Distribution/Neutral labels
    """
    if df.empty:
        logger.warning("No data to plot for %s", symbol)
        return output_path

    # Set up dark theme
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 0.7]},
    )
    fig.patch.set_facecolor(COLORS["background"])

    for ax in axes:
        _apply_primary_axis_style(ax)

    dates = df["date"]

    # Panel 1: Short Sale Buy Ratio
    ax1 = axes[0]
    short_ratio = df["short_buy_sell_ratio"]

    valid_mask = ~short_ratio.isna()
    if valid_mask.any():
        _plot_smooth_line(ax1, dates, short_ratio, COLORS["cyan"], valid_mask, linewidth=MAIN_LINE_WIDTH)
        ax1.scatter(
            dates,
            short_ratio,
            c=COLORS["cyan"],
            s=MARKER_SIZE,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    _set_abs_ratio_axis(
        ax1,
        short_ratio,
        neutral_value=NEUTRAL_RATIO,
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        alpha=0.6,
    )
    _add_ratio_thresholds(ax1, bot=BOT_THRESHOLD, sell=SELL_THRESHOLD)
    ax1.set_ylabel("Short Sale Buy/Sell Ratio", color=COLORS["text"], fontsize=10)
    ax1.set_title(f"{symbol} - Short Sale Buy/Sell Ratio", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    legend_handles_1 = [
        Line2D([0], [0], color=COLORS["cyan"], linewidth=MAIN_LINE_WIDTH, label="Short Sale Buy/Sell Ratio"),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["cyan"],
            markeredgecolor=COLORS["white"],
            markersize=4.5,
            label="Observations",
        ),
        Line2D([0], [0], color=COLORS["neutral"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="Neutral (1.0)"),
        Line2D([0], [0], color=COLORS["green"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="BOT threshold (1.25)"),
        Line2D([0], [0], color=COLORS["red"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="SELL threshold (0.75)"),
    ]
    _add_panel_legend(ax1, legend_handles_1, [h.get_label() for h in legend_handles_1], loc="upper left")

    # Panel 2: Lit Buy Ratio + Log Buy Ratio
    ax2 = axes[1]
    lit_buy_ratio = df["lit_buy_ratio"]
    log_buy_ratio = df["log_buy_sell"]

    valid_mask2 = ~lit_buy_ratio.isna()
    if valid_mask2.any():
        _plot_smooth_line(ax2, dates, lit_buy_ratio, COLORS["cyan"], valid_mask2, linewidth=MAIN_LINE_WIDTH)
        ax2.scatter(
            dates,
            lit_buy_ratio,
            c=COLORS["cyan"],
            s=MARKER_SIZE_SMALL,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    ax2b = ax2.twinx()
    _apply_secondary_axis_style(ax2b)
    valid_mask2b = ~log_buy_ratio.isna()
    if valid_mask2b.any():
        _plot_smooth_line(ax2b, dates, log_buy_ratio, COLORS["yellow"], valid_mask2b, linewidth=SECONDARY_LINE_WIDTH)
        ax2b.scatter(
            dates,
            log_buy_ratio,
            c=COLORS["yellow"],
            s=MARKER_SIZE_SMALL,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    _set_abs_ratio_axis(
        ax2,
        lit_buy_ratio,
        neutral_value=NEUTRAL_RATIO,
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        alpha=0.6,
    )
    _set_log_ratio_axis(
        ax2b,
        log_buy_ratio,
        neutral_value=0.0,
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        alpha=0.6,
    )

    ax2.set_ylabel("Lit Buy Ratio", color=COLORS["cyan"], fontsize=10)
    ax2b.set_ylabel("Log Buy Ratio", color=COLORS["yellow"], fontsize=10)
    ax2.set_title("Lit Directional Flow", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    legend_handles_2 = [
        Line2D([0], [0], color=COLORS["cyan"], linewidth=MAIN_LINE_WIDTH, label="Lit Buy Ratio"),
        Line2D([0], [0], color=COLORS["yellow"], linewidth=SECONDARY_LINE_WIDTH, label="Log Buy Ratio"),
        Line2D([0], [0], color=COLORS["neutral"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="Neutral (1.0)"),
        Line2D([0], [0], color=COLORS["neutral"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="Neutral (0.0)"),
    ]
    _add_panel_legend(ax2, legend_handles_2, [h.get_label() for h in legend_handles_2], loc="upper left")

    # Panel 3: OTC Buy/Sell Volumes + Weekly Buy Ratio
    ax3 = axes[2]
    otc_buy = df["otc_buy_volume"]
    otc_sell = df["otc_sell_volume"]
    otc_ratio = df["otc_weekly_buy_ratio"]

    valid_mask3 = ~otc_buy.isna() & ~otc_sell.isna()
    if valid_mask3.any():
        ax3.bar(
            dates[valid_mask3],
            otc_buy[valid_mask3],
            color=COLORS["green"],
            alpha=BAR_ALPHA_PRIMARY,
            width=0.75,
            label="OTC Buy",
            zorder=2,
        )
        ax3.bar(
            dates[valid_mask3],
            otc_sell[valid_mask3],
            bottom=otc_buy[valid_mask3],
            color=COLORS["red"],
            alpha=BAR_ALPHA_SECONDARY,
            width=0.75,
            label="OTC Sell",
            zorder=2,
        )

    ax3.set_ylabel("OTC Vol", color=COLORS["text"], fontsize=10)
    ax3.set_title("OTC Weekly Anchor", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_volume(x)))

    ax3b = ax3.twinx()
    _apply_secondary_axis_style(ax3b)
    valid_mask3b = ~otc_ratio.isna()
    if valid_mask3b.any():
        _plot_smooth_line(ax3b, dates, otc_ratio, COLORS["yellow"], valid_mask3b, linewidth=SECONDARY_LINE_WIDTH)
        ax3b.scatter(
            dates,
            otc_ratio,
            c=COLORS["yellow"],
            s=MARKER_SIZE_SMALL,
            zorder=6,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )
    _set_abs_ratio_axis(
        ax3b,
        otc_ratio,
        neutral_value=NEUTRAL_RATIO,
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        alpha=0.6,
    )
    ax3b.set_ylabel("OTC Buy Ratio", color=COLORS["yellow"], fontsize=10)
    legend_handles_3 = [
        Patch(facecolor=COLORS["green"], edgecolor="none", alpha=BAR_ALPHA_PRIMARY, label="OTC Buy"),
        Patch(facecolor=COLORS["red"], edgecolor="none", alpha=BAR_ALPHA_SECONDARY, label="OTC Sell"),
        Line2D([0], [0], color=COLORS["yellow"], linewidth=SECONDARY_LINE_WIDTH, label="OTC Buy Ratio"),
        Line2D([0], [0], color=COLORS["neutral"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="Neutral (1.0)"),
    ]
    _add_panel_legend(ax3, legend_handles_3, [h.get_label() for h in legend_handles_3], loc="upper right")

    # Decision strip
    ax4 = axes[3]
    ax4.set_facecolor(COLORS["panel_bg"])
    ax4.tick_params(colors=COLORS["text"], labelsize=8)
    decisions = df["pressure_context_label"].fillna("Neutral").tolist()
    decision_colors = []
    for label in decisions:
        if label == "Accumulating":
            decision_colors.append(COLORS["green"])
        elif label == "Distribution":
            decision_colors.append(COLORS["red"])
        else:
            decision_colors.append(COLORS["neutral"])
    ax4.bar(dates, [1] * len(dates), color=decision_colors, width=0.8)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_ylabel("Decision", color=COLORS["text_muted"], fontsize=8)
    ax4.grid(False)
    for spine in ax4.spines.values():
        spine.set_visible(False)
    legend_handles_4 = [
        Patch(facecolor=COLORS["green"], edgecolor="none", label="Accumulating"),
        Patch(facecolor=COLORS["red"], edgecolor="none", label="Distribution"),
        Patch(facecolor=COLORS["neutral"], edgecolor="none", label="Neutral"),
    ]
    _add_panel_legend(ax4, legend_handles_4, [h.get_label() for h in legend_handles_4], loc="upper right")

    # Format x-axis dates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

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
        "Short sale volume is a buy-side proxy. OTC buy/sell is inferred from weekly lit ratios.",
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


def plot_short_only_metrics(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    """Generate a short-only plot using daily short sale data and price context."""
    if df.empty:
        logger.warning("No data to plot for %s", symbol)
        return output_path

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor(COLORS["background"])

    for ax in axes[:3]:
        _apply_primary_axis_style(ax)

    dates = df["date"]

    # Panel 1: Short Ratio (daily)
    ax1 = axes[0]
    short_ratio = df["short_buy_sell_ratio"]
    denom_types = df["short_ratio_denominator_type"]
    colors1 = [_get_denom_color(v) for v in denom_types]

    valid_mask = ~short_ratio.isna()
    if valid_mask.any():
        _plot_smooth_line(ax1, dates, short_ratio, COLORS["cyan"], valid_mask, linewidth=MAIN_LINE_WIDTH)
        ax1.scatter(
            dates,
            short_ratio,
            c=colors1,
            s=MARKER_SIZE,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    _set_abs_ratio_axis(
        ax1,
        short_ratio,
        neutral_value=NEUTRAL_RATIO,
        linestyle="--",
        linewidth=THRESHOLD_LINE_WIDTH,
        alpha=0.6,
    )
    _add_ratio_thresholds(ax1, bot=BOT_THRESHOLD, sell=SELL_THRESHOLD)
    ax1.set_ylabel("Short Sale Buy/Sell Ratio", color=COLORS["text"], fontsize=10)
    ax1.set_title(f"{symbol} - Short Sale Buy/Sell Ratio", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    ax1.text(
        0.99,
        0.95,
        "FINRA_TOTAL",
        transform=ax1.transAxes,
        fontsize=8,
        color=COLORS["cyan"],
        ha="right",
        va="top",
    )
    ax1.text(
        0.99,
        0.85,
        "POLYGON_TOTAL",
        transform=ax1.transAxes,
        fontsize=8,
        color=COLORS["yellow"],
        ha="right",
        va="top",
    )
    legend_handles_1 = [
        Line2D([0], [0], color=COLORS["cyan"], linewidth=MAIN_LINE_WIDTH, label="Short Sale Buy/Sell Ratio"),
        Line2D([0], [0], color=COLORS["neutral"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="Neutral (1.0)"),
        Line2D([0], [0], color=COLORS["green"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="BOT threshold (1.25)"),
        Line2D([0], [0], color=COLORS["red"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="SELL threshold (0.75)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["cyan"], markeredgecolor=COLORS["white"], markersize=4.5, label="Observations"),
    ]
    _add_panel_legend(ax1, legend_handles_1, [h.get_label() for h in legend_handles_1], loc="upper left")

    # Panel 2: Short Sale Volume
    ax2 = axes[1]
    short_vol = df["short_buy_volume"]
    valid_mask2 = ~short_vol.isna()
    if valid_mask2.any():
        ax2.bar(
            dates[valid_mask2],
            short_vol[valid_mask2],
            color=COLORS["yellow"],
            alpha=0.5,
            width=0.75,
        )
    ax2.set_ylabel("Short Vol", color=COLORS["text"], fontsize=10)
    ax2.set_title("Short Sale Volume", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_volume(x)))
    legend_handles_2 = [Patch(facecolor=COLORS["yellow"], edgecolor="none", alpha=0.5, label="Short Sale Vol")]
    _add_panel_legend(ax2, legend_handles_2, [h.get_label() for h in legend_handles_2], loc="upper left")

    # Panel 3: Close Price
    ax3 = axes[2]
    close = df["close"]
    valid_mask3 = ~close.isna()
    if valid_mask3.any():
        ax3.plot(dates[valid_mask3], close[valid_mask3], color=COLORS["green"], linewidth=MAIN_LINE_WIDTH, alpha=0.85)
        ax3.scatter(
            dates[valid_mask3],
            close[valid_mask3],
            color=COLORS["green"],
            s=MARKER_SIZE_SMALL,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )
    ax3.set_ylabel("Close", color=COLORS["text"], fontsize=10)
    ax3.set_title("Close Price", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    legend_handles_3 = [
        Line2D([0], [0], color=COLORS["green"], linewidth=MAIN_LINE_WIDTH, label="Close Price"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor=COLORS["white"], markersize=4.0, label="Observations"),
    ]
    _add_panel_legend(ax3, legend_handles_3, [h.get_label() for h in legend_handles_3], loc="upper left")

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle(
        f"Short Sale Pressure (Daily){title_suffix}",
        color=COLORS["white"],
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5, 0.01,
        "Daily short sale volume is facility-specific; ratio uses FINRA total when available, else Polygon.",
        ha="center",
        fontsize=8,
        color=COLORS["text_muted"],
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

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
    mode: str = "layered",
) -> list[Path]:
    """Render multi-panel plots for each ticker."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    mode = mode.lower()
    valid_modes = {"layered", "short_only", "both"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid plot mode: {mode}. Expected one of {sorted(valid_modes)}.")

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

            if mode in ("layered", "both"):
                output_path = output_dir / f"{symbol.lower()}_metrics_{file_suffix}.png"
                plot_symbol_metrics(df, symbol, output_path)
                output_paths.append(output_path)

            if mode in ("short_only", "both"):
                output_path = output_dir / f"{symbol.lower()}_short_only_{file_suffix}.png"
                plot_short_only_metrics(df, symbol, output_path)
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
    parser.add_argument(
        "--mode",
        type=str,
        default="layered",
        choices=["layered", "short_only", "both"],
        help="Plot mode: layered, short_only, or both (default: layered)",
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
        mode=args.mode,
    )
    print(f"Generated {len(paths)} plot(s)")


if __name__ == "__main__":
    main()
