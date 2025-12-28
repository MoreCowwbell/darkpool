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
            lit_flow_imbalance,
            lit_flow_imbalance_z,
            lit_buy_volume,
            lit_sell_volume,
            lit_total_volume,
            otc_off_exchange_volume,
            otc_buy_volume,
            otc_sell_volume,
            otc_weekly_buy_ratio,
            otc_buy_ratio_z,
            otc_week_used,
            weekly_total_volume,
            otc_participation_rate,
            otc_participation_z,
            otc_participation_delta,
            close,
            return_1d,
            return_z,
            otc_status,
            accumulation_score,
            accumulation_score_display,
            confidence,
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
        figsize=(12, 16.5),  # 1.5x taller
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2]},  # Panel 1 = 1/3, others equal
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

    # Agreement markers: show when short and lit signals agree/disagree
    lit_imbalance = df["lit_flow_imbalance"]
    for i, (d, sr, li) in enumerate(zip(dates, short_ratio, lit_imbalance)):
        if pd.isna(sr) or pd.isna(li):
            continue
        short_bullish = sr > BOT_THRESHOLD
        short_bearish = sr < SELL_THRESHOLD
        lit_bullish = li > 0.1
        lit_bearish = li < -0.1
        if short_bullish and lit_bullish:
            ax1.scatter([d], [sr], c=COLORS["green"], s=80, marker="^", zorder=10, alpha=0.8)
        elif short_bearish and lit_bearish:
            ax1.scatter([d], [sr], c=COLORS["red"], s=80, marker="v", zorder=10, alpha=0.8)
        elif (short_bullish and lit_bearish) or (short_bearish and lit_bullish):
            ax1.scatter([d], [sr], c=COLORS["yellow"], s=60, marker="D", zorder=10, alpha=0.7)

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
    ax1.set_title("Short Sale Buy/Sell Ratio", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    legend_handles_1 = [
        Line2D([0], [0], color=COLORS["cyan"], linewidth=MAIN_LINE_WIDTH, label="Ratio"),
        Line2D([0], [0], color=COLORS["green"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="BOT (1.25)"),
        Line2D([0], [0], color=COLORS["red"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="SELL (0.75)"),
        Line2D([0], [0], linestyle="none", marker="^", color=COLORS["green"], markersize=6, label="Both Bull"),
        Line2D([0], [0], linestyle="none", marker="v", color=COLORS["red"], markersize=6, label="Both Bear"),
        Line2D([0], [0], linestyle="none", marker="D", color=COLORS["yellow"], markersize=5, label="Diverge"),
    ]
    # Horizontal legend at top right
    legend1 = ax1.legend(
        legend_handles_1,
        [h.get_label() for h in legend_handles_1],
        loc="upper right",
        ncol=len(legend_handles_1),
        fontsize=7,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for text in legend1.get_texts():
        text.set_color(COLORS["text"])

    # Panel 2: Lit Flow Imbalance (bounded [-1, +1])
    ax2 = axes[1]
    lit_imbalance_series = df["lit_flow_imbalance"]

    valid_mask2 = ~lit_imbalance_series.isna()
    if valid_mask2.any():
        _plot_smooth_line(ax2, dates, lit_imbalance_series, COLORS["yellow"], valid_mask2, linewidth=MAIN_LINE_WIDTH)
        ax2.scatter(
            dates,
            lit_imbalance_series,
            c=COLORS["yellow"],
            s=MARKER_SIZE_SMALL,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    # Set y-axis bounds: default [-0.5, +0.5], expand if data exceeds
    max_abs = lit_imbalance_series.abs().max(skipna=True) if valid_mask2.any() else 0.3
    if pd.isna(max_abs):
        max_abs = 0.3
    y_bound = max(0.5, float(np.ceil(max_abs * 10) / 10))  # Round up to nearest 0.1
    ax2.set_ylim(-y_bound, y_bound)
    # Dynamic yticks based on range
    if y_bound <= 0.5:
        ax2.set_yticks([-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5])
    else:
        ax2.set_yticks([-y_bound, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, y_bound])
    ax2.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.6)
    ax2.axhline(y=0.1, color=COLORS["green"], linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.5)
    ax2.axhline(y=-0.1, color=COLORS["red"], linestyle="--", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.5)
    ax2.axhline(y=0.2, color=COLORS["green"], linestyle=":", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.4)
    ax2.axhline(y=-0.2, color=COLORS["red"], linestyle=":", linewidth=THRESHOLD_LINE_WIDTH, alpha=0.4)

    ax2.set_ylabel("Lit Flow Imbalance", color=COLORS["yellow"], fontsize=10)
    ax2.set_title("Lit Flow Imbalance (Confirmation)", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    legend_handles_2 = [
        Line2D([0], [0], color=COLORS["yellow"], linewidth=MAIN_LINE_WIDTH, label="Lit Imbalance"),
    ]
    _add_panel_legend(ax2, legend_handles_2, [h.get_label() for h in legend_handles_2], loc="upper right")

    # Panel 3: OTC Participation Rate (weekly step bands)
    ax3 = axes[2]
    otc_part_rate = df["otc_participation_rate"]
    otc_part_z = df["otc_participation_z"]
    otc_part_delta = df["otc_participation_delta"]
    otc_week_used = df["otc_week_used"] if "otc_week_used" in df.columns else None
    otc_status_series = df["otc_status"]

    # Group data by week to draw step bands
    valid_mask3 = ~otc_part_rate.isna()
    if not valid_mask3.any():
        # No Weekly OTC Weekly Report available for this ticker - show placeholder message
        ax3.text(
            0.5, 0.5,
            "No OTC Weekly Report Available for this Ticker",
            ha="center", va="center",
            transform=ax3.transAxes,
            fontsize=10,
            color=COLORS["text_muted"],
            style="italic",
        )
    elif otc_week_used is not None:
        # Get unique weeks and their data
        week_groups = df[valid_mask3].groupby("otc_week_used")
        for week_start, week_data in week_groups:
            if pd.isna(week_start):
                continue
            week_dates = week_data["date"]
            rate = week_data["otc_participation_rate"].iloc[0]
            z_score = week_data["otc_participation_z"].iloc[0] if "otc_participation_z" in week_data.columns else 0
            status = week_data["otc_status"].iloc[0] if "otc_status" in week_data.columns else None

            # Color intensity based on z-score (higher z = darker/more intense)
            if pd.isna(z_score):
                z_score = 0
            # Map z-score to alpha: z=0 -> 0.3, z=2 -> 0.6, z=-2 -> 0.15
            alpha = np.clip(0.3 + z_score * 0.15, 0.1, 0.7)

            # Gray out if stale
            if status == "Staled":
                color = COLORS["neutral"]
                alpha = 0.2
            else:
                # Color based on z-score: positive = cyan (elevated), negative = muted
                if z_score > 0.5:
                    color = COLORS["cyan"]
                elif z_score < -0.5:
                    color = COLORS["text_muted"]
                else:
                    color = COLORS["yellow"]

            # Draw horizontal band for this week
            if len(week_dates) > 0:
                x_start = mdates.date2num(week_dates.iloc[0]) - 0.4
                x_end = mdates.date2num(week_dates.iloc[-1]) + 0.4
                ax3.fill_between(
                    [mdates.num2date(x_start), mdates.num2date(x_end)],
                    [0, 0],
                    [rate, rate],
                    color=color,
                    alpha=alpha,
                    zorder=2,
                )
                # Add week label
                mid_x = mdates.num2date((x_start + x_end) / 2)
                week_label = pd.to_datetime(week_start).strftime("%m-%d")
                label_text = f"Wk {week_label}"
                if status == "Staled":
                    label_text += " (STALE)"
                ax3.text(
                    mid_x,
                    rate + 0.02,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=COLORS["text_muted"],
                    zorder=5,
                )

    # Also show participation rate as line for trend visibility
    if valid_mask3.any():
        ax3.plot(
            dates[valid_mask3],
            otc_part_rate[valid_mask3],
            color=COLORS["yellow"],
            linewidth=SECONDARY_LINE_WIDTH,
            alpha=0.8,
            zorder=4,
        )
        ax3.scatter(
            dates[valid_mask3],
            otc_part_rate[valid_mask3],
            c=COLORS["yellow"],
            s=MARKER_SIZE_SMALL,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    # Set y-axis for participation rate (0 to ~0.6 typical)
    max_rate = otc_part_rate.max(skipna=True) if valid_mask3.any() else 0.4
    if pd.isna(max_rate) or max_rate < 0.2:
        max_rate = 0.4
    y_upper = min(float(np.ceil(max_rate * 10) / 10 + 0.1), 1.0)
    ax3.set_ylim(0, y_upper)
    ax3.set_yticks(np.arange(0, y_upper + 0.05, 0.1))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))

    ax3.set_ylabel("OTC Participation", color=COLORS["text"], fontsize=10)
    ax3.set_title("OTC Participation (proxy)", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")

    # Delta indicator on secondary axis
    ax3b = ax3.twinx()
    _apply_secondary_axis_style(ax3b)
    valid_delta = ~otc_part_delta.isna() if otc_part_delta is not None else pd.Series([False] * len(dates))
    if valid_delta.any():
        delta_colors = [COLORS["green"] if d > 0 else COLORS["red"] for d in otc_part_delta[valid_delta]]
        ax3b.bar(
            dates[valid_delta],
            otc_part_delta[valid_delta],
            color=delta_colors,
            alpha=0.4,
            width=0.3,
            zorder=1,
        )
        max_delta = otc_part_delta.abs().max(skipna=True)
        if pd.isna(max_delta) or max_delta < 0.05:
            max_delta = 0.05
        ax3b.set_ylim(-max_delta * 1.5, max_delta * 1.5)
    ax3b.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.4)
    ax3b.set_ylabel("WoW Delta", color=COLORS["text_muted"], fontsize=8)

    legend_handles_3 = [
        Patch(facecolor=COLORS["cyan"], edgecolor="none", alpha=0.4, label="High (z>0.5)"),
        Patch(facecolor=COLORS["yellow"], edgecolor="none", alpha=0.3, label="Normal"),
        Patch(facecolor=COLORS["neutral"], edgecolor="none", alpha=0.2, label="Stale/Low"),
        Line2D([0], [0], color=COLORS["yellow"], linewidth=SECONDARY_LINE_WIDTH, label="Rate"),
    ]
    # Horizontal legend at top right
    legend3 = ax3.legend(
        legend_handles_3,
        [h.get_label() for h in legend_handles_3],
        loc="upper right",
        ncol=len(legend_handles_3),
        fontsize=7,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for text in legend3.get_texts():
        text.set_color(COLORS["text"])

    # Panel 4: Accumulation Score + Confidence Bar
    ax4 = axes[3]
    ax4.set_facecolor(COLORS["panel_bg"])
    ax4.tick_params(colors=COLORS["text"], labelsize=8)

    # Get score and confidence data
    score_display = df["accumulation_score_display"].fillna(50)  # Default to neutral (50)
    confidence = df["confidence"].fillna(0.5)

    # Create RdYlGn-like gradient: Red (0) -> Gray (50) -> Green (100)
    from matplotlib.colors import LinearSegmentedColormap

    cmap_colors = [
        (0.0, COLORS["red"]),      # 0 = Distribution
        (0.5, "#888888"),          # 50 = Neutral (gray)
        (1.0, COLORS["green"]),    # 100 = Accumulating
    ]
    score_cmap = LinearSegmentedColormap.from_list(
        "score_cmap",
        [(pos, color) for pos, color in cmap_colors]
    )

    # Draw score bars with gradient color
    for i, (d, score, conf) in enumerate(zip(dates, score_display, confidence)):
        if pd.isna(score):
            score = 50
        if pd.isna(conf):
            conf = 0.5

        # Normalize score to [0, 1] for colormap
        norm_score = np.clip(score / 100.0, 0, 1)
        bar_color = score_cmap(norm_score)

        # Reduce opacity if low confidence
        alpha = 0.8 if conf >= 0.6 else 0.4

        # Main score bar - height proportional to score (0-100 -> 0.0-1.0)
        bar_height = score / 100.0  # Normalize to 0-1 range
        ax4.bar(d, bar_height, bottom=0, color=bar_color, alpha=alpha, width=0.8, zorder=2)

        # Thin confidence bar below (shows confidence level)
        conf_height = 0.08 * conf  # Scale confidence to bar height
        ax4.bar(d, conf_height, bottom=-0.12, color=COLORS["white"], alpha=0.5, width=0.6, zorder=3)

        # Score label on bar (position at bar top or center, whichever is visible)
        if not pd.isna(score):
            label_y = max(bar_height / 2, 0.15)  # Ensure label is visible even for low scores
            ax4.text(
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

    # Threshold reference lines at 30%, 50%, 70%
    ax4.axhline(y=0.30, color=COLORS["red"], linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    ax4.axhline(y=0.50, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    ax4.axhline(y=0.70, color=COLORS["green"], linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)

    # Y-axis: extend below 0 to show confidence bar, main range is 0-1
    ax4.set_ylim(-0.15, 1.05)
    ax4.set_yticks([0, 0.3, 0.5, 0.7, 1.0])
    ax4.set_yticklabels(["0", "30", "50", "70", "100"])

    ax4.set_ylabel("Score", color=COLORS["text_muted"], fontsize=8)
    ax4.set_title("Accumulation Score (0-100)", color=COLORS["white"], fontsize=9, fontweight="bold", loc="left")
    ax4.grid(False)
    for spine in ax4.spines.values():
        spine.set_visible(False)

    legend_handles_4 = [
        Patch(facecolor=COLORS["green"], edgecolor="none", alpha=0.8, label=">70 Accum"),
        Patch(facecolor="#888888", edgecolor="none", alpha=0.8, label="30-70 Neutral"),
        Patch(facecolor=COLORS["red"], edgecolor="none", alpha=0.8, label="<30 Distrib"),
        Patch(facecolor=COLORS["white"], edgecolor="none", alpha=0.5, label="Confidence"),
    ]
    # Horizontal legend at top right
    legend4 = ax4.legend(
        legend_handles_4,
        [h.get_label() for h in legend_handles_4],
        loc="upper right",
        ncol=len(legend_handles_4),
        fontsize=7,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for text in legend4.get_texts():
        text.set_color(COLORS["text"])

    # Format x-axis dates on ALL panels (not just bottom)
    for ax in [axes[0], axes[1], axes[2], ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Show every day
        ax.tick_params(axis="x", labelbottom=True)  # Enable x labels on all panels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    # Title and footer
    fig.suptitle(
        f"{symbol} - Institutional Pressure Analysis{title_suffix}",
        color=COLORS["white"],
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Footer definitions - left-justified at bottom left, terms in cyan
    footer_y_start = 0.045
    footer_line_height = 0.012
    footer_definitions = [
        ("Short Sale Ratio", "Buy pressure proxy from FINRA short sale data (>1.25 = buying, <0.75 = selling)"),
        ("Lit Imbalance", "Net buying vs selling on lit exchanges (positive = buyers dominating)"),
        ("OTC Participation", "Off-exchange volume share - measures institutional dark pool activity"),
        ("Accumulation Score", "Combined signal (0-100) weighting short, lit, and price momentum"),
    ]
    for i, (term, definition) in enumerate(footer_definitions):
        y_pos = footer_y_start - (i * footer_line_height)
        # Term in cyan
        fig.text(0.02, y_pos, f"{term}:", ha="left", va="top", fontsize=7, color=COLORS["cyan"], fontweight="bold")
        # Definition in muted text, offset to the right
        fig.text(0.14, y_pos, definition, ha="left", va="top", fontsize=7, color=COLORS["text_muted"])

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])  # More bottom space for multi-line footer

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

    # Format x-axis dates on ALL panels (not just bottom)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Show every day
        ax.tick_params(axis="x", labelbottom=True)  # Enable x labels on all panels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

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
