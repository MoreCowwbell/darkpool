"""
Multi-panel dark-theme plotter for daily metrics.

Generates PNG visualizations from DuckDB daily_metrics data:
- Layered: configurable panel 1 metric, short sale buy/sell ratio, lit flow imbalance,
  OTC participation, accumulation score
- Short-only: short ratio, short sale volume, close price
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LinearSegmentedColormap
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
    "orange": "#ff8c00",
}

MAIN_LINE_WIDTH = 2.3
SECONDARY_LINE_WIDTH = 1.8
THRESHOLD_LINE_WIDTH = 1.0
CENTERLINE_WIDTH = 1.4
GRID_ALPHA = 0.18
MARKER_SIZE = 28
MARKER_SIZE_SMALL = 22
YLABEL_COLOR = COLORS["white"]
YLABEL_SIZE = 10
PANEL1_LINE_WIDTH = 1.7
AGREEMENT_MARKER_SIZE = 120
AGREEMENT_MARKER_SIZE_SMALL = 70
AGREEMENT_MARKER_EDGE = 0.6
BAR_ALPHA_PRIMARY = 0.35
BAR_ALPHA_SECONDARY = 0.25
BOT_THRESHOLD = 1.25
SELL_THRESHOLD = 0.75
NEUTRAL_RATIO = 1.0
DEFAULT_PANEL1_METRIC = "vw_flow"
PANEL1_METRICS = {
    "vw_flow": {
        "label": "Volume Weighted Directional Flow",
        "legend": "VW Flow",
        "axis": "flow",
        "definition": "Net buy minus sell volume across short sale + lit markets",
    },
    "combined_ratio": {
        "label": "Combined Buy/Sell Ratio",
        "legend": "Combined Ratio",
        "axis": "ratio",
        "definition": "Ratio of (short + lit) buy volume to (short + lit) sell volume",
    },
    "finra_buy_volume": {
        "label": "FINRA Buy Volume (B)",
        "legend": "FINRA Buy Vol",
        "axis": "volume",
        "definition": "FINRA daily short sale buy volume proxy (B)",
    },
}


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


def _compute_fig_width(day_count: int) -> float:
    if day_count <= 0:
        return 12.0
    buckets = int(np.ceil(day_count / 50))
    return 12.0 * max(1, buckets)


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


def _set_flow_axis(
    ax,
    values: pd.Series,
    padding_ratio: float = 0.1,
) -> Line2D:
    series = pd.to_numeric(values, errors="coerce")
    max_abs = series.abs().max(skipna=True)
    if pd.isna(max_abs) or max_abs == 0:
        max_abs = 1.0
    padding = max(max_abs * padding_ratio, 1.0)
    y_max = max_abs + padding
    y_min = -y_max
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.linspace(y_min, y_max, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_volume(x)))
    return ax.axhline(
        y=0.0,
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=CENTERLINE_WIDTH,
        alpha=0.7,
        zorder=1,
    )


def _set_volume_axis(
    ax,
    values: pd.Series,
    padding_ratio: float = 0.1,
) -> None:
    series = pd.to_numeric(values, errors="coerce")
    max_val = series.max(skipna=True)
    if pd.isna(max_val) or max_val <= 0:
        max_val = 1.0
    y_max = max_val * (1 + padding_ratio)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_volume(x)))


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


def _resolve_panel1_metric(value: str | None) -> str:
    if not value:
        return DEFAULT_PANEL1_METRIC
    normalized = value.strip().lower()
    if normalized in PANEL1_METRICS:
        return normalized
    logger.warning("Unknown panel1_metric '%s'; defaulting to %s.", value, DEFAULT_PANEL1_METRIC)
    return DEFAULT_PANEL1_METRIC


def _compute_combined_ratio(df: pd.DataFrame) -> pd.Series:
    short_buy = pd.to_numeric(df.get("short_buy_volume"), errors="coerce")
    short_sell = pd.to_numeric(df.get("short_sell_volume"), errors="coerce")
    lit_buy = pd.to_numeric(df.get("lit_buy_volume"), errors="coerce")
    lit_sell = pd.to_numeric(df.get("lit_sell_volume"), errors="coerce")
    total_buy = short_buy.fillna(0.0) + lit_buy.fillna(0.0)
    total_sell = short_sell.fillna(0.0) + lit_sell.fillna(0.0)
    ratio = pd.Series(np.nan, index=df.index)
    valid = total_sell > 0
    ratio.loc[valid] = total_buy.loc[valid] / total_sell.loc[valid]
    return ratio


def _compute_vw_flow(df: pd.DataFrame) -> pd.Series:
    short_buy = pd.to_numeric(df.get("short_buy_volume"), errors="coerce")
    short_sell = pd.to_numeric(df.get("short_sell_volume"), errors="coerce")
    lit_buy = pd.to_numeric(df.get("lit_buy_volume"), errors="coerce")
    lit_sell = pd.to_numeric(df.get("lit_sell_volume"), errors="coerce")
    total_buy = short_buy.fillna(0.0) + lit_buy.fillna(0.0)
    total_sell = short_sell.fillna(0.0) + lit_sell.fillna(0.0)
    flow = pd.Series(np.nan, index=df.index)
    has_flow = short_buy.notna() | short_sell.notna() | lit_buy.notna() | lit_sell.notna()
    flow.loc[has_flow] = total_buy.loc[has_flow] - total_sell.loc[has_flow]
    return flow
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
            combined_ratio,
            vw_flow,
            finra_buy_volume,
            vwbr,
            vwbr_z,
            short_ratio_denominator_type,
            short_ratio_denominator_value,
            short_buy_volume,
            short_sell_volume,
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
    x_values,
    values,
    color,
    valid_mask,
    linewidth: float = MAIN_LINE_WIDTH,
    alpha: float = 0.85,
    zorder: int = 3,
):
    """Plot a smooth PCHIP-interpolated line through valid data points."""
    if valid_mask.sum() >= 3:
        valid_x = x_values[valid_mask]
        valid_values = values[valid_mask]

        if pd.api.types.is_datetime64_any_dtype(valid_x):
            x_nums = mdates.date2num(pd.to_datetime(valid_x))
            x_smooth = np.linspace(x_nums.min(), x_nums.max(), 150)
            interp = PchipInterpolator(x_nums, valid_values.to_numpy())
            y_smooth = interp(x_smooth)
            x_plot = mdates.num2date(x_smooth)
        else:
            x_nums = np.asarray(valid_x, dtype=float)
            x_smooth = np.linspace(x_nums.min(), x_nums.max(), 150)
            interp = PchipInterpolator(x_nums, valid_values.to_numpy())
            y_smooth = interp(x_smooth)
            x_plot = x_smooth

        ax.plot(
            x_plot,
            y_smooth,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        ax.plot(
            x_values[valid_mask],
            values[valid_mask],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )


def _build_plot_x(df: pd.DataFrame, plot_trading_gaps: bool) -> tuple[pd.Series, list[str] | None]:
    """Return x-values and labels for plotting with or without trading-day gaps."""
    dates = df["date"]
    if plot_trading_gaps:
        return dates, None
    x_values = pd.Series(np.arange(len(dates)), index=dates.index)
    labels = pd.to_datetime(dates).dt.strftime("%y-%m-%d").tolist()
    return x_values, labels


def plot_symbol_metrics(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path,
    title_suffix: str = "",
    plot_trading_gaps: bool = True,
    panel1_metric: str | None = None,
) -> Path:
    """
    Generate a multi-panel plot plus footer for a single symbol.

    Panel 1: Configurable metric (VW Flow, combined ratio, or FINRA buy volume)
    Panel 2: Short Sale Buy/Sell Ratio
    Panel 3: Lit Flow Imbalance
    Panel 4: OTC Participation Rate
    Panel 5: Accumulation Score
    """
    if df.empty:
        logger.warning("No data to plot for %s", symbol)
        return output_path

    day_count = df["date"].nunique()
    fig_width = _compute_fig_width(day_count)

    # Set up dark theme
    plt.style.use("dark_background")

    # Use GridSpec for explicit control over panel spacing
    fig = plt.figure(figsize=(fig_width, 21))
    fig.patch.set_facecolor(COLORS["background"])

    # Create GridSpec with fixed spacing to prevent layout shifts
    gs = GridSpec(
        6, 1,
        figure=fig,
        height_ratios=[3, 3, 2, 2, 2, 1],
        hspace=0.35,  # Fixed vertical spacing between panels
        left=0.08,    # Fixed left margin
        right=0.92,   # Fixed right margin
        top=0.94,     # Fixed top margin (room for suptitle)
        bottom=0.04,  # Fixed bottom margin
    )

    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    for ax in axes[:5]:  # Only style the data panels
        _apply_primary_axis_style(ax)

    dates = df["date"]
    x_values, x_labels = _build_plot_x(df, plot_trading_gaps)

    # Panel 1: Configurable metric
    ax0 = axes[0]
    panel1_key = _resolve_panel1_metric(panel1_metric)
    panel1_meta = PANEL1_METRICS[panel1_key]

    if panel1_key == "vw_flow":
        panel1_series = df["vw_flow"] if "vw_flow" in df.columns else df["vwbr"]
        if panel1_series.isna().all():
            panel1_series = _compute_vw_flow(df)
    elif panel1_key == "combined_ratio":
        panel1_series = df["combined_ratio"] if "combined_ratio" in df.columns else _compute_combined_ratio(df)
        if panel1_series.isna().all():
            panel1_series = _compute_combined_ratio(df)
    else:
        panel1_series = df["finra_buy_volume"] if "finra_buy_volume" in df.columns else df["short_buy_volume"]
        if panel1_series.isna().all():
            panel1_series = df["short_buy_volume"]

    panel1_series = pd.to_numeric(panel1_series, errors="coerce")
    valid_mask0 = ~panel1_series.isna()
    if valid_mask0.any():
        _plot_smooth_line(ax0, x_values, panel1_series, COLORS["cyan"], valid_mask0, linewidth=PANEL1_LINE_WIDTH)
        ax0.scatter(
            x_values[valid_mask0],
            panel1_series[valid_mask0],
            c=COLORS["cyan"],
            s=MARKER_SIZE,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    if panel1_meta["axis"] == "flow":
        _set_flow_axis(ax0, panel1_series)
    elif panel1_meta["axis"] == "ratio":
        _set_abs_ratio_axis(
            ax0,
            panel1_series,
            neutral_value=NEUTRAL_RATIO,
            linestyle="--",
            linewidth=THRESHOLD_LINE_WIDTH,
            alpha=0.6,
        )
        _add_ratio_thresholds(ax0, bot=BOT_THRESHOLD, sell=SELL_THRESHOLD)
    else:
        _set_volume_axis(ax0, panel1_series)

    ax0.set_ylabel(panel1_meta["label"], color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax0.set_title(panel1_meta["label"], color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")

    # Add mean line for Panel 1
    panel1_mean = panel1_series.mean(skipna=True)
    if not pd.isna(panel1_mean):
        ax0.axhline(
            y=panel1_mean,
            color=COLORS["orange"],
            linestyle="--",
            linewidth=THRESHOLD_LINE_WIDTH,
            alpha=0.8,
            zorder=2,
        )

    legend_handles_0 = [
        Line2D([0], [0], color=COLORS["cyan"], linewidth=MAIN_LINE_WIDTH, label=panel1_meta["legend"]),
        Line2D([0], [0], color=COLORS["orange"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label=f"Mean ({panel1_mean:.2f})" if not pd.isna(panel1_mean) else "Mean"),
    ]
    legend0 = ax0.legend(
        legend_handles_0,
        [h.get_label() for h in legend_handles_0],
        loc="upper right",
        ncol=len(legend_handles_0),
        fontsize=7,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for text in legend0.get_texts():
        text.set_color(COLORS["text"])

    # Panel 2: Short Sale Buy Ratio
    ax1 = axes[1]
    short_ratio = df["short_buy_sell_ratio"]

    valid_mask = ~short_ratio.isna()
    if valid_mask.any():
        _plot_smooth_line(ax1, x_values, short_ratio, COLORS["cyan"], valid_mask, linewidth=PANEL1_LINE_WIDTH)
        ax1.scatter(
            x_values[valid_mask],
            short_ratio[valid_mask],
            c=COLORS["cyan"],
            s=MARKER_SIZE,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    # Agreement markers: show when short and lit signals agree/disagree
    lit_imbalance = df["lit_flow_imbalance"]
    for d, sr, li in zip(x_values, short_ratio, lit_imbalance):
        if pd.isna(sr) or pd.isna(li):
            continue
        short_bullish = sr > BOT_THRESHOLD
        short_bearish = sr < SELL_THRESHOLD
        lit_bullish = li > 0.1
        lit_bearish = li < -0.1
        if short_bullish and lit_bullish:
            ax1.scatter(
                [d],
                [sr],
                c=COLORS["green"],
                s=AGREEMENT_MARKER_SIZE,
                marker="^",
                zorder=10,
                alpha=0.85,
                edgecolors=COLORS["white"],
                linewidths=AGREEMENT_MARKER_EDGE,
            )
        elif short_bearish and lit_bearish:
            ax1.scatter(
                [d],
                [sr],
                c=COLORS["red"],
                s=AGREEMENT_MARKER_SIZE,
                marker="v",
                zorder=10,
                alpha=0.85,
                edgecolors=COLORS["white"],
                linewidths=AGREEMENT_MARKER_EDGE,
            )
        elif (short_bullish and lit_bearish) or (short_bearish and lit_bullish):
            ax1.scatter(
                [d],
                [sr],
                c=COLORS["yellow"],
                s=AGREEMENT_MARKER_SIZE,
                marker="D",
                zorder=10,
                alpha=0.75,
                edgecolors=COLORS["white"],
                linewidths=AGREEMENT_MARKER_EDGE,
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

    # Add mean line for Panel 2
    short_ratio_mean = short_ratio.mean(skipna=True)
    if not pd.isna(short_ratio_mean):
        ax1.axhline(
            y=short_ratio_mean,
            color=COLORS["orange"],
            linestyle="--",
            linewidth=THRESHOLD_LINE_WIDTH,
            alpha=0.8,
            zorder=2,
        )

    ax1.set_ylabel("Short Sale Buy/Sell Ratio", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax1.set_title("Short Sale Buy/Sell Ratio", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    legend_handles_1 = [
        Line2D([0], [0], color=COLORS["cyan"], linewidth=MAIN_LINE_WIDTH, label="Ratio"),
        Line2D([0], [0], color=COLORS["orange"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label=f"Mean ({short_ratio_mean:.2f})" if not pd.isna(short_ratio_mean) else "Mean"),
        Line2D([0], [0], color=COLORS["green"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="BOT (1.25)"),
        Line2D([0], [0], color=COLORS["red"], linewidth=THRESHOLD_LINE_WIDTH, linestyle="--", label="SELL (0.75)"),
        Line2D([0], [0], linestyle="none", marker="^", color=COLORS["green"], markersize=6, label="Short+Lit Bull"),
        Line2D([0], [0], linestyle="none", marker="v", color=COLORS["red"], markersize=6, label="Short+Lit Bear"),
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

    # Panel 3: Lit Flow Imbalance (bounded [-1, +1])
    ax2 = axes[2]
    lit_imbalance_series = df["lit_flow_imbalance"]

    valid_mask2 = ~lit_imbalance_series.isna()
    if valid_mask2.any():
        _plot_smooth_line(ax2, x_values, lit_imbalance_series, COLORS["yellow"], valid_mask2, linewidth=MAIN_LINE_WIDTH)
        ax2.scatter(
            x_values[valid_mask2],
            lit_imbalance_series[valid_mask2],
            c=COLORS["yellow"],
            s=MARKER_SIZE_SMALL,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    for d, sr, li in zip(x_values, short_ratio, lit_imbalance_series):
        if pd.isna(sr) or pd.isna(li):
            continue
        short_bullish = sr > BOT_THRESHOLD
        short_bearish = sr < SELL_THRESHOLD
        lit_bullish = li > 0.1
        lit_bearish = li < -0.1
        if short_bullish and lit_bullish:
            ax2.scatter(
                [d],
                [li],
                c=COLORS["green"],
                s=AGREEMENT_MARKER_SIZE_SMALL,
                marker="^",
                zorder=6,
                alpha=0.85,
                edgecolors=COLORS["white"],
                linewidths=AGREEMENT_MARKER_EDGE,
            )
        elif short_bearish and lit_bearish:
            ax2.scatter(
                [d],
                [li],
                c=COLORS["red"],
                s=AGREEMENT_MARKER_SIZE_SMALL,
                marker="v",
                zorder=6,
                alpha=0.85,
                edgecolors=COLORS["white"],
                linewidths=AGREEMENT_MARKER_EDGE,
            )
        elif (short_bullish and lit_bearish) or (short_bearish and lit_bullish):
            ax2.scatter(
                [d],
                [li],
                c=COLORS["cyan"],
                s=AGREEMENT_MARKER_SIZE_SMALL,
                marker="D",
                zorder=6,
                alpha=0.8,
                edgecolors=COLORS["white"],
                linewidths=AGREEMENT_MARKER_EDGE,
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

    ax2.set_ylabel("Lit Flow Imbalance", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax2.set_title("Lit Flow Imbalance (Confirmation)", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")
    legend_handles_2 = [
        Line2D([0], [0], color=COLORS["yellow"], linewidth=MAIN_LINE_WIDTH, label="Lit Imbalance"),
        Line2D([0], [0], linestyle="none", marker="^", color=COLORS["green"], markersize=6, label="Short+Lit Bull"),
        Line2D([0], [0], linestyle="none", marker="v", color=COLORS["red"], markersize=6, label="Short+Lit Bear"),
        Line2D([0], [0], linestyle="none", marker="D", color=COLORS["cyan"], markersize=6, label="Diverge"),
    ]
    _add_panel_legend(ax2, legend_handles_2, [h.get_label() for h in legend_handles_2], loc="upper right")

    # Panel 4: OTC Participation Rate (weekly step bands)
    ax3 = axes[3]
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
            week_x = x_values.loc[week_data.index]
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
                if plot_trading_gaps:
                    x_start = mdates.date2num(week_dates.iloc[0]) - 0.4
                    x_end = mdates.date2num(week_dates.iloc[-1]) + 0.4
                    x_span = [mdates.num2date(x_start), mdates.num2date(x_end)]
                    mid_x = mdates.num2date((x_start + x_end) / 2)
                else:
                    x_start = float(week_x.iloc[0]) - 0.4
                    x_end = float(week_x.iloc[-1]) + 0.4
                    x_span = [x_start, x_end]
                    mid_x = (x_start + x_end) / 2

                ax3.fill_between(
                    x_span,
                    [0, 0],
                    [rate, rate],
                    color=color,
                    alpha=alpha,
                    zorder=2,
                )
                # Add week label
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
            x_values[valid_mask3],
            otc_part_rate[valid_mask3],
            color=COLORS["yellow"],
            linewidth=SECONDARY_LINE_WIDTH,
            alpha=0.8,
            zorder=4,
        )
        ax3.scatter(
            x_values[valid_mask3],
            otc_part_rate[valid_mask3],
            c=COLORS["yellow"],
            s=MARKER_SIZE_SMALL,
            zorder=5,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    # Set y-axis for participation rate (0 to 1.0, typical range 20-60%)
    # Formula: OTC / (OTC + Lit) guarantees values in [0, 1] range
    max_rate = otc_part_rate.max(skipna=True) if valid_mask3.any() else 0.4
    if pd.isna(max_rate) or max_rate < 0.2:
        max_rate = 0.4
    # Cap at 1.0 (100%) since participation rate can't exceed this
    # Add 10% headroom for visual clarity
    y_upper = min(float(np.ceil(max_rate * 10) / 10 + 0.1), 1.0)
    ax3.set_ylim(0, y_upper)
    ax3.set_yticks(np.arange(0, y_upper + 0.05, 0.1))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))

    ax3.set_ylabel("OTC Participation", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax3.set_title("OTC Participation Rate - Weekly", color=COLORS["white"], fontsize=11, fontweight="bold", loc="left")

    # Delta indicator on secondary axis
    ax3b = ax3.twinx()
    _apply_secondary_axis_style(ax3b)
    valid_delta = ~otc_part_delta.isna() if otc_part_delta is not None else pd.Series([False] * len(dates))
    if valid_delta.any():
        delta_colors = [COLORS["green"] if d > 0 else COLORS["red"] for d in otc_part_delta[valid_delta]]
        ax3b.bar(
            x_values[valid_delta],
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
    ax3b.set_ylabel("WoW Delta (pp)", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)

    legend_handles_3 = [
        Patch(facecolor=COLORS["cyan"], edgecolor="none", alpha=0.4, label="High (z>0.5)"),
        Patch(facecolor=COLORS["yellow"], edgecolor="none", alpha=0.3, label="Normal"),
        Patch(facecolor=COLORS["neutral"], edgecolor="none", alpha=0.2, label="Stale/Low"),
        Line2D([0], [0], color=COLORS["yellow"], linewidth=SECONDARY_LINE_WIDTH, label="OTC Participation Rate - Weekly"),
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

    # Panel 5: Accumulation Score + Confidence Bar
    ax4 = axes[4]
    ax4.set_facecolor(COLORS["panel_bg"])
    ax4.tick_params(colors=COLORS["text"], labelsize=8)

    # Get score and confidence data
    score_display = df["accumulation_score_display"].fillna(50)  # Default to neutral (50)
    confidence = df["confidence"].fillna(0.5)

    # Create RdYlGn-like gradient: Red (0) -> Gray (50) -> Green (100)
    cmap_colors = [
        (0.0, "#b026ff"),      # 0 = Distribution
        (0.5, "#555555"),      # 50 = Neutral (gray)
        (1.0, "#00ff88"),      # 100 = Accumulating
    ]
    score_cmap = LinearSegmentedColormap.from_list(
        "score_cmap",
        [(pos, color) for pos, color in cmap_colors]
    )

    # Draw score bars with gradient color
    for i, (xv, score, conf) in enumerate(zip(x_values, score_display, confidence)):
        if pd.isna(score):
            score = 50
        if pd.isna(conf):
            conf = 0.5

        # Normalize score to [0, 1] for colormap
        norm_score = np.clip(score / 100.0, 0, 1)
        dev = norm_score - 0.5
        norm_score = 0.5 + np.sign(dev) * (abs(dev) ** 0.85)
        norm_score = np.clip(norm_score, 0, 1)
        bar_color = score_cmap(norm_score)

        # Reduce opacity if low confidence
        alpha = 0.8 if conf >= 0.6 else 0.4

        # Main score bar - height proportional to score (0-100 -> 0.0-1.0)
        bar_height = score / 100.0  # Normalize to 0-1 range
        ax4.bar(xv, bar_height, bottom=0, color=bar_color, alpha=alpha, width=0.8, zorder=2)

        # Thin confidence bar below (shows confidence level) - colored by confidence
        conf_height = 0.08 * conf  # Scale confidence to bar height
        if conf >= 0.7:
            conf_color = COLORS["green"]
        elif conf >= 0.4:
            conf_color = COLORS["yellow"]
        else:
            conf_color = COLORS["red"]
        ax4.bar(xv, conf_height, bottom=-0.12, color=conf_color, alpha=0.6, width=0.6, zorder=3)

        # Score label on bar (position at bar top or center, whichever is visible)
        if not pd.isna(score):
            label_y = max(bar_height / 2, 0.15)  # Ensure label is visible even for low scores
            ax4.text(
                xv,
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

    ax4.set_ylabel("Score", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax4.set_title("Accumulation Score (0-100)", color=COLORS["white"], fontsize=9, fontweight="bold", loc="left")
    ax4.grid(False)
    for spine in ax4.spines.values():
        spine.set_visible(False)

    legend_handles_4 = [
        Patch(facecolor="#00ff88", edgecolor="none", alpha=0.8, label=">70 Accum"),
        Patch(facecolor="#555555", edgecolor="none", alpha=0.8, label="30-70 Neutral"),
        Patch(facecolor="#b026ff", edgecolor="none", alpha=0.8, label="<30 Distrib"),
        Line2D([0], [0], color=COLORS["green"], linewidth=4, alpha=0.6, label="High Conf"),
        Line2D([0], [0], color=COLORS["yellow"], linewidth=4, alpha=0.6, label="Med Conf"),
        Line2D([0], [0], color=COLORS["red"], linewidth=4, alpha=0.6, label="Low Conf"),
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

    if plot_trading_gaps:
        x_min = dates.min() - timedelta(hours=12)
        x_max = dates.max() + timedelta(hours=12)
        x_locator = mdates.DayLocator(interval=1)
        x_formatter = mdates.DateFormatter("%y-%m-%d")

        # Format x-axis dates on ALL panels (not just bottom)
        for ax in [axes[0], axes[1], axes[2], axes[3], ax4]:
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(x_formatter)
            ax.xaxis.set_major_locator(x_locator)  # Show every day
            ax.tick_params(axis="x", labelbottom=True)  # Enable x labels on all panels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    else:
        x_min = -0.5
        x_max = len(dates) - 0.5
        x_ticks = x_values.tolist()
        labels = x_labels or ["" for _ in x_ticks]
        for ax in [axes[0], axes[1], axes[2], axes[3], ax4]:
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.tick_params(axis="x", labelbottom=True)

    ax3b.set_xlim(x_min, x_max)

    # Title and footer
    fig.suptitle(
        f"{symbol} - Institutional Pressure Analysis{title_suffix}",
        color=COLORS["white"],
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 6: Footer with definitions and table (as proper subplot)
    ax_footer = axes[5]
    ax_footer.set_facecolor(COLORS["background"])
    ax_footer.axis("off")  # Hide all axis elements

    # Footer layout using axis coordinates (0-1)
    footer_y_start = 0.95
    footer_line_height = 0.16

    # Left side: Definitions
    footer_definitions = [
        (panel1_meta["label"], panel1_meta["definition"]),
        ("Short Sale Ratio", "Institutional buy/sell pressure proxy from FINRA 'Daily Short-Sale Volume'"),
        ("Lit Imbalance", "Net buying vs selling on lit exchanges (positive = buyers dominate)"),
        ("OTC Participation", "Institutional dark-pool activity proxy from FINRA 'Weekly Over-The-Counter(OTC) Report'"),
        ("Accumulation Score", "Composite Signal 0-100 weighting short-sale pressure, lit imbalance, and price momentum"),
    ]

    for i, (term, definition) in enumerate(footer_definitions):
        y_pos = footer_y_start - (i * footer_line_height)
        ax_footer.text(0.0, y_pos, f"{term}:", transform=ax_footer.transAxes,
                       ha="left", va="top", fontsize=7, color=COLORS["cyan"], fontweight="bold")
        ax_footer.text(0.1, y_pos, definition, transform=ax_footer.transAxes,
                       ha="left", va="top", fontsize=7, color=COLORS["text_muted"])

    # Right side: Accumulation Score Components table
    table_x_start = 0.6

    # Table title
    ax_footer.text(table_x_start, footer_y_start, "Accumulation Score Components:",
                   transform=ax_footer.transAxes, ha="left", va="top",
                   fontsize=7, color=COLORS["cyan"], fontweight="bold")

    # Table data with tighter column spacing
    score_inputs_table = [
        ("Input", "Weight", "Source", "What it measures"),  # Header
        ("short_buy_sell_ratio_z / vwbr_z", "55%", "FINRA + Polygon", "Configurable short signal z-score (flow z-score)"),
        ("lit_flow_imbalance_z", "30%", "Polygon Trades", "Z-score of lit imbalance"),
        ("return_z", "15%", "Polygon Daily Agg", "Z-score of price momentum"),
        ("otc_participation_z", "Mult.", "FINRA OTC Weekly", "Modulates score intensity"),
    ]

    # Column x positions (tighter spacing)
    col_x = [table_x_start, table_x_start + 0.12, table_x_start + 0.17, table_x_start + 0.28]

    for row_idx, row in enumerate(score_inputs_table):
        y_pos = footer_y_start - ((row_idx + 1) * footer_line_height)
        for col_idx, cell in enumerate(row):
            color = COLORS["cyan"] if row_idx == 0 else COLORS["text_muted"]
            weight = "bold" if row_idx == 0 else "normal"
            ax_footer.text(col_x[col_idx], y_pos, cell, transform=ax_footer.transAxes,
                           ha="left", va="top", fontsize=6, color=color, fontweight=weight)

    # Soft line under table header row
    header_line_y = footer_y_start - footer_line_height - 0.02
    ax_footer.plot([table_x_start, 0.98], [header_line_y, header_line_y],
                   transform=ax_footer.transAxes, color=COLORS["grid"], linewidth=0.5, alpha=0.5)

    # Note: tight_layout removed - using GridSpec with fixed margins instead

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
    plot_trading_gaps: bool = True,
) -> Path:
    """Generate a short-only plot using daily short sale data and price context."""
    if df.empty:
        logger.warning("No data to plot for %s", symbol)
        return output_path

    plt.style.use("dark_background")

    # Use GridSpec for explicit control over panel spacing
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor(COLORS["background"])

    # Create GridSpec with fixed spacing to prevent layout shifts
    gs = GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[1, 1, 1],
        hspace=0.35,  # Fixed vertical spacing between panels
        left=0.08,    # Fixed left margin
        right=0.95,   # Fixed right margin
        top=0.94,     # Fixed top margin (room for suptitle)
        bottom=0.08,  # Fixed bottom margin (room for footer text)
    )

    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax in axes:
        _apply_primary_axis_style(ax)

    dates = df["date"]
    x_values, x_labels = _build_plot_x(df, plot_trading_gaps)

    # Panel 1: Short Ratio (daily)
    ax1 = axes[0]
    short_ratio = df["short_buy_sell_ratio"]
    denom_types = df["short_ratio_denominator_type"]
    colors1 = [_get_denom_color(v) for v in denom_types]

    valid_mask = ~short_ratio.isna()
    if valid_mask.any():
        _plot_smooth_line(ax1, x_values, short_ratio, COLORS["cyan"], valid_mask, linewidth=MAIN_LINE_WIDTH)
        ax1.scatter(
            x_values,
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
            x_values[valid_mask2],
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
        ax3.plot(x_values[valid_mask3], close[valid_mask3], color=COLORS["green"], linewidth=MAIN_LINE_WIDTH, alpha=0.85)
        ax3.scatter(
            x_values[valid_mask3],
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

    if plot_trading_gaps:
        # Format x-axis dates on ALL panels (not just bottom)
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Show every day
            ax.tick_params(axis="x", labelbottom=True)  # Enable x labels on all panels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

        # Add padding to x-axis so data doesn't sit at edges
        x_min = dates.min() - timedelta(hours=12)
        x_max = dates.max() + timedelta(hours=12)
        for ax in axes:
            ax.set_xlim(x_min, x_max)
    else:
        x_min = -0.5
        x_max = len(dates) - 0.5
        x_ticks = x_values.tolist()
        labels = x_labels or ["" for _ in x_ticks]
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.tick_params(axis="x", labelbottom=True)

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

    # Note: tight_layout removed - using GridSpec with fixed margins instead

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
    plot_trading_gaps: bool = True,
    panel1_metric: str | None = None,
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
                plot_symbol_metrics(
                    df,
                    symbol,
                    output_path,
                    plot_trading_gaps=plot_trading_gaps,
                    panel1_metric=panel1_metric,
                )
                output_paths.append(output_path)

            if mode in ("short_only", "both"):
                output_path = output_dir / f"{symbol.lower()}_short_only_{file_suffix}.png"
                plot_short_only_metrics(df, symbol, output_path, plot_trading_gaps=plot_trading_gaps)
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
    parser.add_argument(
        "--plot-trading-gaps",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Show gaps for non-trading days (true/false). Defaults to config.",
    )
    parser.add_argument(
        "--panel1-metric",
        type=str,
        default=None,
        choices=sorted(PANEL1_METRICS.keys()),
        help="Panel 1 metric: vw_flow, combined_ratio, or finra_buy_volume (default: config).",
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

    plot_trading_gaps = config.plot_trading_gaps
    if args.plot_trading_gaps is not None:
        plot_trading_gaps = args.plot_trading_gaps.lower() == "true"
    panel1_metric = args.panel1_metric or config.panel1_metric

    paths = render_metrics_plots(
        db_path=config.db_path,
        output_dir=output_dir,
        dates=dates_list,
        tickers=tickers,
        mode=args.mode,
        plot_trading_gaps=plot_trading_gaps,
        panel1_metric=panel1_metric,
    )
    print(f"Generated {len(paths)} plot(s)")


if __name__ == "__main__":
    main()
