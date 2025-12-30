"""
Combination plotter for multi-ticker runs.

Generates an additional multi-panel figure when COMBINATION_PLOT is enabled
and 2+ tickers are selected.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import cm, colors as mcolors

logger = logging.getLogger(__name__)

try:
    from .config import load_config
    from .plotter import (
        COLORS,
        MAIN_LINE_WIDTH,
        MARKER_SIZE,
        PANEL1_LINE_WIDTH,
        YLABEL_COLOR,
        YLABEL_SIZE,
        _add_ratio_thresholds,
        _apply_primary_axis_style,
        _compute_fig_width,
        _plot_smooth_line,
        _set_abs_ratio_axis,
    )
except ImportError:
    from config import load_config
    from plotter import (
        COLORS,
        MAIN_LINE_WIDTH,
        MARKER_SIZE,
        PANEL1_LINE_WIDTH,
        YLABEL_COLOR,
        YLABEL_SIZE,
        _add_ratio_thresholds,
        _apply_primary_axis_style,
        _compute_fig_width,
        _plot_smooth_line,
        _set_abs_ratio_axis,
    )


REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "short_buy_sell_ratio",
    "accumulation_score_display",
}

SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_cmap",
    [
        (0.0, COLORS["red"]),
        (0.5, "#888888"),
        (1.0, COLORS["green"]),
    ],
)


def _resolve_unique_output_path(output_dir: Path, base_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{base_name}.png"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = output_dir / f"{base_name}_{counter}.png"
        if not candidate.exists():
            return candidate
        counter += 1


def _get_table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:
        return []
    return [row[1] for row in rows]


def _ensure_required_columns(conn: duckdb.DuckDBPyConnection) -> None:
    columns = _get_table_columns(conn, "daily_metrics")
    if not columns:
        raise RuntimeError("Missing required table: daily_metrics")
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise RuntimeError(f"Missing required columns in daily_metrics: {', '.join(missing)}")


def _parse_dates(value: Optional[str]) -> Optional[list[date]]:
    if not value:
        return None
    return [date.fromisoformat(d.strip()) for d in value.split(",") if d.strip()]


def _fetch_combination_df(
    conn: duckdb.DuckDBPyConnection,
    tickers: list[str],
    dates: Optional[list[date]],
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    ticker_placeholders = ", ".join(["?" for _ in tickers])
    params: list[object] = list(tickers)
    date_clause = ""
    if dates:
        date_placeholders = ", ".join(["?" for _ in dates])
        date_clause = f" AND date IN ({date_placeholders})"
        params += list(dates)

    query = f"""
        SELECT
            date,
            symbol,
            short_buy_sell_ratio,
            accumulation_score_display,
            confidence
        FROM daily_metrics
        WHERE symbol IN ({ticker_placeholders}){date_clause}
        ORDER BY date, symbol
    """
    return conn.execute(query, params).df()


def _build_ticker_colors(tickers: list[str]) -> dict[str, str]:
    base_palette = [
        COLORS["cyan"],
        COLORS["yellow"],
        COLORS["green"],
        COLORS["red"],
        "#c77dff",
        "#ff9f1c",
        "#4cc9f0",
        "#9bdeac",
        "#f28482",
        "#8d99ae",
    ]
    if len(tickers) <= len(base_palette):
        return {ticker: base_palette[idx] for idx, ticker in enumerate(tickers)}
    cmap = cm.get_cmap("tab20", len(tickers))
    return {ticker: mcolors.to_hex(cmap(idx)) for idx, ticker in enumerate(tickers)}


def _resolve_weights(config, tickers: list[str]) -> dict[str, float]:
    weights = None
    for attr in ("combination_weights", "ticker_weights", "weights"):
        if hasattr(config, attr):
            weights = getattr(config, attr)
            break
    if not isinstance(weights, dict):
        equal = 1.0 / len(tickers) if tickers else 0.0
        return {ticker: equal for ticker in tickers}
    resolved = {ticker: float(weights.get(ticker, 0.0)) for ticker in tickers}
    if not any(resolved.values()):
        equal = 1.0 / len(tickers) if tickers else 0.0
        return {ticker: equal for ticker in tickers}
    return resolved


def _weighted_average_row(row: pd.Series, weights: dict[str, float]) -> float:
    values = row.dropna()
    if values.empty:
        return np.nan
    weight_values = np.array([weights.get(key, 0.0) for key in values.index], dtype=float)
    if weight_values.sum() <= 0:
        weight_values = np.ones_like(weight_values)
    weight_values = weight_values / weight_values.sum()
    return float(np.dot(values.values, weight_values))


def _plot_accumulation_bars(
    ax,
    dates: pd.Series,
    scores: pd.Series,
    confidence: pd.Series,
    label: str,
    label_color: str,
    emphasize: bool = False,
) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    score_display = scores.fillna(50)
    conf_display = confidence.fillna(0.5)

    for d, score, conf in zip(dates, score_display, conf_display):
        norm_score = np.clip(score / 100.0, 0, 1)
        bar_color = SCORE_CMAP(norm_score)
        alpha = 0.85 if conf >= 0.6 else 0.45
        bar_height = score / 100.0
        ax.bar(d, bar_height, bottom=0, color=bar_color, alpha=alpha, width=0.8, zorder=2)

        conf_height = 0.08 * conf
        if conf >= 0.7:
            conf_color = COLORS["green"]
        elif conf >= 0.4:
            conf_color = COLORS["yellow"]
        else:
            conf_color = COLORS["red"]
        ax.bar(d, conf_height, bottom=-0.12, color=conf_color, alpha=0.6, width=0.6, zorder=3)

        if not pd.isna(score):
            label_y = max(bar_height / 2, 0.15)
            ax.text(
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

    ax.set_ylim(-0.15, 1.05)
    ax.set_yticks([0, 0.3, 0.5, 0.7, 1.0])
    ax.set_yticklabels(["0", "30", "50", "70", "100"])
    ax.set_ylabel("Score", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)

    label_size = 13 if emphasize else 11
    ax.text(
        -0.06,
        0.5,
        label,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=label_size,
        fontweight="bold",
        color=label_color,
    )


def render_combination_plot(
    db_path: Path,
    output_dir: Path,
    dates: Optional[list[date]],
    tickers: list[str],
    combination_plot: bool,
) -> Optional[Path]:
    if not combination_plot:
        logger.info("Combination plot disabled (COMBINATION_PLOT=False).")
        return None
    if len(tickers) < 2:
        logger.info("Combination plot skipped (requires 2+ tickers).")
        return None

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        _ensure_required_columns(conn)
        df = _fetch_combination_df(conn, tickers, dates)
    finally:
        conn.close()

    if df.empty:
        logger.warning("No daily_metrics rows found for combination plot.")
        return None

    available = set(df["symbol"].unique())
    tickers = [ticker for ticker in tickers if ticker in available]
    if len(tickers) < 2:
        logger.info("Combination plot skipped (fewer than 2 tickers with data).")
        return None

    df["date"] = pd.to_datetime(df["date"])
    if dates:
        dates_index = pd.to_datetime(sorted(dates))
    else:
        dates_index = pd.to_datetime(sorted(df["date"].unique()))

    short_pivot = (
        df.pivot(index="date", columns="symbol", values="short_buy_sell_ratio")
        .reindex(dates_index)
    )
    accum_pivot = (
        df.pivot(index="date", columns="symbol", values="accumulation_score_display")
        .reindex(dates_index)
    )
    if "confidence" in df.columns:
        conf_pivot = df.pivot(index="date", columns="symbol", values="confidence").reindex(dates_index)
    else:
        conf_pivot = pd.DataFrame(index=dates_index, columns=tickers, data=np.nan)

    weights = _resolve_weights(load_config(), tickers)
    weighted_short = short_pivot[tickers].apply(
        lambda row: _weighted_average_row(row, weights), axis=1
    )
    weighted_accum = accum_pivot[tickers].apply(
        lambda row: _weighted_average_row(row, weights), axis=1
    )
    weighted_conf = conf_pivot[tickers].apply(
        lambda row: _weighted_average_row(row, weights), axis=1
    )

    ticker_colors = _build_ticker_colors(tickers)
    day_count = len(dates_index)
    fig_width = _compute_fig_width(day_count)
    panel1_height = 4.5
    accum_height = 1.9
    fig_height = panel1_height + accum_height * (len(tickers) + 1)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(COLORS["background"])

    grid = fig.add_gridspec(2, 1, height_ratios=[panel1_height, accum_height * (len(tickers) + 1)])
    ax1 = fig.add_subplot(grid[0, 0])
    _apply_primary_axis_style(ax1)

    # Panel 1: Daily Short Sale overlay
    for ticker in tickers:
        series = short_pivot[ticker]
        valid_mask = ~series.isna()
        if not valid_mask.any():
            continue
        color = ticker_colors[ticker]
        _plot_smooth_line(ax1, dates_index, series, color, valid_mask, linewidth=PANEL1_LINE_WIDTH, zorder=3)
        ax1.scatter(
            dates_index[valid_mask],
            series[valid_mask],
            c=color,
            s=MARKER_SIZE,
            zorder=4,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    weighted_valid = ~weighted_short.isna()
    if weighted_valid.any():
        _plot_smooth_line(
            ax1,
            dates_index,
            weighted_short,
            COLORS["white"],
            weighted_valid,
            linewidth=MAIN_LINE_WIDTH + 0.9,
            zorder=6,
        )
        ax1.scatter(
            dates_index[weighted_valid],
            weighted_short[weighted_valid],
            c=COLORS["white"],
            s=MARKER_SIZE,
            zorder=7,
            edgecolors=COLORS["white"],
            linewidths=0.4,
        )

    _set_abs_ratio_axis(ax1, short_pivot[tickers].stack(dropna=False))
    _add_ratio_thresholds(ax1)
    ax1.set_ylabel("Short Sale Buy/Sell Ratio", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax1.set_title(
        "Daily Short Sale Buy/Sell Ratio (Overlay)",
        color=COLORS["white"],
        fontsize=11,
        fontweight="bold",
        loc="left",
    )

    legend_handles = [
        Line2D([0], [0], color=ticker_colors[t], linewidth=MAIN_LINE_WIDTH, label=t)
        for t in tickers
    ]
    legend_handles.append(
        Line2D([0], [0], color=COLORS["white"], linewidth=MAIN_LINE_WIDTH + 0.9, label="Weighted Avg")
    )
    legend = ax1.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper left",
        fontsize=8,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text"])

    # Panel 2: Accumulation Score stack
    subgrid = grid[1, 0].subgridspec(len(tickers) + 1, 1, hspace=0.15)
    ax_weighted = fig.add_subplot(subgrid[0, 0])
    _plot_accumulation_bars(
        ax_weighted,
        dates_index,
        weighted_accum,
        weighted_conf,
        label="WEIGHTED",
        label_color=COLORS["white"],
        emphasize=True,
    )

    axes_accum = [ax_weighted]
    for idx, ticker in enumerate(tickers, start=1):
        ax = fig.add_subplot(subgrid[idx, 0], sharex=ax_weighted)
        _plot_accumulation_bars(
            ax,
            dates_index,
            accum_pivot[ticker],
            conf_pivot.get(ticker, pd.Series(index=dates_index, data=np.nan)),
            label=ticker,
            label_color=ticker_colors[ticker],
            emphasize=False,
        )
        axes_accum.append(ax)

    x_min = dates_index.min() - timedelta(hours=12)
    x_max = dates_index.max() + timedelta(hours=12)
    x_locator = mdates.DayLocator(interval=1)
    x_formatter = mdates.DateFormatter("%y-%m-%d")

    for ax in [ax1] + axes_accum:
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(x_locator)

    for ax in [ax1] + axes_accum[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes_accum[-1].tick_params(axis="x", labelbottom=True)
    plt.setp(axes_accum[-1].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    fig.suptitle(
        "Combination Plot - Multi-Ticker Overlay",
        color=COLORS["white"],
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    market_date = dates_index.max().strftime("%Y-%m-%d")
    tickers_slug = "-".join([t.lower() for t in tickers])
    output_path = _resolve_unique_output_path(
        output_dir, f"combination_plot_{market_date}_{tickers_slug}"
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(output_path, dpi=300, facecolor=COLORS["background"])
    plt.close(fig)
    logger.info("Saved combination plot: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render combination plot for multi-ticker runs.")
    parser.add_argument(
        "--dates",
        type=str,
        default=None,
        help="Comma-separated dates in YYYY-MM-DD format (default: config target dates)",
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

    config = load_config()
    dates = _parse_dates(args.dates) if args.dates else config.target_dates
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else config.tickers
    output_dir = Path(args.output_dir) if args.output_dir else config.plot_dir

    render_combination_plot(
        db_path=config.db_path,
        output_dir=output_dir,
        dates=dates,
        tickers=tickers,
        combination_plot=config.combination_plot,
    )


if __name__ == "__main__":
    main()
