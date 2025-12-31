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
from matplotlib.transforms import blended_transform_factory

SPX_UP_COLOR = "#4aa3ff"
SPX_DOWN_COLOR = "#ff9f43"
SPX_OHLC_LINE_WIDTH = 1.2

logger = logging.getLogger(__name__)

try:
    from .config import (
        COMMODITIES_TICKERS,
        GLOBAL_MACRO_TICKERS,
        MAG8_TICKERS,
        SECTOR_CORE_TICKERS,
        load_config,
    )
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
        _set_log_ratio_axis,
    )
except ImportError:
    from config import (
        COMMODITIES_TICKERS,
        GLOBAL_MACRO_TICKERS,
        MAG8_TICKERS,
        SECTOR_CORE_TICKERS,
        load_config,
    )
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
        _set_log_ratio_axis,
    )


REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "short_buy_sell_ratio",
    "accumulation_score_display",
    "close",
}

SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_cmap",
    [
        (0.0, "#b026ff"),
        (0.5, "#555555"),
        (1.0, "#00ff88"),
    ],
)

LOG_BOT_THRESHOLD = np.log(1.25)
LOG_SELL_THRESHOLD = np.log(0.75)


def _add_log_ratio_thresholds(ax) -> None:
    ax.axhline(
        y=LOG_BOT_THRESHOLD,
        color=COLORS["green"],
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )
    ax.axhline(
        y=LOG_SELL_THRESHOLD,
        color=COLORS["red"],
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
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
    spx_columns = _get_table_columns(conn, "polygon_daily_agg_raw")
    if not spx_columns:
        raise RuntimeError("Missing required table: polygon_daily_agg_raw")
    required_spx = {"symbol", "trade_date", "open", "high", "low", "close"}
    missing_spx = sorted(required_spx.difference(spx_columns))
    if missing_spx:
        raise RuntimeError(
            "Missing required columns in polygon_daily_agg_raw: "
            + ", ".join(missing_spx)
        )


def _parse_dates(value: Optional[str]) -> Optional[list[date]]:
    if not value:
        return None
    return [date.fromisoformat(d.strip()) for d in value.split(",") if d.strip()]


def _fetch_issue_names(
    conn: duckdb.DuckDBPyConnection,
    tickers: list[str],
) -> dict[str, str]:
    if not tickers:
        return {}
    columns = _get_table_columns(conn, "finra_otc_weekly_raw")
    if not columns or "issue_name" not in columns:
        return {}
    placeholders = ", ".join(["?" for _ in tickers])
    query = f"""
        SELECT symbol, issue_name
        FROM (
            SELECT symbol, issue_name, week_start_date,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY week_start_date DESC) AS rn
            FROM finra_otc_weekly_raw
            WHERE symbol IN ({placeholders}) AND issue_name IS NOT NULL
        )
        WHERE rn = 1
    """
    rows = conn.execute(query, tickers).fetchall()
    return {row[0]: row[1] for row in rows if row[1]}


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
            confidence,
            close
        FROM daily_metrics
        WHERE symbol IN ({ticker_placeholders}){date_clause}
        ORDER BY date, symbol
    """
    return conn.execute(query, params).df()


def _resolve_spx_symbol(
    conn: duckdb.DuckDBPyConnection,
    min_date: date,
    max_date: date,
) -> Optional[str]:
    candidates = ["SPX", "I:SPX", "^SPX"]
    placeholders = ", ".join(["?" for _ in candidates])
    query = f"""
        SELECT symbol, COUNT(*) AS cnt
        FROM polygon_daily_agg_raw
        WHERE symbol IN ({placeholders}) AND trade_date BETWEEN ? AND ?
        GROUP BY symbol
        ORDER BY cnt DESC
    """
    rows = conn.execute(query, candidates + [min_date, max_date]).fetchall()
    if not rows:
        return None
    return rows[0][0]


def _fetch_spx_ohlc(
    conn: duckdb.DuckDBPyConnection,
    dates: list[date],
) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    min_date = min(dates)
    max_date = max(dates)
    symbol = _resolve_spx_symbol(conn, min_date, max_date)
    if not symbol:
        return pd.DataFrame()
    query = """
        SELECT trade_date AS date, open, high, low, close
        FROM polygon_daily_agg_raw
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    df = conn.execute(query, [symbol, min_date, max_date]).df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _plot_spx_ohlc_bars(ax, df: pd.DataFrame) -> None:
    dates_num = mdates.date2num(df["date"])
    if len(dates_num) > 1:
        bar_width = np.median(np.diff(dates_num)) * 0.6
    else:
        bar_width = 0.6
    half_width = bar_width / 2

    for x, open_, high, low, close in zip(
        dates_num, df["open"], df["high"], df["low"], df["close"]
    ):
        if pd.isna(open_) or pd.isna(close) or pd.isna(high) or pd.isna(low):
            continue
        color = SPX_UP_COLOR if close >= open_ else SPX_DOWN_COLOR
        ax.vlines(x, low, high, color=color, linewidth=SPX_OHLC_LINE_WIDTH, zorder=3)
        ax.hlines(open_, x - half_width, x, color=color, linewidth=SPX_OHLC_LINE_WIDTH, zorder=3)
        ax.hlines(close, x, x + half_width, color=color, linewidth=SPX_OHLC_LINE_WIDTH, zorder=3)


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


def _resolve_combination_title(tickers: list[str]) -> str:
    tickers_set = set(tickers)
    if tickers_set == set(SECTOR_CORE_TICKERS):
        return "Sector Core Overlay"
    if tickers_set == set(GLOBAL_MACRO_TICKERS):
        return "Global Macro Overlay"
    if tickers_set == set(COMMODITIES_TICKERS):
        return "Commodities Overlay"
    if tickers_set == set(MAG8_TICKERS):
        return "Mag 8 Overlay"
    return "Multi-Ticker Overlay"


def _compute_breadth_diffusion(scores: pd.DataFrame) -> pd.Series:
    if scores.empty:
        return pd.Series(dtype=float)
    valid = scores.notna()
    total = valid.sum(axis=1).replace(0, np.nan)
    above = (scores >= 70).sum(axis=1)
    below = (scores <= 30).sum(axis=1)
    return (above - below) / total


def _plot_breadth_diffusion(
    ax,
    dates: pd.Series,
    diffusion: pd.Series,
    label: str,
    label_color: str,
    display_name: str,
    color_boost: float = 1.0,
    curve_exp: float = 1.0,
) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    for d, value in zip(dates, diffusion):
        if pd.isna(value):
            continue
        adjusted_value = value * color_boost
        if curve_exp != 1.0:
            adjusted_value = np.sign(adjusted_value) * (abs(adjusted_value) ** curve_exp)
        adjusted_value = np.clip(adjusted_value, -1.0, 1.0)
        norm_value = (adjusted_value + 1.0) / 2.0
        bar_color = SCORE_CMAP(np.clip(norm_value, 0, 1))
        ax.bar(d, value, bottom=0, color=bar_color, alpha=0.7, width=0.8, zorder=2)

    ax.axhline(y=0.0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["-100", "-50", "0", "50", "100"])
    # ax.set_ylabel("Diffusion", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)

    ax.text(
        -0.06,
        0.5,
        label,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=label_color,
    )
    ax.text(
        0.01,
        0.92,
        display_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text_muted"],
    )


def _plot_accumulation_bars(
    ax,
    dates: pd.Series,
    scores: pd.Series,
    confidence: pd.Series,
    label: str,
    label_color: str,
    display_name: str,
    emphasize: bool = False,
    color_boost: float = 1.0,
    curve_exp: float = 1.0,
) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    score_display = scores.fillna(50)
    conf_display = confidence.fillna(0.5)

    for d, score, conf in zip(dates, score_display, conf_display):
        adjusted_score = 50 + (score - 50) * color_boost
        adjusted_score = np.clip(adjusted_score, 0, 100)
        norm_score = np.clip(adjusted_score / 100.0, 0, 1)
        if curve_exp != 1.0:
            dev = norm_score - 0.5
            norm_score = 0.5 + np.sign(dev) * (abs(dev) ** curve_exp)
            norm_score = np.clip(norm_score, 0, 1)
        bar_color = SCORE_CMAP(norm_score)
        alpha = 0.85 if conf >= 0.6 else 0.45
        bar_height = score / 100.0
        ax.bar(d, bar_height, bottom=0, color=bar_color, alpha=alpha, width=0.8, zorder=2)

        conf_height = 0.08 * conf
        if conf >= 0.7:
            conf_color = COLORS["neutral"]
        elif conf >= 0.4:
            conf_color = COLORS["yellow"]
        else:
            conf_color = COLORS["red"]
        ax.bar(d, conf_height, bottom=-0.12, color=conf_color, alpha=0.6, width=0.6, zorder=3)

        if not pd.isna(score):
            label_y = min(bar_height + 0.03, 1.02)
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
    ax.axhline(y=0.30, color=COLORS["red"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(y=0.70, color=COLORS["green"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    # ax.set_ylabel("Score", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)

    label_size = 11 if emphasize else 11
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
    display_size = 9 if emphasize else 8
    ax.text(
        0.01,
        0.92,
        display_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=display_size,
        fontweight="bold",
        color=COLORS["text_muted"],
    )


def _plot_accumulation_bars_centered(
    ax,
    dates: pd.Series,
    scores: pd.Series,
    confidence: pd.Series,
    label: str,
    label_color: str,
    display_name: str,
    emphasize: bool = False,
    color_boost: float = 1.0,
    curve_exp: float = 1.0,
) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    score_display = scores.fillna(50)
    conf_display = confidence.fillna(0.5)

    for d, score, conf in zip(dates, score_display, conf_display):
        adjusted_score = 50 + (score - 50) * color_boost
        adjusted_score = np.clip(adjusted_score, 0, 100)
        norm_score = np.clip(adjusted_score / 100.0, 0, 1)
        if curve_exp != 1.0:
            dev = norm_score - 0.5
            norm_score = 0.5 + np.sign(dev) * (abs(dev) ** curve_exp)
            norm_score = np.clip(norm_score, 0, 1)
        bar_color = SCORE_CMAP(norm_score)
        alpha = 0.85 if conf >= 0.6 else 0.45
        bar_height = score - 50.0
        ax.bar(d, bar_height, bottom=0, color=bar_color, alpha=alpha, width=0.8, zorder=2)

        conf_height = 8.0 * conf
        if conf >= 0.7:
            conf_color = COLORS["neutral"]
        elif conf >= 0.4:
            conf_color = COLORS["yellow"]
        else:
            conf_color = COLORS["red"]
        ax.bar(d, conf_height, bottom=-48.0, color=conf_color, alpha=0.6, width=0.6, zorder=3)

        if not pd.isna(score) and round(float(score)) != 50:
            if bar_height >= 0:
                label_y = min(bar_height + 4.0, 48.0)
            else:
                label_y = max(bar_height - 4.0, -48.0)
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

    ax.set_ylim(-50, 50)
    ax.set_yticks([-50, -25, 0, 25, 50])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.axhline(y=0.0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.axhline(y=-20.0, color=COLORS["red"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(y=20.0, color=COLORS["green"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    # ax.set_ylabel("Score", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)

    label_size = 11 if emphasize else 11
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
    display_size = 9 if emphasize else 8
    ax.text(
        0.01,
        0.92,
        display_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=display_size,
        fontweight="bold",
        color=COLORS["text_muted"],
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
        issue_names = _fetch_issue_names(conn, tickers)
        spx_df = pd.DataFrame()
        if not df.empty:
            if dates:
                spx_dates = dates
            else:
                spx_dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
            spx_df = _fetch_spx_ohlc(conn, spx_dates)
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
    short_log_pivot = short_pivot.where(short_pivot > 0)
    short_log_pivot = np.log(short_log_pivot)
    price_pivot = df.pivot(index="date", columns="symbol", values="close").reindex(dates_index)
    accum_pivot = (
        df.pivot(index="date", columns="symbol", values="accumulation_score_display")
        .reindex(dates_index)
    )
    if "confidence" in df.columns:
        conf_pivot = df.pivot(index="date", columns="symbol", values="confidence").reindex(dates_index)
    else:
        conf_pivot = pd.DataFrame(index=dates_index, columns=tickers, data=np.nan)

    weights = _resolve_weights(load_config(), tickers)
    weighted_short = short_log_pivot[tickers].apply(
        lambda row: _weighted_average_row(row, weights), axis=1
    )
    weighted_accum = accum_pivot[tickers].apply(
        lambda row: _weighted_average_row(row, weights), axis=1
    )
    weighted_conf = conf_pivot[tickers].apply(
        lambda row: _weighted_average_row(row, weights), axis=1
    )
    diffusion = _compute_breadth_diffusion(accum_pivot[tickers])

    ticker_colors = _build_ticker_colors(tickers)
    day_count = len(dates_index)
    fig_width = _compute_fig_width(day_count)
    panel1_height = 4.5 * 3 * 0.5 * 0.75
    base_accum_height = 1.9 * 0.5
    weighted_height = base_accum_height * 2.5
    ticker_height = base_accum_height * 2
    panel0_height = panel1_height
    spx_height = panel1_height
    fig_height = (
        spx_height
        + panel0_height
        + panel1_height
        + weighted_height * 3
        + ticker_height * len(tickers)
    )

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(COLORS["background"])

    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[
            spx_height,
            panel0_height,
            panel1_height,
            weighted_height * 3 + ticker_height * len(tickers),
        ],
    )
    ticker_line_width = max(PANEL1_LINE_WIDTH * 0.6, 0.9)
    ax_spx = fig.add_subplot(grid[0, 0])
    _apply_primary_axis_style(ax_spx)
    if spx_df.empty:
        logger.warning("No SPX OHLC data found for the selected date range.")
        ax_spx.set_ylim(0, 1)
        ax_spx.set_ylabel("SPX Price", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
        ax_spx.set_title(
            "SPX Price Action (OHLC)",
            color=COLORS["white"],
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        ax_spx.text(
            0.5,
            0.5,
            "SPX data unavailable",
            transform=ax_spx.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=COLORS["text_muted"],
        )
    else:
        _plot_spx_ohlc_bars(ax_spx, spx_df)
        spx_low = spx_df["low"].min()
        spx_high = spx_df["high"].max()
        if pd.isna(spx_low) or pd.isna(spx_high):
            spx_low = spx_df["close"].min()
            spx_high = spx_df["close"].max()
        spx_span = spx_high - spx_low if pd.notna(spx_high) and pd.notna(spx_low) else 0
        if not spx_span or spx_span <= 0:
            spx_span = float(spx_df["close"].mean() or 1.0)
        spx_pad = spx_span * 0.04
        ax_spx.set_ylim(spx_low - spx_pad * 3, spx_high + spx_pad * 3)
        ax_spx.set_ylabel("SPX Price", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
        ax_spx.set_title(
            "SPX Price Action (OHLC)",
            color=COLORS["white"],
            fontsize=11,
            fontweight="bold",
            loc="left",
        )

    ax0 = fig.add_subplot(grid[1, 0], sharex=ax_spx)
    _apply_primary_axis_style(ax0)

    # Panel 0: Normalized price action overlay (start at 0)
    price_labels: dict[str, float] = {}
    for ticker in tickers:
        series = price_pivot[ticker]
        if series.dropna().empty:
            continue
        baseline = series.dropna().iloc[0]
        if pd.isna(baseline) or baseline == 0:
            continue
        normalized = series / baseline - 1.0
        valid_mask = ~normalized.isna()
        if not valid_mask.any():
            continue
        color = ticker_colors[ticker]
        price_labels[ticker] = float(normalized[valid_mask].iloc[-1])
        ax0.plot(
            dates_index[valid_mask],
            normalized[valid_mask],
            color=color,
            linewidth=ticker_line_width,
            zorder=3,
        )

    if not price_pivot.empty:
        normalized_all = []
        for ticker in tickers:
            series = price_pivot[ticker]
            if series.dropna().empty:
                continue
            baseline = series.dropna().iloc[0]
            if pd.isna(baseline) or baseline == 0:
                continue
            normalized_all.append((series / baseline - 1.0).dropna())
        if normalized_all:
            max_abs = pd.concat(normalized_all, axis=0).abs().max()
        else:
            max_abs = 0.02
    else:
        max_abs = 0.02
    if pd.isna(max_abs) or max_abs == 0:
        max_abs = 0.02
    bound = float(np.ceil(max_abs * 1.10 * 100) / 100)
    ax0.set_ylim(-bound, bound)
    ax0.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax0.set_ylabel("Price (Normalized)", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax0.set_title(
        "Price Action (Normalized)",
        color=COLORS["white"],
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    if price_labels:
        for ticker, value in price_labels.items():
            ax0.annotate(
                f"{value * 100:+.1f}%",
                xy=(dates_index.max(), value),
                xytext=(6, 4),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=5,
                fontweight="bold",
                color=ticker_colors[ticker],
                clip_on=True,
            )
    legend0_handles = [
        Line2D([0], [0], color=ticker_colors[t], linewidth=MAIN_LINE_WIDTH, label=t)
        for t in tickers
    ]
    legend0 = ax0.legend(
        legend0_handles,
        [h.get_label() for h in legend0_handles],
        loc="upper right",
        ncol=len(legend0_handles),
        fontsize=8,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for text in legend0.get_texts():
        text.set_color(COLORS["text"])

    ax1 = fig.add_subplot(grid[2, 0], sharex=ax_spx)
    _apply_primary_axis_style(ax1)

    # Panel 1: Daily Short Sale overlay
    for ticker in tickers:
        series = short_log_pivot[ticker]
        valid_mask = ~series.isna()
        if not valid_mask.any():
            continue
        color = ticker_colors[ticker]
        _plot_smooth_line(
            ax1,
            dates_index,
            series,
            color,
            valid_mask,
            linewidth=ticker_line_width,
            zorder=3,
        )

    weighted_valid = ~weighted_short.isna()
    if weighted_valid.any():
        _plot_smooth_line(
            ax1,
            dates_index,
            weighted_short,
            COLORS["white"],
            weighted_valid,
            linewidth=MAIN_LINE_WIDTH + 1.4,
            zorder=6,
        )

    stacked_log = short_log_pivot[tickers].stack(dropna=False)
    min_val = stacked_log.min(skipna=True)
    max_val = stacked_log.max(skipna=True)
    if pd.isna(min_val) or pd.isna(max_val):
        bound = 1.5
    else:
        bound = max(abs(float(min_val)), abs(float(max_val)), 1.5)
        bound = float(np.ceil((bound + 0.1) * 10) / 10)
    ax1.set_ylim(-bound, bound)
    ax1.set_yticks(np.arange(-bound, bound + 0.001, 0.5))
    ax1.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.9, alpha=0.6, zorder=1)
    _add_log_ratio_thresholds(ax1)
    ax1.set_ylabel("Short Sale Log(Buy/Sell)", color=YLABEL_COLOR, fontsize=YLABEL_SIZE)
    ax1.set_title(
        "Daily Short Sale Log(Buy/Sell) (Overlay)",
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
        loc="upper right",
        ncol=len(legend_handles),
        fontsize=8,
        frameon=True,
        facecolor=COLORS["background"],
        framealpha=0.7,
        edgecolor=COLORS["grid"],
        columnspacing=1.0,
        handletextpad=0.4,
    )
    for text in legend.get_texts():
        text.set_color(COLORS["text"])

    # Panel 2: Accumulation Score stack
    subgrid = grid[3, 0].subgridspec(
        len(tickers) + 3,
        1,
        hspace=0.28,
        height_ratios=[weighted_height, weighted_height, weighted_height]
        + [base_accum_height] * len(tickers),
    )
    ax_breadth = fig.add_subplot(subgrid[0, 0])
    _plot_breadth_diffusion(
        ax_breadth,
        dates_index,
        diffusion,
        label="BREADTH",
        label_color=COLORS["white"],
        display_name="Breadth / Diffusion",
        color_boost=1.0,
        curve_exp=0.75,
    )
    ax_centered = fig.add_subplot(subgrid[1, 0], sharex=ax_breadth)
    _plot_accumulation_bars_centered(
        ax_centered,
        dates_index,
        weighted_accum,
        weighted_conf,
        label="CENTERED",
        label_color=COLORS["white"],
        display_name="Weighted (Centered)",
        emphasize=True,
        color_boost=1.0,
        curve_exp=0.75,
    )
    ax_weighted = fig.add_subplot(subgrid[2, 0], sharex=ax_breadth)
    _plot_accumulation_bars(
        ax_weighted,
        dates_index,
        weighted_accum,
        weighted_conf,
        label="WEIGHTED",
        label_color=COLORS["white"],
        display_name="Weighted Average",
        emphasize=True,
        color_boost=1.0,
        curve_exp=0.75,
    )

    axes_accum = [ax_breadth, ax_centered, ax_weighted]
    for idx, ticker in enumerate(tickers, start=3):
        ax = fig.add_subplot(subgrid[idx, 0], sharex=ax_centered)
        display_name = issue_names.get(ticker, ticker)
        _plot_accumulation_bars(
            ax,
            dates_index,
            accum_pivot[ticker],
            conf_pivot.get(ticker, pd.Series(index=dates_index, data=np.nan)),
            label=ticker,
            label_color=ticker_colors[ticker],
            display_name=display_name,
            emphasize=False,
            color_boost=1.0,
            curve_exp=0.85,
        )
        axes_accum.append(ax)

    x_min = dates_index.min() - timedelta(hours=12)
    x_max = dates_index.max() + timedelta(hours=36)
    x_locator = mdates.DayLocator(interval=1)
    x_formatter = mdates.DateFormatter("%y-%m-%d")

    for ax in [ax_spx, ax0, ax1] + axes_accum:
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(x_locator)

    axes_with_labels = {axes_accum[2], axes_accum[-1]}
    for ax in [ax_spx, ax0, ax1] + axes_accum:
        if ax in axes_with_labels or ax in (ax0, ax1):
            ax.tick_params(axis="x", labelbottom=True)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
        else:
            ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(
        _resolve_combination_title(tickers),
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
