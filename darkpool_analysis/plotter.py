from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

# Dark mode color palette
COLORS = {
    "background": "#0d1117",
    "line": "#00d4ff",
    "marker_edge": "#ffffff",
    "bot_line": "#00ff88",
    "sell_line": "#ff6b6b",
    "neutral": "#888888",
    "grid": "#30363d",
    "fill": "#1e3a5f",
    "title": "#ffffff",
    "label": "#8b949e",
    "tick": "#8b949e",
    "spine": "#30363d",
}

# Thresholds in log space
LOG_BOT_THRESHOLD = math.log(1.25)   # ≈ +0.223
LOG_SELL_THRESHOLD = math.log(0.80)  # ≈ -0.223

# Thresholds in absolute space
ABS_BOT_THRESHOLD = 1.25
ABS_SELL_THRESHOLD = 0.80
ABS_NEUTRAL = 1.0

# Marker size range for volume scaling
MARKER_SIZE_MIN = 30
MARKER_SIZE_MAX = 300


def _compute_log_ratio(bought: float, sold: float) -> float | None:
    """Compute ln(bought/sold), returning None for invalid inputs."""
    if bought is None or sold is None or bought <= 0 or sold <= 0:
        return None
    return math.log(bought / sold)


def _scale_marker_sizes(volumes: np.ndarray) -> np.ndarray:
    """Scale marker sizes based on volume (min-max normalization)."""
    if len(volumes) == 0:
        return np.array([])
    v_min, v_max = volumes.min(), volumes.max()
    if v_max == v_min:
        return np.full(len(volumes), (MARKER_SIZE_MIN + MARKER_SIZE_MAX) / 2)
    normalized = (volumes - v_min) / (v_max - v_min)
    return MARKER_SIZE_MIN + normalized * (MARKER_SIZE_MAX - MARKER_SIZE_MIN)


def plot_buy_ratio_series(db_path: Path, output_dir: Path, symbols: list[str], plot_mode: str = "log") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use dark background style
    plt.style.use("dark_background")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        for symbol in symbols:
            df = conn.execute(
                """
                SELECT date, estimated_bought, estimated_sold, buy_ratio, total_off_exchange_volume
                FROM darkpool_daily_summary
                WHERE symbol = ?
                ORDER BY date
                """,
                [symbol],
            ).df()
            if df.empty:
                logger.warning("No estimated flow data available for %s.", symbol)
                continue

            # Compute values based on plot mode
            if plot_mode == "absolute":
                # Use raw buy_ratio directly
                df["plot_value"] = df["buy_ratio"]
                df = df.dropna(subset=["plot_value"])
                if df.empty:
                    logger.warning("No valid buy_ratio values for %s.", symbol)
                    continue
            else:
                # Compute log ratio: ln(bought / sold)
                df["plot_value"] = df.apply(
                    lambda row: _compute_log_ratio(row["estimated_bought"], row["estimated_sold"]),
                    axis=1,
                )
                df = df.dropna(subset=["plot_value"])
                if df.empty:
                    logger.warning("No valid log ratio values for %s.", symbol)
                    continue

            # Scale marker sizes by volume
            volumes = df["total_off_exchange_volume"].fillna(0).values.astype(float)
            marker_sizes = _scale_marker_sizes(volumes)

            # Create figure with dark background
            fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["background"])
            ax.set_facecolor(COLORS["background"])

            # Mode-dependent thresholds and neutral zone
            if plot_mode == "absolute":
                bot_thresh = ABS_BOT_THRESHOLD
                sell_thresh = ABS_SELL_THRESHOLD
                neutral_val = ABS_NEUTRAL
                bot_label = f"BOT threshold ({bot_thresh})"
                sell_label = f"SELL threshold ({sell_thresh})"
                neutral_label = "Neutral (1.0)"
            else:
                bot_thresh = LOG_BOT_THRESHOLD
                sell_thresh = LOG_SELL_THRESHOLD
                neutral_val = 0.0
                bot_label = "BOT threshold (+0.22)"
                sell_label = "SELL threshold (-0.22)"
                neutral_label = "Neutral (0)"

            # Subtle fill for neutral zone
            ax.axhspan(sell_thresh, bot_thresh, alpha=0.08, color=COLORS["fill"], zorder=1)

            # Threshold lines
            ax.axhline(
                bot_thresh,
                color=COLORS["bot_line"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=bot_label,
                zorder=2,
            )
            ax.axhline(
                neutral_val,
                color=COLORS["neutral"],
                linestyle=":",
                linewidth=1,
                alpha=0.5,
                label=neutral_label,
                zorder=2,
            )
            ax.axhline(
                sell_thresh,
                color=COLORS["sell_line"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=sell_label,
                zorder=2,
            )

            # Smooth spline line connecting data points
            if len(df) >= 4:
                # Convert dates to numeric for interpolation
                date_nums = mdates.date2num(df["date"])
                values = df["plot_value"].values

                # Smooth shape-preserving curve through points
                if len(df) >= 3:
                    date_nums = mdates.date2num(df["date"])
                    values = df["plot_value"].to_numpy()

                    x_smooth = np.linspace(date_nums.min(), date_nums.max(), 150)  # 120–180 is plenty
                    interp = PchipInterpolator(date_nums, values)
                    y_smooth = interp(x_smooth)

                    ax.plot(
                        mdates.num2date(x_smooth),
                        y_smooth,
                        color=COLORS["line"],
                        linewidth=2,
                        alpha=0.6,
                        zorder=2,
                    )
            else:
                # Fall back to straight lines for small datasets (< 4 points)
                ax.plot(
                    df["date"],
                    df["plot_value"],
                    color=COLORS["line"],
                    linewidth=2,
                    alpha=0.6,
                    zorder=2,
                )

            # Scatter with volume-scaled markers
            scatter_label = "Buy ratio (size ∝ volume)" if plot_mode == "absolute" else "Log ratio (size ∝ volume)"
            ax.scatter(
                df["date"],
                df["plot_value"],
                s=marker_sizes,
                c=COLORS["line"],
                edgecolors=COLORS["marker_edge"],
                linewidths=1.5,
                zorder=3,
                label=scatter_label,
            )

            # Title and labels (mode-dependent)
            if plot_mode == "absolute":
                title_suffix = "(Absolute)"
                y_label = "Buy / Sell Ratio"
            else:
                title_suffix = "(Log)"
                y_label = "log(Buy / Sell)"

            ax.set_title(
                f"{symbol} DarkPool Buy/Sell Ratio {title_suffix}",
                fontsize=12,
                fontweight="bold",
                color=COLORS["title"],
                pad=20,
            )
            ax.set_ylabel(
                y_label,
                fontsize=12,
                color=COLORS["label"],
                labelpad=10,
            )
            ax.set_xlabel("Date", fontsize=12, color=COLORS["label"], labelpad=10)

            # Set y-axis range based on mode
            if plot_mode == "absolute":
                # Absolute mode: 0.0 to 2.0+, expand if data exceeds 2.0
                y_min = 0.0
                y_max = max(2.0, df["plot_value"].max() * 1.1)
                ax.set_ylim(y_min, y_max)
            else:
                # Log mode: symmetric around 0, default ±1.0
                max_abs = max(abs(df["plot_value"].min()), abs(df["plot_value"].max()))
                y_limit = max(1.0, max_abs * 1.1)  # 10% padding if beyond ±1.0
                ax.set_ylim(-y_limit, y_limit)

            # Grid styling
            ax.grid(True, alpha=0.3, color=COLORS["grid"], linestyle="-", linewidth=0.5)

            # Tick styling
            ax.tick_params(colors=COLORS["tick"], labelsize=10, length=5)
            plt.xticks(rotation=45, ha="right")

            # Spine styling
            for spine in ax.spines.values():
                spine.set_color(COLORS["spine"])
                spine.set_linewidth(0.5)

            # Colored annotations for threshold crossings
            bot_mask = df["plot_value"] > bot_thresh
            sell_mask = df["plot_value"] < sell_thresh

            for d, r in zip(df.loc[bot_mask, "date"], df.loc[bot_mask, "plot_value"]):
                ax.annotate(
                    "BOT",
                    (d, r),
                    xytext=(0, 12),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color=COLORS["bot_line"],
                    ha="center",
                    va="bottom",
                )

            for d, r in zip(df.loc[sell_mask, "date"], df.loc[sell_mask, "plot_value"]):
                ax.annotate(
                    "SELL",
                    (d, r),
                    xytext=(0, -12),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color=COLORS["sell_line"],
                    ha="center",
                    va="top",
                )

            # Legend with dark styling
            legend = ax.legend(
                loc="upper right",
                fontsize=9,
                framealpha=0.3,
                facecolor=COLORS["background"],
                edgecolor=COLORS["spine"],
            )
            for text in legend.get_texts():
                text.set_color(COLORS["label"])

            # Tight layout and save
            fig.tight_layout()
            filename_suffix = "abs_ratio" if plot_mode == "absolute" else "log_ratio"
            output_path = output_dir / f"{symbol.lower()}_{filename_suffix}.png"
            fig.savefig(
                output_path,
                dpi=300,
                facecolor=COLORS["background"],
                edgecolor="none",
                bbox_inches="tight",
            )
            plt.close(fig)
            logger.info("Saved plot: %s", output_path)
    finally:
        conn.close()


def main() -> None:
    """CLI entry point for plotter."""
    parser = argparse.ArgumentParser(
        description="Generate dark pool ratio plots"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["log", "absolute"],
        help="Plot mode: 'log' or 'absolute' (default: from config)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load config
    try:
        from .config import load_config
    except ImportError:
        from config import load_config

    config = load_config()

    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = config.tickers

    # Get output dir
    output_dir = Path(args.output_dir) if args.output_dir else config.plot_dir

    # Get plot mode
    plot_mode = args.mode if args.mode else config.plot_mode

    # Generate plots
    plot_buy_ratio_series(config.db_path, output_dir, symbols, plot_mode)
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
