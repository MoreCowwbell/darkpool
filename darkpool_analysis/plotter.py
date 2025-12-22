from __future__ import annotations

import logging
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

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


def plot_buy_ratio_series(db_path: Path, output_dir: Path, symbols: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use dark background style
    plt.style.use("dark_background")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        for symbol in symbols:
            df = conn.execute(
                """
                SELECT date, estimated_bought, estimated_sold, total_off_exchange_volume
                FROM darkpool_daily_summary
                WHERE symbol = ?
                ORDER BY date
                """,
                [symbol],
            ).df()
            if df.empty:
                logger.warning("No estimated flow data available for %s.", symbol)
                continue

            # Compute log ratio: ln(bought / sold)
            df["log_ratio"] = df.apply(
                lambda row: _compute_log_ratio(row["estimated_bought"], row["estimated_sold"]),
                axis=1,
            )
            df = df.dropna(subset=["log_ratio"])
            if df.empty:
                logger.warning("No valid log ratio values for %s.", symbol)
                continue

            # Scale marker sizes by volume
            volumes = df["total_off_exchange_volume"].fillna(0).values.astype(float)
            marker_sizes = _scale_marker_sizes(volumes)

            # Create figure with dark background
            fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["background"])
            ax.set_facecolor(COLORS["background"])

            # Subtle fill for neutral zone (between log thresholds)
            ax.axhspan(LOG_SELL_THRESHOLD, LOG_BOT_THRESHOLD, alpha=0.08, color=COLORS["fill"], zorder=1)

            # Threshold lines in log space
            ax.axhline(
                LOG_BOT_THRESHOLD,
                color=COLORS["bot_line"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="BOT threshold (+0.22)",
                zorder=2,
            )
            ax.axhline(
                0.0,
                color=COLORS["neutral"],
                linestyle=":",
                linewidth=1,
                alpha=0.5,
                label="Neutral (0)",
                zorder=2,
            )
            ax.axhline(
                LOG_SELL_THRESHOLD,
                color=COLORS["sell_line"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="SELL threshold (-0.22)",
                zorder=2,
            )

            # Line connecting data points
            ax.plot(
                df["date"],
                df["log_ratio"],
                color=COLORS["line"],
                linewidth=2,
                alpha=0.6,
                zorder=2,
            )

            # Scatter with volume-scaled markers
            ax.scatter(
                df["date"],
                df["log_ratio"],
                s=marker_sizes,
                c=COLORS["line"],
                edgecolors=COLORS["marker_edge"],
                linewidths=1.5,
                zorder=3,
                label="Log ratio (size ∝ volume)",
            )

            # Title and labels
            ax.set_title(
                f"{symbol} Dark Pool Log Buy/Sell Ratio",
                fontsize=18,
                fontweight="bold",
                color=COLORS["title"],
                pad=20,
            )
            ax.set_ylabel(
                "log(Buy / Sell)",
                fontsize=12,
                color=COLORS["label"],
                labelpad=10,
            )
            ax.set_xlabel("Date", fontsize=12, color=COLORS["label"], labelpad=10)

            # Set y-axis range: default ±1.0, expand symmetrically if data exceeds
            max_abs = max(abs(df["log_ratio"].min()), abs(df["log_ratio"].max()))
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

            # Colored annotations for threshold crossings (using log thresholds)
            bot_mask = df["log_ratio"] > LOG_BOT_THRESHOLD
            sell_mask = df["log_ratio"] < LOG_SELL_THRESHOLD

            for d, r in zip(df.loc[bot_mask, "date"], df.loc[bot_mask, "log_ratio"]):
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

            for d, r in zip(df.loc[sell_mask, "date"], df.loc[sell_mask, "log_ratio"]):
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
            output_path = output_dir / f"{symbol.lower()}_log_ratio.png"
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
