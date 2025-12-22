from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt

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


def plot_buy_ratio_series(db_path: Path, output_dir: Path, symbols: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use dark background style
    plt.style.use("dark_background")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        for symbol in symbols:
            df = conn.execute(
                """
                SELECT date, buy_ratio
                FROM darkpool_daily_summary
                WHERE symbol = ?
                ORDER BY date
                """,
                [symbol],
            ).df()
            if df.empty:
                logger.warning("No buy ratio data available for %s.", symbol)
                continue

            df = df.dropna(subset=["buy_ratio"])
            if df.empty:
                logger.warning("No non-null buy ratio values for %s.", symbol)
                continue

            # Create figure with dark background
            fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["background"])
            ax.set_facecolor(COLORS["background"])

            # Subtle fill for neutral zone
            ax.axhspan(0.80, 1.25, alpha=0.08, color=COLORS["fill"], zorder=1)

            # Threshold lines
            ax.axhline(
                1.25,
                color=COLORS["bot_line"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="BOT (>1.25)",
                zorder=2,
            )
            ax.axhline(
                1.0,
                color=COLORS["neutral"],
                linestyle=":",
                linewidth=1,
                alpha=0.5,
                label="Neutral (1.0)",
                zorder=2,
            )
            ax.axhline(
                0.80,
                color=COLORS["sell_line"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="SELL (<0.80)",
                zorder=2,
            )

            # Main data line with styled markers
            ax.plot(
                df["date"],
                df["buy_ratio"],
                color=COLORS["line"],
                linewidth=2.5,
                marker="o",
                markersize=8,
                markerfacecolor=COLORS["line"],
                markeredgecolor=COLORS["marker_edge"],
                markeredgewidth=1.5,
                zorder=3,
                label="Buy Ratio",
            )

            # Title and labels
            ax.set_title(
                f"{symbol} Dark Pool Buy Ratio",
                fontsize=18,
                fontweight="bold",
                color=COLORS["title"],
                pad=20,
            )
            ax.set_ylabel(
                "Buy Ratio (bought ÷ sold)",
                fontsize=12,
                color=COLORS["label"],
                labelpad=10,
            )
            ax.set_xlabel("Date", fontsize=12, color=COLORS["label"], labelpad=10)

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
            bot_mask = df["buy_ratio"] > 1.25
            sell_mask = df["buy_ratio"] < 0.80

            for d, r in zip(df.loc[bot_mask, "date"], df.loc[bot_mask, "buy_ratio"]):
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

            for d, r in zip(df.loc[sell_mask, "date"], df.loc[sell_mask, "buy_ratio"]):
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
            output_path = output_dir / f"{symbol.lower()}_buy_ratio.png"
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
