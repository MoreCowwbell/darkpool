from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_buy_ratio_series(db_path: Path, output_dir: Path, symbols: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["date"], df["buy_ratio"], marker="o", linewidth=1.5)
            ax.axhline(1.25, color="green", linestyle="--", linewidth=1)
            ax.axhline(0.80, color="orange", linestyle="--", linewidth=1)
            ax.set_title(f"{symbol} Buy Ratio (Inferred)")
            ax.set_ylabel("Buy Ratio (Estimated)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)

            for idx, row in df.iterrows():
                ratio = row["buy_ratio"]
                if ratio > 1.25:
                    ax.annotate("BOT", (row["date"], ratio), xytext=(0, 8), textcoords="offset points")
                elif ratio < 0.80:
                    ax.annotate("SELL", (row["date"], ratio), xytext=(0, -12), textcoords="offset points")

            fig.tight_layout()
            output_path = output_dir / f"{symbol.lower()}_buy_ratio.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            logger.info("Saved plot: %s", output_path)
    finally:
        conn.close()
