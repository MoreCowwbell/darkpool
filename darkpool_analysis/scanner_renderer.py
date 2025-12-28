from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _plot_top_scores(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return

    colors = []
    for direction in df["scanner_direction"].fillna("Neutral"):
        if direction == "Accumulating":
            colors.append("#3cbf8a")
        elif direction == "Distribution":
            colors.append("#d06c6c")
        else:
            colors.append("#8b8b8b")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(df["symbol"], df["scanner_score"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Scanner Score (abs-weighted)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_score_histogram(scores: pd.Series, path: Path, title: str) -> None:
    scores = scores.dropna()
    if scores.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=50, color="#7ab6ff", alpha=0.85)
    ax.set_xlabel("Scanner Score (abs-weighted)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_scanner_outputs(
    metrics_df: pd.DataFrame,
    output_root: Path,
    run_date: date,
    top_n: int,
    export_full: bool,
) -> None:
    date_tag = run_date.isoformat()
    run_dir = output_root / date_tag

    daily = metrics_df[metrics_df["date"] == run_date].copy()
    if daily.empty:
        return

    daily = daily[daily["scanner_score"].notna()].copy()
    if daily.empty:
        return
    daily = daily.sort_values(["scanner_score"], ascending=False)
    top_df = daily.head(top_n).copy()
    top_df["rank"] = range(1, len(top_df) + 1)

    if export_full:
        _write_csv(daily, run_dir / "scanner_metrics.csv")
    _write_csv(top_df, run_dir / "scanner_top.csv")

    _plot_top_scores(
        top_df,
        run_dir / "scanner_top.png",
        f"Scanner Top {top_n} - {date_tag}",
    )
    _plot_score_histogram(
        daily["scanner_score"],
        run_dir / "scanner_scores.png",
        f"Scanner Score Distribution - {date_tag}",
    )
