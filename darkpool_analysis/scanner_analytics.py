from __future__ import annotations

from datetime import date
import logging

import pandas as pd

try:
    from .config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)

SCANNER_TREND_WEIGHT = 0.5
SCANNER_RATIO_WEIGHT = 0.35
SCANNER_VOLUME_WEIGHT = 0.15


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    return (series - mean) / std


def _aggregate_short_raw(short_raw_df: pd.DataFrame) -> pd.DataFrame:
    if short_raw_df.empty:
        return short_raw_df
    df = short_raw_df.copy()
    df["short_exempt_volume"] = df["short_exempt_volume"].fillna(0.0)
    grouped = (
        df.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            {
                "short_volume": "sum",
                "short_exempt_volume": "sum",
                "total_volume": "sum",
            }
        )
    )
    grouped["short_total_volume"] = grouped["total_volume"]
    grouped["short_volume_total"] = (
        grouped["short_volume"].fillna(0.0) + grouped["short_exempt_volume"].fillna(0.0)
    )
    return grouped


def _label_trend(score: float | None, config: Config) -> str:
    if score is None or pd.isna(score):
        return "Neutral"
    if score >= config.short_z_high:
        return "Accumulating"
    if score <= config.short_z_low:
        return "Distribution"
    return "Neutral"


def build_scanner_metrics(
    short_raw_df: pd.DataFrame,
    target_dates: list[date],
    config: Config,
) -> pd.DataFrame:
    if short_raw_df.empty or not target_dates:
        return pd.DataFrame()

    aggregated = _aggregate_short_raw(short_raw_df)
    if aggregated.empty:
        return pd.DataFrame()

    aggregated = aggregated.sort_values(["symbol", "trade_date"])
    aggregated["short_buy_volume"] = pd.to_numeric(aggregated["short_volume"], errors="coerce")
    aggregated["short_total_volume"] = pd.to_numeric(
        aggregated["short_total_volume"], errors="coerce"
    )
    aggregated["short_sell_volume"] = aggregated["short_total_volume"] - aggregated["short_buy_volume"]
    aggregated["short_sell_volume"] = pd.to_numeric(
        aggregated["short_sell_volume"], errors="coerce"
    ).clip(lower=0.0)

    aggregated["short_ratio"] = pd.NA
    has_total = aggregated["short_total_volume"].notna() & (aggregated["short_total_volume"] > 0)
    aggregated.loc[has_total, "short_ratio"] = (
        aggregated.loc[has_total, "short_buy_volume"]
        / aggregated.loc[has_total, "short_total_volume"]
    )

    aggregated["short_buy_sell_ratio"] = pd.NA
    has_sell = aggregated["short_sell_volume"].notna() & (aggregated["short_sell_volume"] > 0)
    aggregated.loc[has_sell, "short_buy_sell_ratio"] = (
        aggregated.loc[has_sell, "short_buy_volume"]
        / aggregated.loc[has_sell, "short_sell_volume"]
    )

    aggregated["short_ratio"] = pd.to_numeric(aggregated["short_ratio"], errors="coerce")
    aggregated["short_buy_sell_ratio"] = pd.to_numeric(
        aggregated["short_buy_sell_ratio"], errors="coerce"
    )

    aggregated["short_ratio_z"] = (
        aggregated.groupby("symbol")["short_ratio"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )
    aggregated["short_buy_sell_ratio_z"] = (
        aggregated.groupby("symbol")["short_buy_sell_ratio"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )
    aggregated["short_total_volume_z"] = (
        aggregated.groupby("symbol")["short_total_volume"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )

    trend_min_periods = min(config.scanner_trend_days, config.zscore_min_periods)
    aggregated["trend_score"] = (
        aggregated.groupby("symbol")["short_buy_sell_ratio_z"]
        .transform(lambda s: s.rolling(config.scanner_trend_days, min_periods=trend_min_periods).mean())
    )
    aggregated["trend_label"] = aggregated["trend_score"].apply(
        lambda score: _label_trend(score, config)
    )

    aggregated["scanner_direction"] = pd.NA
    aggregated.loc[aggregated["trend_score"] > 0, "scanner_direction"] = "Accumulating"
    aggregated.loc[aggregated["trend_score"] < 0, "scanner_direction"] = "Distribution"
    aggregated.loc[aggregated["trend_score"] == 0, "scanner_direction"] = "Neutral"

    outlier_candidates = aggregated[
        ["short_ratio_z", "short_buy_sell_ratio_z", "short_total_volume_z"]
    ].abs()
    aggregated["outlier_score"] = outlier_candidates.max(axis=1, skipna=True)

    aggregated["is_outlier"] = aggregated["outlier_score"] >= config.scanner_outlier_z
    aggregated["is_unusual_volume"] = (
        aggregated["short_total_volume_z"].abs() >= config.scanner_volume_z
    )
    aggregated["is_unusual_ratio"] = (
        aggregated["short_buy_sell_ratio_z"].abs() >= config.scanner_ratio_z
    )

    trend_component = aggregated["trend_score"].abs()
    ratio_component = aggregated["short_buy_sell_ratio_z"].abs()
    volume_component = aggregated["short_total_volume_z"].abs()
    score_components = pd.concat(
        [trend_component, ratio_component, volume_component], axis=1
    )
    has_component = score_components.notna().any(axis=1)
    score = (
        trend_component.fillna(0) * SCANNER_TREND_WEIGHT
        + ratio_component.fillna(0) * SCANNER_RATIO_WEIGHT
        + volume_component.fillna(0) * SCANNER_VOLUME_WEIGHT
    )
    aggregated["scanner_score"] = pd.NA
    aggregated.loc[has_component, "scanner_score"] = score[has_component]

    aggregated["trend_days"] = config.scanner_trend_days
    aggregated["inference_version"] = config.scanner_inference_version

    all_dates = pd.to_datetime(pd.Series(target_dates)).dt.date.unique().tolist()
    aggregated["date"] = pd.to_datetime(aggregated["trade_date"]).dt.date
    output = aggregated[
        [
            "date",
            "symbol",
            "short_volume",
            "short_exempt_volume",
            "total_volume",
            "short_total_volume",
            "short_buy_volume",
            "short_sell_volume",
            "short_ratio",
            "short_ratio_z",
            "short_buy_sell_ratio",
            "short_buy_sell_ratio_z",
            "short_total_volume_z",
            "trend_score",
            "trend_label",
            "scanner_direction",
            "trend_days",
            "outlier_score",
            "is_outlier",
            "is_unusual_volume",
            "is_unusual_ratio",
            "scanner_score",
            "inference_version",
        ]
    ].copy()

    output = output[output["date"].isin(all_dates)].copy()
    return output
