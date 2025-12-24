from __future__ import annotations

from datetime import date

import pandas as pd


def build_darkpool_estimates(
    finra_week_df: pd.DataFrame,
    lit_flow_df: pd.DataFrame,
    snapshot_date: date,
    inference_version: str,
    finra_week: date,
    include_polygon_only: bool = True,
) -> pd.DataFrame:
    finra = finra_week_df.copy()
    ratios = lit_flow_df[["symbol", "lit_buy_ratio"]].copy()

    # Use outer join to include Polygon-only tickers (no FINRA data)
    if include_polygon_only:
        merged = ratios.merge(finra, on="symbol", how="left")
    else:
        # Original behavior: only include tickers with FINRA data
        merged = finra.merge(ratios, on="symbol", how="left")

    # Handle FINRA volume - will be NaN for Polygon-only tickers
    merged["finra_off_exchange_volume"] = pd.to_numeric(
        merged.get("off_exchange_volume", pd.Series(dtype=float)), errors="coerce"
    )
    merged["applied_lit_buy_ratio"] = pd.to_numeric(merged["lit_buy_ratio"], errors="coerce")

    # Flag whether FINRA data exists for this ticker
    merged["has_finra_data"] = merged["finra_off_exchange_volume"].notna()

    # Estimated volumes - will be NaN for Polygon-only tickers (no FINRA volume to apply)
    merged["estimated_dark_buy_volume"] = (
        merged["finra_off_exchange_volume"] * merged["applied_lit_buy_ratio"]
    )
    merged["estimated_dark_sell_volume"] = (
        merged["finra_off_exchange_volume"] * (1.0 - merged["applied_lit_buy_ratio"])
    )
    merged["date"] = snapshot_date
    merged["inference_version"] = inference_version
    merged["finra_week_used"] = finra_week

    return merged[
        [
            "symbol",
            "date",
            "finra_off_exchange_volume",
            "estimated_dark_buy_volume",
            "estimated_dark_sell_volume",
            "applied_lit_buy_ratio",
            "inference_version",
            "finra_week_used",
            "has_finra_data",
        ]
    ]


def build_daily_summary(
    estimated_flow_df: pd.DataFrame, finra_period_type: str
) -> pd.DataFrame:
    summary = estimated_flow_df.copy()
    summary["estimated_bought"] = summary["estimated_dark_buy_volume"]
    summary["estimated_sold"] = summary["estimated_dark_sell_volume"]
    summary["total_off_exchange_volume"] = summary["finra_off_exchange_volume"]
    summary["finra_period_type"] = finra_period_type
    # buy_ratio = bought / sold (ratio scale: 1.0 = neutral, >1 = net buying, <1 = net selling)
    summary["buy_ratio"] = summary["estimated_bought"] / summary["estimated_sold"]
    summary.loc[summary["estimated_sold"] <= 0, "buy_ratio"] = pd.NA

    # sell_ratio = sold / bought (inverse of buy_ratio)
    summary["sell_ratio"] = summary["estimated_sold"] / summary["estimated_bought"]
    summary.loc[summary["estimated_bought"] <= 0, "sell_ratio"] = pd.NA

    # finra_week_used and has_finra_data are already in estimated_flow_df from build_darkpool_estimates

    return summary[
        [
            "date",
            "symbol",
            "estimated_bought",
            "estimated_sold",
            "buy_ratio",
            "sell_ratio",
            "total_off_exchange_volume",
            "finra_period_type",
            "finra_week_used",
            "has_finra_data",
        ]
    ]
