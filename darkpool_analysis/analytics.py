from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


def build_darkpool_estimates(
    finra_week_df: pd.DataFrame,
    lit_flow_df: pd.DataFrame,
    snapshot_date: date,
    inference_version: str,
    finra_week: date,
    include_polygon_only: bool = True,
) -> pd.DataFrame:
    # Defensive check for empty/malformed lit_flow_df
    if lit_flow_df.empty or "symbol" not in lit_flow_df.columns:
        logger.warning("No lit flow data available - returning empty estimates")
        return pd.DataFrame(columns=[
            "symbol", "date", "finra_off_exchange_volume",
            "estimated_dark_buy_volume", "estimated_dark_sell_volume",
            "applied_lit_buy_ratio", "inference_version", "finra_week_used", "has_finra_data"
        ])

    finra = finra_week_df.copy()
    ratios = lit_flow_df[["symbol", "lit_buy_ratio"]].copy()

    # Handle empty FINRA DataFrame (Polygon-only mode)
    # When FINRA is empty, skip merge and use ratios as base
    if finra.empty:
        logger.info("FINRA data empty - using Polygon-only ratios")
        merged = ratios.copy()
        # No FINRA volume available
        merged["finra_off_exchange_volume"] = pd.NA
    else:
        # Use outer join to include Polygon-only tickers (no FINRA data)
        if include_polygon_only:
            merged = ratios.merge(finra, on="symbol", how="left")
        else:
            # Original behavior: only include tickers with FINRA data
            merged = finra.merge(ratios, on="symbol", how="left")

        # Handle FINRA volume - will be NaN for Polygon-only tickers
        if "off_exchange_volume" in merged.columns:
            merged["finra_off_exchange_volume"] = pd.to_numeric(
                merged["off_exchange_volume"], errors="coerce"
            )
        else:
            merged["finra_off_exchange_volume"] = pd.NA

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

    # For tickers WITH FINRA data: buy_ratio = bought / sold
    # For Polygon-only tickers: use applied_lit_buy_ratio directly
    summary["buy_ratio"] = summary["estimated_bought"] / summary["estimated_sold"]
    summary.loc[summary["estimated_sold"] <= 0, "buy_ratio"] = pd.NA

    # For Polygon-only rows (no FINRA data), convert lit_buy_ratio (proportion 0-1) to buy_ratio
    # lit_buy_ratio = 0.6 means 60% buys → buy_ratio = 0.6/0.4 = 1.5
    polygon_only_mask = ~summary["has_finra_data"]
    lit_ratio = summary.loc[polygon_only_mask, "applied_lit_buy_ratio"]
    summary.loc[polygon_only_mask, "buy_ratio"] = lit_ratio / (1.0 - lit_ratio)

    # sell_ratio = sold / bought (inverse of buy_ratio)
    summary["sell_ratio"] = summary["estimated_sold"] / summary["estimated_bought"]
    summary.loc[summary["estimated_bought"] <= 0, "sell_ratio"] = pd.NA

    # For Polygon-only rows, sell_ratio is inverse of buy_ratio
    summary.loc[polygon_only_mask, "sell_ratio"] = (1.0 - lit_ratio) / lit_ratio

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
