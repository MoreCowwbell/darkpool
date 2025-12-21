from __future__ import annotations

from datetime import date
import logging
from typing import Iterable

import pandas as pd
import pytz

from config import Config

logger = logging.getLogger(__name__)


def _placeholder_rows(symbols: Iterable[str], trade_date: date) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "symbol": symbol,
                "date": trade_date,
                "lit_buy_volume": 0.0,
                "lit_sell_volume": 0.0,
                "lit_buy_ratio": pd.NA,
                "classification_method": "TICK",
                "lit_coverage_pct": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _classify_group(group: pd.DataFrame, config: Config) -> dict:
    group = group.sort_values("timestamp").copy()
    nbbo_mask = (
        group["bid"].notna()
        & group["ask"].notna()
        & (group["bid"] > 0)
        & (group["ask"] > 0)
        & (group["ask"] >= group["bid"])
    )
    method = "NBBO" if len(group) > 0 and nbbo_mask.all() else "TICK"

    if method == "NBBO":
        buy_mask = group["price"] >= group["ask"]
        sell_mask = group["price"] <= group["bid"]
    else:
        price = group["price"]
        prev_price = price.shift(1)
        buy_mask = price > prev_price
        sell_mask = price < prev_price

    buy_volume = float(group.loc[buy_mask, "size"].sum())
    sell_volume = float(group.loc[sell_mask, "size"].sum())
    classified_count = int(buy_mask.sum() + sell_mask.sum())
    coverage_pct = 0.0
    if len(group) > 0:
        coverage_pct = round(100.0 * classified_count / len(group), 4)

    total_volume = buy_volume + sell_volume
    if total_volume < config.min_lit_volume:
        lit_buy_ratio = pd.NA
    else:
        lit_buy_ratio = buy_volume / total_volume if total_volume else pd.NA

    return {
        "lit_buy_volume": buy_volume,
        "lit_sell_volume": sell_volume,
        "lit_buy_ratio": lit_buy_ratio,
        "classification_method": method,
        "lit_coverage_pct": coverage_pct,
    }


def compute_lit_directional_flow(
    trades_df: pd.DataFrame,
    expected_symbols: Iterable[str],
    trade_date: date,
    config: Config,
) -> pd.DataFrame:
    if trades_df.empty:
        logger.warning("No Polygon trades available; lit flow will be empty.")
        return _placeholder_rows(expected_symbols, trade_date)

    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    market_tz = pytz.timezone(config.market_tz)
    local_ts = df["timestamp"].dt.tz_convert(market_tz)
    df["trade_date"] = local_ts.dt.date
    df = df[df["trade_date"] == trade_date].copy()
    if df.empty:
        logger.warning("No trades matched target date after timezone conversion.")
        return _placeholder_rows(expected_symbols, trade_date)

    local_ts = df["timestamp"].dt.tz_convert(market_tz)
    rth_mask = (local_ts.dt.time >= config.rth_start) & (local_ts.dt.time <= config.rth_end)
    df = df[rth_mask].copy()
    if df.empty:
        logger.warning("No trades in regular trading hours for %s.", trade_date)
        return _placeholder_rows(expected_symbols, trade_date)

    results = []
    for symbol in expected_symbols:
        group = df[df["symbol"] == symbol]
        if group.empty:
            results.append(
                {
                    "symbol": symbol,
                    "date": trade_date,
                    "lit_buy_volume": 0.0,
                    "lit_sell_volume": 0.0,
                    "lit_buy_ratio": pd.NA,
                    "classification_method": "TICK",
                    "lit_coverage_pct": 0.0,
                }
            )
            continue
        classified = _classify_group(group, config)
        results.append(
            {
                "symbol": symbol,
                "date": trade_date,
                **classified,
            }
        )

    return pd.DataFrame(results)
