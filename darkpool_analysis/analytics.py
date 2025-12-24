from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from .config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    z = (series - mean) / std
    return z


def _aggregate_finra_weekly(finra_weekly_df: pd.DataFrame) -> pd.DataFrame:
    if finra_weekly_df.empty:
        return finra_weekly_df
    grouped = (
        finra_weekly_df.groupby(["symbol", "week_start_date"], as_index=False)
        .agg({"off_exchange_volume": "sum"})
        .rename(columns={"off_exchange_volume": "otc_off_exchange_volume"})
    )
    return grouped


def _aggregate_short_raw(
    short_raw_df: pd.DataFrame, preferred_source: str
) -> pd.DataFrame:
    if short_raw_df.empty:
        return short_raw_df
    df = short_raw_df.copy()
    df["short_exempt_volume"] = df["short_exempt_volume"].fillna(0.0)
    df["short_volume_total"] = df["short_volume"] + df["short_exempt_volume"]

    has_preferred = df["source"].fillna("").str.upper() == preferred_source
    preferred_df = df[has_preferred].copy()
    if not preferred_df.empty:
        logger.info("Using preferred short sale source: %s", preferred_source)
        source_df = preferred_df
        source_label = preferred_source
    else:
        source_df = df
        source_label = "SUM"

    grouped = (
        source_df.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            {
                "short_volume": "sum",
                "short_exempt_volume": "sum",
                "short_volume_total": "sum",
                "total_volume": "sum",
            }
        )
        .rename(columns={"total_volume": "short_total_volume"})
    )
    grouped["short_ratio_source"] = source_label
    return grouped


def _compute_price_context(
    daily_agg_df: pd.DataFrame, config: Config
) -> pd.DataFrame:
    if daily_agg_df.empty:
        return daily_agg_df
    df = daily_agg_df.copy()
    df = df.sort_values(["symbol", "trade_date"])
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["return_1d"] = df["close"] / df["prev_close"] - 1.0
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["return_z"] = (
        df.groupby("symbol")["return_1d"]
        .apply(lambda s: _rolling_zscore(s, config.return_z_window, config.zscore_min_periods))
        .reset_index(level=0, drop=True)
    )
    return df


def _label_pressure_context(
    return_z: Optional[float],
    return_1d: Optional[float],
    short_z: Optional[float],
    config: Config,
) -> str:
    if short_z is None or pd.isna(short_z):
        return "NEUTRAL"

    if return_z is not None and not pd.isna(return_z):
        if abs(return_z) < config.return_z_min:
            return "NEUTRAL"
        return_sign = 1 if return_z > 0 else -1
    elif return_1d is not None and not pd.isna(return_1d):
        return_sign = 1 if return_1d > 0 else -1
    else:
        return "NEUTRAL"

    if short_z >= config.short_z_high and return_sign > 0:
        return "SHORT_INTO_STRENGTH"
    if short_z >= config.short_z_high and return_sign < 0:
        return "SHORT_ON_WEAKNESS"
    if short_z <= config.short_z_low and return_sign > 0:
        return "LOW_SHORT_STRONG_UP"
    if short_z <= config.short_z_low and return_sign < 0:
        return "LOW_SHORT_SELL_OFF"
    return "NEUTRAL"


def build_daily_metrics(
    finra_weekly_df: pd.DataFrame,
    short_raw_df: pd.DataFrame,
    daily_agg_df: pd.DataFrame,
    lit_df: pd.DataFrame,
    target_dates: list[date],
    config: Config,
) -> pd.DataFrame:
    if not target_dates:
        return pd.DataFrame()

    all_dates = pd.to_datetime(pd.Series(target_dates)).dt.date.unique().tolist()
    start_date = min(all_dates)
    end_date = max(all_dates)

    otc_weekly = _aggregate_finra_weekly(finra_weekly_df)
    short_daily = _aggregate_short_raw(short_raw_df, config.short_sale_preferred_source)
    price_context = _compute_price_context(daily_agg_df, config)

    if short_daily.empty:
        short_daily = pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "short_volume",
                "short_exempt_volume",
                "short_volume_total",
                "short_total_volume",
                "short_ratio_source",
            ]
        )

    if price_context.empty:
        price_context = pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
                "return_1d",
                "return_z",
                "range_pct",
            ]
        )

    daily_base = pd.DataFrame(
        [(symbol, d) for symbol in config.tickers for d in all_dates],
        columns=["symbol", "date"],
    )

    merged = daily_base.merge(
        lit_df[["symbol", "date", "log_buy_sell"]],
        on=["symbol", "date"],
        how="left",
    )
    merged = merged.merge(
        short_daily.rename(columns={"trade_date": "date"}),
        on=["symbol", "date"],
        how="left",
    )
    merged = merged.merge(
        price_context.rename(columns={"trade_date": "date"}),
        on=["symbol", "date"],
        how="left",
    )

    # OTC weekly mapping (latest week_start_date <= date)
    if otc_weekly.empty:
        merged["otc_off_exchange_volume"] = pd.NA
        merged["otc_week_used"] = pd.NA
    else:
        mapped = []
        for symbol, group in merged.groupby("symbol"):
            weekly = otc_weekly[otc_weekly["symbol"] == symbol].copy()
            weekly = weekly.sort_values("week_start_date")
            weekly["week_start_date"] = pd.to_datetime(weekly["week_start_date"])
            if weekly.empty:
                group["otc_off_exchange_volume"] = pd.NA
                group["otc_week_used"] = pd.NA
                mapped.append(group)
                continue
            daily_sorted = group.sort_values("date")
            daily_sorted["date"] = pd.to_datetime(daily_sorted["date"])
            aligned = pd.merge_asof(
                daily_sorted,
                weekly,
                left_on="date",
                right_on="week_start_date",
                direction="backward",
            )
            aligned = aligned.rename(columns={"week_start_date": "otc_week_used"})
            mapped.append(aligned)
        merged = pd.concat(mapped, ignore_index=True)

    # Short ratio denominator logic
    merged["short_volume_total"] = (
        merged["short_volume"].fillna(0) + merged["short_exempt_volume"].fillna(0)
    )
    merged["short_ratio_denominator_type"] = pd.NA
    merged["short_ratio_denominator_value"] = pd.NA
    merged["short_ratio"] = pd.NA

    has_finra_total = merged["short_total_volume"].notna() & (merged["short_total_volume"] > 0)
    merged.loc[has_finra_total, "short_ratio_denominator_type"] = "FINRA_TOTAL"
    merged.loc[has_finra_total, "short_ratio_denominator_value"] = merged.loc[
        has_finra_total, "short_total_volume"
    ]

    needs_polygon_total = ~has_finra_total & merged["volume"].notna() & (merged["volume"] > 0)
    merged.loc[needs_polygon_total, "short_ratio_denominator_type"] = "POLYGON_TOTAL"
    merged.loc[needs_polygon_total, "short_ratio_denominator_value"] = merged.loc[
        needs_polygon_total, "volume"
    ]

    denom_ready = merged["short_ratio_denominator_value"].notna()
    merged.loc[denom_ready, "short_ratio"] = (
        merged.loc[denom_ready, "short_volume_total"]
        / merged.loc[denom_ready, "short_ratio_denominator_value"]
    )

    merged["short_ratio_z"] = (
        merged.sort_values(["symbol", "date"])
        .groupby("symbol")["short_ratio"]
        .apply(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
        .reset_index(level=0, drop=True)
    )

    merged["has_otc"] = merged["otc_off_exchange_volume"].notna()
    merged["has_short"] = merged["short_ratio"].notna()
    merged["has_lit"] = merged["log_buy_sell"].notna()
    merged["has_price"] = merged["close"].notna()

    merged["data_quality"] = "PRE_OTC"
    if "otc_week_used" in merged.columns:
        week_end = merged["otc_week_used"] + pd.to_timedelta(4, unit="D")
        anchored = merged["has_otc"] & merged["date"].between(
            merged["otc_week_used"], week_end
        )
        merged.loc[anchored, "data_quality"] = "OTC_ANCHORED"

    merged["pressure_context_label"] = merged.apply(
        lambda row: _label_pressure_context(
            row.get("return_z"),
            row.get("return_1d"),
            row.get("short_ratio_z"),
            config,
        ),
        axis=1,
    )
    merged["inference_version"] = config.inference_version

    output = merged[
        [
            "date",
            "symbol",
            "log_buy_sell",
            "short_volume",
            "short_exempt_volume",
            "short_total_volume",
            "short_ratio",
            "short_ratio_z",
            "short_ratio_denominator_type",
            "short_ratio_denominator_value",
            "short_ratio_source",
            "close",
            "vwap",
            "high",
            "low",
            "volume",
            "return_1d",
            "return_z",
            "range_pct",
            "otc_off_exchange_volume",
            "otc_week_used",
            "data_quality",
            "has_otc",
            "has_short",
            "has_lit",
            "has_price",
            "pressure_context_label",
            "inference_version",
        ]
    ].copy()

    output = output[output["date"].isin(all_dates)].copy()
    return output


def _load_constituents_from_file(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    columns = {col.lower(): col for col in df.columns}
    index_col = columns.get("index_symbol") or columns.get("index")
    symbol_col = columns.get("symbol") or columns.get("ticker")
    if not index_col or not symbol_col:
        raise ValueError("Constituent file requires index_symbol and symbol columns.")
    df = df[[index_col, symbol_col]].copy()
    df[index_col] = df[index_col].astype(str).str.upper()
    df[symbol_col] = df[symbol_col].astype(str).str.upper()
    grouped = df.groupby(index_col)[symbol_col].apply(list).to_dict()
    return grouped


def load_index_constituents(config: Config) -> dict[str, list[str]]:
    if config.index_constituents_file:
        path = Path(config.index_constituents_file)
        return _load_constituents_from_file(path)
    if config.index_constituents_dir.exists():
        frames = {}
        for csv_path in sorted(config.index_constituents_dir.glob("*.csv")):
            try:
                frames.update(_load_constituents_from_file(csv_path))
            except Exception as exc:
                logger.warning("Failed to load constituents from %s: %s", csv_path.name, exc)
        return frames
    return {}


def build_index_constituent_short_agg(
    daily_metrics_df: pd.DataFrame,
    price_context_df: pd.DataFrame,
    target_dates: list[date],
    config: Config,
) -> pd.DataFrame:
    constituents = load_index_constituents(config)
    if not constituents:
        logger.warning("No index constituents provided; skipping index aggregation.")
        return pd.DataFrame()

    daily = daily_metrics_df.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    price_context_df = price_context_df.copy()
    price_context_df["trade_date"] = pd.to_datetime(price_context_df["trade_date"]).dt.date

    rows = []
    for index_symbol, members in constituents.items():
        members = [m.upper() for m in members]
        expected_count = len(members)
        for trade_date in target_dates:
            subset = daily[(daily["symbol"].isin(members)) & (daily["date"] == trade_date)]
            coverage_count = int(subset["has_short"].sum()) if not subset.empty else 0
            total_short = (
                subset["short_volume"].fillna(0).sum()
                + subset["short_exempt_volume"].fillna(0).sum()
            )
            total_denom = subset["short_ratio_denominator_value"].sum(min_count=1)

            denom_types = subset["short_ratio_denominator_type"].dropna().unique().tolist()
            if len(denom_types) == 1:
                denom_type = denom_types[0]
            elif len(denom_types) > 1:
                denom_type = "MIXED"
            else:
                denom_type = pd.NA

            agg_short_ratio = (
                total_short / total_denom if total_denom and not pd.isna(total_denom) else pd.NA
            )
            coverage_pct = (
                coverage_count / expected_count if expected_count else pd.NA
            )

            proxy_symbol = config.index_proxy_map.get(index_symbol.upper())
            price_symbol = index_symbol.upper()
            price_source = "INDEX"
            price_row = price_context_df[
                (price_context_df["symbol"] == price_symbol)
                & (price_context_df["trade_date"] == trade_date)
            ]
            if price_row.empty and proxy_symbol:
                price_symbol = proxy_symbol.upper()
                price_source = "PROXY"
                price_row = price_context_df[
                    (price_context_df["symbol"] == price_symbol)
                    & (price_context_df["trade_date"] == trade_date)
                ]

            index_return = price_row["return_1d"].iloc[0] if not price_row.empty else pd.NA
            index_return_z = price_row["return_z"].iloc[0] if not price_row.empty else pd.NA

            rows.append(
                {
                    "index_symbol": index_symbol.upper(),
                    "trade_date": trade_date,
                    "total_short_volume": total_short,
                    "total_denominator_volume": total_denom,
                    "denominator_type": denom_type,
                    "agg_short_ratio": agg_short_ratio,
                    "coverage_count": coverage_count,
                    "expected_constituent_count": expected_count,
                    "coverage_pct": coverage_pct,
                    "index_price_symbol": price_symbol,
                    "index_price_source": price_source,
                    "index_price_return": index_return,
                    "index_return_z": index_return_z,
                }
            )

    if not rows:
        return pd.DataFrame()

    agg_df = pd.DataFrame(rows)
    agg_df = agg_df.sort_values(["index_symbol", "trade_date"])
    agg_df["agg_short_ratio_z"] = (
        agg_df.groupby("index_symbol")["agg_short_ratio"]
        .apply(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
        .reset_index(level=0, drop=True)
    )
    agg_df["interpretation_label"] = agg_df.apply(
        lambda row: _label_pressure_context(
            row.get("index_return_z"),
            row.get("index_price_return"),
            row.get("agg_short_ratio_z"),
            config,
        ),
        axis=1,
    )

    agg_df["data_quality"] = "PRE_OTC"
    agg_df["inference_version"] = config.inference_version

    return agg_df[
        [
            "index_symbol",
            "trade_date",
            "total_short_volume",
            "total_denominator_volume",
            "denominator_type",
            "agg_short_ratio",
            "agg_short_ratio_z",
            "coverage_count",
            "expected_constituent_count",
            "coverage_pct",
            "index_price_symbol",
            "index_price_source",
            "index_price_return",
            "index_return_z",
            "interpretation_label",
            "data_quality",
            "inference_version",
        ]
    ].copy()
