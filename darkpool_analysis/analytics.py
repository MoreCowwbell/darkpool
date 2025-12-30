from __future__ import annotations

from datetime import date, timedelta
import logging
from pathlib import Path
from typing import Optional

import numpy as np
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
        .rename(columns={"total_volume": "short_total_volume"})
    )
    grouped["short_volume_total"] = (
        grouped["short_volume"].fillna(0.0) + grouped["short_exempt_volume"].fillna(0.0)
    )
    grouped["short_ratio_source"] = "CONSOLIDATED"
    return grouped


def _compute_weekly_lit_ratio(lit_df: pd.DataFrame) -> pd.DataFrame:
    if lit_df.empty:
        return lit_df
    df = lit_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["week_start_date"] = df["date"].apply(lambda d: d - timedelta(days=d.weekday()))
    df["lit_buy_volume"] = pd.to_numeric(df["lit_buy_volume"], errors="coerce").fillna(0.0)
    df["lit_sell_volume"] = pd.to_numeric(df["lit_sell_volume"], errors="coerce").fillna(0.0)
    weekly = (
        df.groupby(["symbol", "week_start_date"], as_index=False)
        .agg({"lit_buy_volume": "sum", "lit_sell_volume": "sum"})
    )
    weekly["lit_week_total"] = weekly["lit_buy_volume"] + weekly["lit_sell_volume"]
    weekly["otc_weekly_buy_ratio"] = pd.NA
    ratio_mask = weekly["lit_week_total"] > 0
    weekly.loc[ratio_mask, "otc_weekly_buy_ratio"] = (
        weekly.loc[ratio_mask, "lit_buy_volume"] / weekly.loc[ratio_mask, "lit_week_total"]
    )
    return weekly[["symbol", "week_start_date", "otc_weekly_buy_ratio"]]


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
        .transform(lambda s: _rolling_zscore(s, config.return_z_window, config.zscore_min_periods))
    )
    return df


def _label_pressure_context(
    return_z: Optional[float],
    return_1d: Optional[float],
    short_z: Optional[float],
    config: Config,
) -> str:
    if short_z is None or pd.isna(short_z):
        return "Neutral"
    if return_z is None or pd.isna(return_z):
        return "Neutral"
    if short_z >= config.short_z_high and return_z >= config.return_z_min:
        return "Accumulating"
    if short_z <= config.short_z_low and return_z <= -config.return_z_min:
        return "Distribution"
    return "Neutral"


def _sigmoid(x: float) -> float:
    """Sigmoid function for intensity scaling."""
    return 1.0 / (1.0 + np.exp(-x))


def _compute_composite_score(
    short_z: Optional[float],
    lit_z: Optional[float],
    price_z: Optional[float],
    otc_participation_z: Optional[float],
    config: Config,
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute composite accumulation score with intensity modulation.

    Returns:
        (accumulation_score, accumulation_score_display) where:
        - accumulation_score: [-1, +1] range
        - accumulation_score_display: [0, 100] scale for visualization
    """
    # Handle missing data - require at least short_z
    if short_z is None or pd.isna(short_z):
        return pd.NA, pd.NA

    # Default missing values to 0 (neutral)
    lit_z = lit_z if lit_z is not None and not pd.isna(lit_z) else 0.0
    price_z = price_z if price_z is not None and not pd.isna(price_z) else 0.0
    otc_z = otc_participation_z if otc_participation_z is not None and not pd.isna(otc_participation_z) else 0.0

    # Weighted sum of tanh-transformed z-scores
    # tanh compresses extreme values to [-1, 1] range
    raw_score = (
        config.composite_w_short * np.tanh(short_z * 0.5) +
        config.composite_w_lit * np.tanh(lit_z * 0.5) +
        config.composite_w_price * np.tanh(price_z * 0.3)
    )

    # Intensity modulation: OTC participation modulates (multiplies) the score
    # sigmoid(0) = 0.5, so at z=0, intensity_scale = 0.7 + 0.6*0.5 = 1.0 (neutral)
    # sigmoid(2) ≈ 0.88, so at z=2, intensity_scale ≈ 0.7 + 0.6*0.88 ≈ 1.23
    # sigmoid(-2) ≈ 0.12, so at z=-2, intensity_scale ≈ 0.7 + 0.6*0.12 ≈ 0.77
    intensity_range = config.intensity_scale_max - config.intensity_scale_min
    intensity_scale = config.intensity_scale_min + intensity_range * _sigmoid(otc_z)

    # Apply intensity modulation and clamp to [-1, 1]
    accumulation_score = np.clip(raw_score * intensity_scale, -1.0, 1.0)

    # Display scale: map [-1, +1] to [0, 100]
    accumulation_score_display = (accumulation_score + 1.0) * 50.0

    return float(accumulation_score), float(accumulation_score_display)


def _compute_confidence(
    otc_status: Optional[str],
    has_short: bool,
    has_lit: bool,
    has_price: bool,
    lit_flow_imbalance: Optional[float],
) -> float:
    """
    Compute data quality confidence score.

    Returns:
        confidence: [0.25, 1.0] range where higher = more reliable signal
    """
    base_confidence = 1.0

    # Staleness penalty
    if otc_status == "Anchored":
        staleness_penalty = 1.0
    elif otc_status == "Staled":
        staleness_penalty = 0.7
    else:  # "None" or missing
        staleness_penalty = 0.5

    # Coverage penalty: count missing data sources
    missing_count = sum([
        not has_short,
        not has_lit,
        not has_price,
    ])
    if missing_count == 0:
        coverage_penalty = 1.0
    elif missing_count == 1:
        coverage_penalty = 0.8
    else:  # missing 2+
        coverage_penalty = 0.5

    # Additional penalty if lit_flow_imbalance is NULL (low volume)
    if lit_flow_imbalance is None or pd.isna(lit_flow_imbalance):
        coverage_penalty *= 0.9

    confidence = base_confidence * staleness_penalty * coverage_penalty
    return max(0.25, min(1.0, confidence))


def _label_from_score(score_display: Optional[float]) -> str:
    """
    Convert accumulation_score_display (0-100) to label.

    Thresholds: <30 = Distribution, 30-70 = Neutral, >70 = Accumulating
    """
    if score_display is None or pd.isna(score_display):
        return "Neutral"
    if score_display > 70:
        return "Accumulating"
    if score_display < 30:
        return "Distribution"
    return "Neutral"


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
    otc_weekly = _aggregate_finra_weekly(finra_weekly_df)
    short_daily = _aggregate_short_raw(short_raw_df)
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
    # Normalize date column to datetime64 for consistent merging with DB-sourced DataFrames
    daily_base["date"] = pd.to_datetime(daily_base["date"])

    # Normalize lit_df date column if not empty
    if not lit_df.empty and "date" in lit_df.columns:
        lit_df = lit_df.copy()
        lit_df["date"] = pd.to_datetime(lit_df["date"])

    # Normalize short_daily and price_context trade_date columns
    if not short_daily.empty and "trade_date" in short_daily.columns:
        short_daily = short_daily.copy()
        short_daily["trade_date"] = pd.to_datetime(short_daily["trade_date"])
    if not price_context.empty and "trade_date" in price_context.columns:
        price_context = price_context.copy()
        price_context["trade_date"] = pd.to_datetime(price_context["trade_date"])

    lit_cols = [
        "symbol",
        "date",
        "log_buy_sell",
        "lit_buy_volume",
        "lit_sell_volume",
        "lit_buy_ratio",
    ]
    merged = daily_base.merge(lit_df[lit_cols], on=["symbol", "date"], how="left")
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
        merged["otc_week_used"] = pd.NaT
    else:
        mapped = []
        for symbol, group in merged.groupby("symbol"):
            weekly = otc_weekly[otc_weekly["symbol"] == symbol].copy()
            weekly = weekly.sort_values("week_start_date")
            weekly["week_start_date"] = pd.to_datetime(weekly["week_start_date"])
            # Drop symbol to avoid duplicate column (symbol_x, symbol_y) after merge_asof
            weekly = weekly.drop(columns=["symbol"])
            if weekly.empty:
                group["otc_off_exchange_volume"] = pd.NA
                group["otc_week_used"] = pd.NaT
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

    merged["date"] = pd.to_datetime(merged["date"])
    if "otc_week_used" in merged.columns:
        merged["otc_week_used"] = pd.to_datetime(merged["otc_week_used"])
    if not otc_weekly.empty:
        latest_week = pd.to_datetime(otc_weekly["week_start_date"].max())
        latest_symbols = set(
            otc_weekly.loc[otc_weekly["week_start_date"] == latest_week, "symbol"]
        )
        missing_latest = (merged["date"] >= latest_week) & (
            ~merged["symbol"].isin(latest_symbols)
        )
        merged.loc[missing_latest, ["otc_off_exchange_volume", "otc_week_used"]] = pd.NA

    # Weekly lit buy ratio proxy for OTC weekly buy ratio
    weekly_lit = _compute_weekly_lit_ratio(lit_df)
    if not weekly_lit.empty and "otc_week_used" in merged.columns:
        weekly_lit = weekly_lit.rename(columns={"week_start_date": "otc_week_used"})
        weekly_lit["otc_week_used"] = pd.to_datetime(weekly_lit["otc_week_used"])
        merged = merged.merge(
            weekly_lit,
            on=["symbol", "otc_week_used"],
            how="left",
        )
    else:
        merged["otc_weekly_buy_ratio"] = pd.NA

    # Short ratio denominator logic
    merged["short_buy_volume"] = pd.to_numeric(merged["short_volume"], errors="coerce")
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

    merged["short_ratio_denominator_value"] = pd.to_numeric(
        merged["short_ratio_denominator_value"], errors="coerce"
    )
    denom_ready = merged["short_ratio_denominator_value"].notna()
    merged.loc[denom_ready, "short_ratio"] = (
        merged.loc[denom_ready, "short_buy_volume"]
        / merged.loc[denom_ready, "short_ratio_denominator_value"]
    )
    merged["short_sell_volume"] = pd.NA
    merged.loc[denom_ready, "short_sell_volume"] = (
        merged.loc[denom_ready, "short_ratio_denominator_value"]
        - merged.loc[denom_ready, "short_buy_volume"]
    )
    merged["short_sell_volume"] = pd.to_numeric(merged["short_sell_volume"], errors="coerce")
    merged.loc[denom_ready, "short_sell_volume"] = merged.loc[
        denom_ready, "short_sell_volume"
    ].clip(lower=0.0)

    merged["short_buy_sell_ratio"] = pd.NA
    valid_sell = merged["short_sell_volume"].notna() & (merged["short_sell_volume"] > 0)
    merged.loc[valid_sell, "short_buy_sell_ratio"] = (
        merged.loc[valid_sell, "short_buy_volume"] / merged.loc[valid_sell, "short_sell_volume"]
    )

    # Ensure numeric dtype for rolling calculations (pd.NA -> np.nan)
    merged["short_ratio"] = pd.to_numeric(merged["short_ratio"], errors="coerce")
    merged["short_buy_sell_ratio"] = pd.to_numeric(merged["short_buy_sell_ratio"], errors="coerce")

    merged["short_ratio_z"] = (
        merged.sort_values(["symbol", "date"])
        .groupby("symbol")["short_ratio"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )
    merged["short_buy_sell_ratio_z"] = (
        merged.sort_values(["symbol", "date"])
        .groupby("symbol")["short_buy_sell_ratio"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )

    merged["lit_total_volume"] = merged["lit_buy_volume"] + merged["lit_sell_volume"]
    merged["lit_buy_ratio"] = pd.to_numeric(merged["lit_buy_ratio"], errors="coerce")
    merged["lit_buy_ratio_z"] = (
        merged.sort_values(["symbol", "date"])
        .groupby("symbol")["lit_buy_ratio"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )

    # Lit flow imbalance: bounded [-1, +1], symmetric around 0
    # Formula: (buy - sell) / (buy + sell)
    merged["lit_flow_imbalance"] = pd.NA
    lit_total = pd.to_numeric(merged["lit_buy_volume"], errors="coerce").fillna(0) + \
                pd.to_numeric(merged["lit_sell_volume"], errors="coerce").fillna(0)
    sufficient_volume = lit_total >= config.min_lit_volume
    merged.loc[sufficient_volume, "lit_flow_imbalance"] = (
        (merged.loc[sufficient_volume, "lit_buy_volume"] - merged.loc[sufficient_volume, "lit_sell_volume"])
        / lit_total[sufficient_volume]
    )
    merged["lit_flow_imbalance"] = pd.to_numeric(merged["lit_flow_imbalance"], errors="coerce")
    merged["lit_flow_imbalance_z"] = (
        merged.sort_values(["symbol", "date"])
        .groupby("symbol")["lit_flow_imbalance"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )

    merged["otc_buy_volume"] = pd.NA
    merged["otc_sell_volume"] = pd.NA
    if "otc_weekly_buy_ratio" in merged.columns:
        merged["otc_buy_volume"] = merged["otc_off_exchange_volume"] * merged["otc_weekly_buy_ratio"]
        merged["otc_sell_volume"] = merged["otc_off_exchange_volume"] - merged["otc_buy_volume"]
        merged["otc_sell_volume"] = pd.to_numeric(merged["otc_sell_volume"], errors="coerce")
        merged["otc_sell_volume"] = merged["otc_sell_volume"].clip(lower=0.0)

    merged["otc_weekly_buy_ratio"] = pd.to_numeric(
        merged.get("otc_weekly_buy_ratio"), errors="coerce"
    )
    merged["otc_buy_ratio_z"] = (
        merged.sort_values(["symbol", "date"])
        .groupby("symbol")["otc_weekly_buy_ratio"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
    )

    # OTC Participation Rate: measures institutional activity intensity (not direction)
    # weekly_total_volume = sum of daily Polygon volumes for the OTC week
    # otc_participation_rate = otc_weekly_volume / weekly_total_volume
    merged["weekly_total_volume"] = pd.NA
    merged["otc_participation_rate"] = pd.NA
    merged["otc_participation_z"] = pd.NA
    merged["otc_participation_delta"] = pd.NA

    if "otc_week_used" in merged.columns and "volume" in merged.columns:
        # Compute weekly total volume by summing daily Polygon volumes for each (symbol, week)
        merged["volume"] = pd.to_numeric(merged["volume"], errors="coerce")
        weekly_vol = (
            merged.groupby(["symbol", "otc_week_used"])["volume"]
            .transform("sum")
        )
        merged["weekly_total_volume"] = weekly_vol

        # Compute participation rate
        has_weekly_total = merged["weekly_total_volume"].notna() & (merged["weekly_total_volume"] > 0)
        has_otc_vol = merged["otc_off_exchange_volume"].notna()
        valid_participation = has_weekly_total & has_otc_vol
        merged.loc[valid_participation, "otc_participation_rate"] = (
            merged.loc[valid_participation, "otc_off_exchange_volume"]
            / merged.loc[valid_participation, "weekly_total_volume"]
        )

        # Per-ticker rolling z-score (12-week window ~ 60 trading days)
        merged["otc_participation_rate"] = pd.to_numeric(merged["otc_participation_rate"], errors="coerce")
        merged["otc_participation_z"] = (
            merged.sort_values(["symbol", "date"])
            .groupby("symbol")["otc_participation_rate"]
            .transform(lambda s: _rolling_zscore(s, window=60, min_periods=5))
        )

        # Week-over-week delta: change in participation rate from prior week
        # Get unique (symbol, week) participation rates and compute delta
        weekly_rates = merged.drop_duplicates(subset=["symbol", "otc_week_used"])[
            ["symbol", "otc_week_used", "otc_participation_rate"]
        ].copy()
        weekly_rates = weekly_rates.sort_values(["symbol", "otc_week_used"])
        weekly_rates["prior_rate"] = weekly_rates.groupby("symbol")["otc_participation_rate"].shift(1)
        weekly_rates["otc_participation_delta"] = (
            weekly_rates["otc_participation_rate"] - weekly_rates["prior_rate"]
        )
        # Merge delta back to main dataframe
        delta_map = weekly_rates.set_index(["symbol", "otc_week_used"])["otc_participation_delta"].to_dict()
        merged["otc_participation_delta"] = merged.apply(
            lambda row: delta_map.get((row["symbol"], row.get("otc_week_used"))), axis=1
        )

    merged["has_otc"] = merged["otc_off_exchange_volume"].notna()
    merged["has_short"] = merged["short_ratio"].notna()
    merged["has_lit"] = merged["lit_buy_ratio"].notna()
    merged["has_price"] = merged["close"].notna()

    merged["data_quality"] = "PRE_OTC"
    if "otc_week_used" in merged.columns:
        week_end = merged["otc_week_used"] + pd.to_timedelta(4, unit="D")
        anchored = merged["has_otc"] & merged["date"].between(
            merged["otc_week_used"], week_end
        )
        merged.loc[anchored, "data_quality"] = "OTC_ANCHORED"

    merged["otc_status"] = "None"
    merged.loc[merged["has_otc"], "otc_status"] = "Staled"
    if "otc_week_used" in merged.columns:
        merged.loc[anchored, "otc_status"] = "Anchored"

    # Compute composite accumulation score and confidence
    def _apply_composite(row):
        score, score_display = _compute_composite_score(
            short_z=row.get("short_buy_sell_ratio_z"),
            lit_z=row.get("lit_flow_imbalance_z"),
            price_z=row.get("return_z"),
            otc_participation_z=row.get("otc_participation_z"),
            config=config,
        )
        return pd.Series({"accumulation_score": score, "accumulation_score_display": score_display})

    composite_scores = merged.apply(_apply_composite, axis=1)
    merged["accumulation_score"] = composite_scores["accumulation_score"]
    merged["accumulation_score_display"] = composite_scores["accumulation_score_display"]

    merged["confidence"] = merged.apply(
        lambda row: _compute_confidence(
            otc_status=row.get("otc_status"),
            has_short=bool(row.get("has_short")),
            has_lit=bool(row.get("has_lit")),
            has_price=bool(row.get("has_price")),
            lit_flow_imbalance=row.get("lit_flow_imbalance"),
        ),
        axis=1,
    )

    # Update pressure_context_label to use new score thresholds
    merged["pressure_context_label"] = merged["accumulation_score_display"].apply(_label_from_score)
    merged["inference_version"] = config.inference_version

    output = merged[
        [
            "date",
            "symbol",
            "log_buy_sell",
            "short_volume",
            "short_exempt_volume",
            "short_total_volume",
            "short_buy_volume",
            "short_sell_volume",
            "short_ratio",
            "short_ratio_z",
            "short_buy_sell_ratio",
            "short_buy_sell_ratio_z",
            "short_ratio_denominator_type",
            "short_ratio_denominator_value",
            "short_ratio_source",
            "lit_buy_volume",
            "lit_sell_volume",
            "lit_total_volume",
            "lit_buy_ratio",
            "lit_buy_ratio_z",
            "lit_flow_imbalance",
            "lit_flow_imbalance_z",
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
            "otc_weekly_buy_ratio",
            "otc_buy_volume",
            "otc_sell_volume",
            "otc_buy_ratio_z",
            "weekly_total_volume",
            "otc_participation_rate",
            "otc_participation_z",
            "otc_participation_delta",
            "otc_status",
            "data_quality",
            "has_otc",
            "has_short",
            "has_lit",
            "has_price",
            "accumulation_score",
            "accumulation_score_display",
            "confidence",
            "pressure_context_label",
            "inference_version",
        ]
    ].copy()

    # Convert date column to date objects for comparison (avoids FutureWarning with datetime64)
    output = output[pd.to_datetime(output["date"]).dt.date.isin(all_dates)].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.date
    if "otc_week_used" in output.columns:
        output["otc_week_used"] = pd.to_datetime(output["otc_week_used"]).dt.date
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
            total_short = subset["short_buy_volume"].fillna(0).sum()
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
    # Ensure numeric dtype for rolling calculations
    agg_df["agg_short_ratio"] = pd.to_numeric(agg_df["agg_short_ratio"], errors="coerce")
    agg_df["agg_short_ratio_z"] = (
        agg_df.groupby("index_symbol")["agg_short_ratio"]
        .transform(lambda s: _rolling_zscore(s, config.short_z_window, config.zscore_min_periods))
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
