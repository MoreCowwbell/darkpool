from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import pandas as pd
import requests

try:
    from .config import Config
    from .db import get_uncached_symbols, mark_ingestion_complete
except ImportError:
    from config import Config
    from db import get_uncached_symbols, mark_ingestion_complete

logger = logging.getLogger(__name__)


def _normalize_daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    columns = {col.lower(): col for col in df.columns}
    symbol_col = columns.get("symbol")
    date_col = columns.get("trade_date") or columns.get("date")
    open_col = columns.get("open") or columns.get("o")
    high_col = columns.get("high") or columns.get("h")
    low_col = columns.get("low") or columns.get("l")
    close_col = columns.get("close") or columns.get("c")
    vwap_col = columns.get("vwap") or columns.get("vw")
    volume_col = columns.get("volume") or columns.get("v")

    if not symbol_col or not date_col or not close_col:
        raise ValueError("Daily aggregates file missing required columns.")

    normalized = pd.DataFrame(
        {
            "symbol": df[symbol_col].astype(str).str.upper(),
            "trade_date": pd.to_datetime(df[date_col], errors="coerce").dt.date,
            "open": pd.to_numeric(df[open_col], errors="coerce") if open_col else pd.NA,
            "high": pd.to_numeric(df[high_col], errors="coerce") if high_col else pd.NA,
            "low": pd.to_numeric(df[low_col], errors="coerce") if low_col else pd.NA,
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "vwap": pd.to_numeric(df[vwap_col], errors="coerce") if vwap_col else pd.NA,
            "volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else pd.NA,
        }
    )
    normalized = normalized.dropna(subset=["symbol", "trade_date", "close"])
    return normalized[
        ["symbol", "trade_date", "open", "high", "low", "close", "vwap", "volume"]
    ]


def _load_daily_agg_file(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "data" in payload:
            df = pd.DataFrame(payload["data"])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError("Unsupported aggregates JSON file structure.")
    else:
        df = pd.read_csv(path)
    return _normalize_daily_agg(df)


def _load_daily_agg_dir(directory: str) -> pd.DataFrame:
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Polygon daily agg dir not found: {directory}")
    frames = []
    for file_path in sorted(path.glob("*.csv")):
        try:
            frames.append(_load_daily_agg_file(str(file_path)))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path.name, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_daily_agg_for_symbol(config: Config, symbol: str, trade_date: date) -> pd.DataFrame:
    if not config.polygon_api_key:
        raise ValueError("POLYGON_API_KEY is required to fetch daily aggregates.")
    url = (
        f"{config.polygon_base_url.rstrip('/')}/v2/aggs/ticker/"
        f"{symbol}/range/1/day/{trade_date.isoformat()}/{trade_date.isoformat()}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": config.polygon_api_key,
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame()
    bar = results[0]
    ts = bar.get("t")
    trade_date_val = (
        datetime.utcfromtimestamp(ts / 1000).date() if ts else trade_date
    )
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date_val,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "vwap": bar.get("vw"),
                "volume": bar.get("v"),
            }
        ]
    )


def fetch_polygon_daily_agg(
    config: Config,
    symbols: list[str],
    trade_date: date,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch Polygon daily aggregates (OHLCV) for price context.

    If conn is provided and config.skip_cached is True, skips symbols that
    are already in polygon_ingestion_state for this date+source.

    Returns:
        Tuple of (DataFrame, cache_stats dict)
        cache_stats: {"cached": int, "fetched": int}
    """
    cache_stats = {"cached": 0, "fetched": 0}
    empty_df = pd.DataFrame(columns=[
        "symbol", "trade_date", "open", "high", "low", "close", "vwap", "volume", "fetch_timestamp"
    ])

    if config.polygon_daily_agg_file:
        logger.info("Loading Polygon daily agg from file: %s", config.polygon_daily_agg_file)
        df = _load_daily_agg_file(config.polygon_daily_agg_file)
        df["fetch_timestamp"] = datetime.utcnow()
    elif config.polygon_daily_agg_dir:
        logger.info("Loading Polygon daily agg from dir: %s", config.polygon_daily_agg_dir)
        df = _load_daily_agg_dir(config.polygon_daily_agg_dir)
        df["fetch_timestamp"] = datetime.utcnow()
    else:
        # Check cache if enabled
        symbols_to_fetch = symbols
        if conn is not None and config.skip_cached:
            symbols_to_fetch = get_uncached_symbols(conn, symbols, trade_date, "daily")
            cache_stats["cached"] = len(symbols) - len(symbols_to_fetch)
            if cache_stats["cached"] > 0:
                logger.info(
                    "Cache hit: %d/%d symbols already fetched for %s (daily agg)",
                    cache_stats["cached"], len(symbols), trade_date.isoformat()
                )
            if not symbols_to_fetch:
                logger.info("All symbols cached for %s (daily agg), skipping fetch", trade_date.isoformat())
                return empty_df, cache_stats

        max_workers = int(os.getenv("POLYGON_MAX_WORKERS", "4"))
        frames = []
        failures = []
        fetched_symbols = []

        def fetch_one(symbol: str) -> Tuple[str, pd.DataFrame | None]:
            try:
                logger.info("Fetching Polygon daily agg for %s on %s", symbol, trade_date.isoformat())
                result = _fetch_daily_agg_for_symbol(config, symbol, trade_date)
                return symbol, result if not result.empty else None
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Polygon daily agg fetch failed for %s: %s", symbol, exc)
                failures.append(symbol)
                return symbol, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, sym): sym for sym in symbols_to_fetch}
            for future in as_completed(futures):
                symbol, result_df = future.result()
                if result_df is not None:
                    frames.append(result_df)
                    fetched_symbols.append(symbol)

        cache_stats["fetched"] = len(fetched_symbols)

        # Update ingestion state for successfully fetched symbols
        if conn is not None and fetched_symbols:
            for symbol in fetched_symbols:
                mark_ingestion_complete(conn, symbol, trade_date, "daily", 1)

        if not frames:
            return empty_df, cache_stats
        df = pd.concat(frames, ignore_index=True)
        df["fetch_timestamp"] = datetime.utcnow()

    if df.empty:
        return df, cache_stats

    df = _normalize_daily_agg(df)
    # Re-add fetch_timestamp after normalization (it gets dropped)
    df["fetch_timestamp"] = datetime.utcnow()
    df = df[df["trade_date"] == trade_date].copy()
    if symbols:
        df = df[df["symbol"].isin([s.upper() for s in symbols])].copy()
    return df, cache_stats
