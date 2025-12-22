from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from typing import Tuple

import pandas as pd
import requests

try:
    from .config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna()
        if sample.empty:
            return pd.to_datetime(series, unit="ms", utc=True)
        max_value = sample.max()
        if max_value > 1e17:
            unit = "ns"
        elif max_value > 1e14:
            unit = "us"
        elif max_value > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(series, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")


def _normalize_trades_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ["symbol", "timestamp", "price", "size"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required trade column: {col}")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    if "bid" not in df.columns:
        df["bid"] = pd.NA
    if "ask" not in df.columns:
        df["ask"] = pd.NA
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["symbol", "timestamp", "price", "size"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def _load_trades_from_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "data" in payload:
            df = pd.DataFrame(payload["data"])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError("Unsupported trades JSON file structure.")
    else:
        df = pd.read_csv(path)
    return _normalize_trades_dataframe(df)


def _build_time_params(trade_date: date, fmt: str) -> dict:
    start = datetime.combine(trade_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    if fmt == "unix_ms":
        return {
            "timestamp.gte": int(start.timestamp() * 1000),
            "timestamp.lt": int(end.timestamp() * 1000),
        }
    if fmt == "unix_ns":
        return {
            "timestamp.gte": int(start.timestamp() * 1_000_000_000),
            "timestamp.lt": int(end.timestamp() * 1_000_000_000),
        }
    return {
        "timestamp.gte": start.isoformat().replace("+00:00", "Z"),
        "timestamp.lt": end.isoformat().replace("+00:00", "Z"),
    }


def _fetch_aggregates_for_symbol(
    config: Config, symbol: str, trade_date: date
) -> pd.DataFrame:
    """Fetch minute aggregates as fallback when trades endpoint is unavailable."""
    if not config.polygon_api_key:
        raise ValueError("POLYGON_API_KEY is required to fetch Polygon data.")

    url = f"{config.polygon_base_url.rstrip('/')}/v2/aggs/ticker/{symbol}/range/1/minute/{trade_date.isoformat()}/{trade_date.isoformat()}"
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
        return pd.DataFrame(columns=["symbol", "timestamp", "price", "size", "bid", "ask"])

    records = []
    for bar in results:
        # Use VWAP as representative price, volume as size
        # Estimate directional flow: if close > open, bullish; else bearish
        open_price = bar.get("o", 0)
        close_price = bar.get("c", 0)
        high_price = bar.get("h", 0)
        low_price = bar.get("l", 0)
        volume = bar.get("v", 0)
        vwap = bar.get("vw", close_price)
        timestamp = bar.get("t")  # Unix ms

        if not timestamp or not volume:
            continue

        # Estimate bid/ask from bar range for directional inference
        # If close > open: price action was bullish, use close as "ask" hit
        # If close < open: price action was bearish, use close as "bid" hit
        if close_price >= open_price:
            # Bullish bar - treat as buy at ask
            bid = low_price
            ask = close_price
        else:
            # Bearish bar - treat as sell at bid
            bid = close_price
            ask = high_price

        records.append({
            "symbol": symbol,
            "timestamp": timestamp,
            "price": vwap,
            "size": volume,
            "bid": bid,
            "ask": ask,
        })

    df = pd.DataFrame(records)
    return _normalize_trades_dataframe(df)


def _fetch_trades_for_symbol(
    config: Config, symbol: str, trade_date: date
) -> pd.DataFrame:
    if not config.polygon_api_key:
        raise ValueError("POLYGON_API_KEY is required to fetch Polygon trades.")
    url_template = config.polygon_base_url.rstrip("/") + "/v3/trades/{symbol}"
    time_params = _build_time_params(
        trade_date, os.getenv("POLYGON_TIMESTAMP_FORMAT", "iso8601")
    )
    params = {
        **time_params,
        "limit": int(os.getenv("POLYGON_LIMIT", "50000")),
        "apiKey": config.polygon_api_key,
    }

    records = []
    url = url_template.format(symbol=symbol)
    max_pages = int(os.getenv("POLYGON_MAX_PAGES", "200"))
    pages = 0
    while url and pages < max_pages:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        for item in results:
            record = {
                "symbol": symbol,
                "timestamp": item.get("sip_timestamp")
                or item.get("participant_timestamp")
                or item.get("timestamp")
                or item.get("t"),
                "price": item.get("price") or item.get("p"),
                "size": item.get("size") or item.get("s"),
                "bid": item.get("bid") or item.get("b"),
                "ask": item.get("ask") or item.get("a"),
            }
            if record["timestamp"] is None or record["price"] is None or record["size"] is None:
                continue
            records.append(record)
        next_url = payload.get("next_url")
        if not next_url:
            break
        url = next_url
        params = {"apiKey": config.polygon_api_key}
        pages += 1

    if not records:
        return pd.DataFrame(columns=["symbol", "timestamp", "price", "size", "bid", "ask"])
    df = pd.DataFrame(records)
    return _normalize_trades_dataframe(df)


def fetch_polygon_trades(
    config: Config, symbols: list[str], trade_date: date
) -> Tuple[pd.DataFrame, list[str]]:
    if config.polygon_trades_file:
        logger.info("Loading Polygon trades from file: %s", config.polygon_trades_file)
        return _load_trades_from_file(config.polygon_trades_file), []

    max_workers = int(os.getenv("POLYGON_MAX_WORKERS", "4"))
    failures = []
    frames = []

    def fetch_one(symbol: str) -> Tuple[str, pd.DataFrame | None, Exception | None]:
        try:
            logger.info("Fetching Polygon trades for %s on %s", symbol, trade_date.isoformat())
            df = _fetch_trades_for_symbol(config, symbol, trade_date)
            return symbol, df if not df.empty else None, None
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                logger.info("Trades 403 for %s, falling back to aggregates", symbol)
                try:
                    df = _fetch_aggregates_for_symbol(config, symbol, trade_date)
                    return symbol, df if not df.empty else None, None
                except Exception as agg_exc:
                    return symbol, None, agg_exc
            return symbol, None, exc
        except Exception as exc:  # pylint: disable=broad-except
            return symbol, None, exc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            symbol, df, exc = future.result()
            if exc:
                logger.warning("Polygon fetch failed for %s: %s", symbol, exc)
                failures.append(symbol)
            elif df is not None:
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "price", "size", "bid", "ask"]), failures

    combined = pd.concat(frames, ignore_index=True)
    return combined, failures
